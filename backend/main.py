"""AgentFlow v2 — Main API"""
import asyncio, json, time, uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional
import structlog
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from core.config import settings
from core.ollama_manager import ollama_manager, initialize_ollama, OllamaHostType, POPULAR_MODELS
from core.cloud_manager import cloud_manager, CLOUD_CATALOG, PROVIDER_META
from agents.specialists import registry, AGENT_CLASSES
from mcp.manager import mcp_manager, POPULAR_MCP_SERVERS

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AgentFlow v2 starting...")
    initialize_ollama()
    await ollama_manager.health_check_all()
    await ollama_manager.start_health_monitor(interval=30)
    has_cloud = bool(settings.anthropic_api_key or settings.openai_api_key)
    registry.initialize_defaults(prefer_local=not has_cloud)
    if settings.ollama_auto_pull_list:
        asyncio.create_task(_auto_pull_models())
    logger.info("Ready", agents=registry.keys(), mode="hybrid" if has_cloud else "local")
    yield
    await ollama_manager.close()

async def _auto_pull_models():
    await asyncio.sleep(8)
    for model in settings.ollama_auto_pull_list:
        async for p in ollama_manager.pull_model(model):
            if p.get("status") in ("success", "error"):
                break

app = FastAPI(title="AgentFlow v2", version="2.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins_list, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class WSManager:
    def __init__(self): self.connections: Dict[str, WebSocket] = {}
    async def connect(self, cid, ws):
        await ws.accept(); self.connections[cid] = ws
    async def disconnect(self, cid): self.connections.pop(cid, None)
    async def send(self, cid, data):
        ws = self.connections.get(cid)
        if ws:
            try: await ws.send_json(data)
            except: await self.disconnect(cid)

ws_manager = WSManager()

class RunAgentReq(BaseModel):
    agent_key: str; task: str; session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None; stream: bool = True
    model_override: Optional[str] = None; prefer_local: bool = False

class AddHostReq(BaseModel):
    url: str; name: Optional[str] = None; host_type: str = "remote"; api_key: Optional[str] = None

class PullModelReq(BaseModel):
    model_name: str; host_url: Optional[str] = None

class OllamaChatReq(BaseModel):
    model: str; messages: List[Dict[str, str]]; stream: bool = True
    host_url: Optional[str] = None; options: Optional[Dict] = None

class SwitchModelReq(BaseModel):
    model: str; agent_key: Optional[str] = None; prefer_local: bool = False

class AddMCPReq(BaseModel):
    name: str; transport: str = "stdio"; command: Optional[str] = None
    args: List[str] = Field(default_factory=list); url: Optional[str] = None
    api_key: Optional[str] = None; env: Dict[str, str] = Field(default_factory=dict)

class IngestReq(BaseModel):
    content: str; collection: str = "research"; metadata: Dict[str, Any] = Field(default_factory=dict)

# Health
@app.get("/health")
async def health():
    stats = await ollama_manager.get_stats()
    return {"status":"healthy","version":"2.0.0","agents":registry.keys(),"ollama":stats,
            "mode":"hybrid" if (settings.anthropic_api_key or settings.openai_api_key) else "local","timestamp":time.time()}

# Agents
@app.get("/agents")
async def list_agents(): return registry.list_agents()

@app.post("/agents/run")
async def run_agent(req: RunAgentReq):
    agent = registry.get(req.agent_key)
    if not agent: raise HTTPException(404, f"Agent '{req.agent_key}' not found. Available: {registry.keys()}")
    if req.model_override: agent.switch_model(req.model_override, req.prefer_local)
    sid = req.session_id or str(uuid.uuid4())
    if not req.stream:
        return await agent.run(task=req.task, session_id=sid, context=req.context)
    async def gen():
        async for evt in agent.stream(task=req.task, session_id=sid, context=req.context):
            yield {"data": json.dumps(evt.model_dump(mode="json")), "event": evt.event_type}
    return EventSourceResponse(gen())

@app.post("/agents/switch-model")
async def switch_model(req: SwitchModelReq):
    if req.agent_key:
        a = registry.get(req.agent_key)
        if not a: raise HTTPException(404, "Agent not found")
        a.switch_model(req.model, req.prefer_local)
        return {"message": f"Switched {req.agent_key} to {req.model}"}
    registry.switch_all_models(req.model, req.prefer_local)
    return {"message": f"All agents switched to {req.model}"}

# Ollama management
@app.get("/ollama/hosts")
async def get_hosts(): return await ollama_manager.get_stats()

@app.post("/ollama/hosts")
async def add_host(req: AddHostReq):
    ht = OllamaHostType(req.host_type) if req.host_type in ("local","remote","cloud") else OllamaHostType.REMOTE
    host = ollama_manager.add_host(req.url, req.name, ht, req.api_key)
    healthy = await ollama_manager.check_health(host)
    return {"url": req.url, "healthy": healthy, "models": len(host.models)}

@app.delete("/ollama/hosts/{host_url:path}")
async def remove_host(host_url: str):
    if host_url in ollama_manager.hosts:
        del ollama_manager.hosts[host_url]; return {"removed": host_url}
    raise HTTPException(404, "Host not found")

@app.post("/ollama/hosts/{host_url:path}/check")
async def check_host(host_url: str):
    host = ollama_manager.hosts.get(host_url)
    if not host: raise HTTPException(404, "Host not found")
    healthy = await ollama_manager.check_health(host)
    return {"url": host_url, "healthy": healthy, "latency_ms": host.latency_ms, "models": len(host.models)}

@app.get("/ollama/models")
async def list_models(host_url: Optional[str] = Query(None)):
    models = await ollama_manager.list_models(host_url)
    return [{"name":m.name,"size":m.size_label,"size_bytes":m.size_bytes,"family":m.family,
             "host_url":m.host_url,"digest":m.digest[:12] if m.digest else ""} for m in models]

@app.get("/ollama/models/running")
async def running_models(): return await ollama_manager.list_running()

@app.get("/ollama/models/popular")
async def popular_models(): return POPULAR_MODELS

@app.post("/ollama/models/pull")
async def pull_model(req: PullModelReq):
    async def gen():
        async for p in ollama_manager.pull_model(req.model_name, req.host_url):
            yield {"data": json.dumps(p), "event": "progress"}
            if p.get("status") in ("success","error"): break
    return EventSourceResponse(gen())

@app.delete("/ollama/models/{model_name:path}")
async def delete_model(model_name: str, host_url: Optional[str] = Query(None)):
    ok = await ollama_manager.delete_model(model_name, host_url)
    if not ok: raise HTTPException(500, "Delete failed")
    return {"deleted": model_name}

@app.get("/ollama/models/{model_name:path}/info")
async def model_info(model_name: str, host_url: Optional[str] = Query(None)):
    return await ollama_manager.show_model_info(model_name, host_url)

@app.post("/ollama/chat")
async def ollama_chat(req: OllamaChatReq):
    if not req.stream:
        result = {}
        async for c in ollama_manager.chat(req.model, req.messages, req.host_url, False, req.options): result = c
        return result
    async def gen():
        async for c in ollama_manager.chat(req.model, req.messages, req.host_url, True, req.options):
            yield {"data": json.dumps(c), "event": "chunk"}
            if c.get("done"): break
    return EventSourceResponse(gen())

# MCP
@app.get("/mcp/servers")
async def list_mcp(): return mcp_manager.get_server_list()

@app.get("/mcp/servers/popular")
async def popular_mcp(): return POPULAR_MCP_SERVERS

@app.post("/mcp/servers")
async def add_mcp(req: AddMCPReq):
    from mcp.manager import MCPTransport
    t = MCPTransport(req.transport) if req.transport in ("stdio","sse","http") else MCPTransport.STDIO
    srv = mcp_manager.register_server(req.name, t, req.command, req.args, req.url, req.api_key, req.env)
    await mcp_manager.connect_server(srv.id)
    return {"id":srv.id,"name":srv.name,"status":srv.status,"tools":len(srv.tools)}

@app.get("/mcp/tools")
async def list_mcp_tools(): return mcp_manager.list_all_tools()

# Knowledge Base
@app.post("/knowledge/ingest")
async def ingest(req: IngestReq):
    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        client = AsyncQdrantClient(url=settings.qdrant_url)
        try:
            embs = await ollama_manager.embed(settings.ollama_default_embedding, req.content)
            vector = embs[0] if embs else None; embed_size = len(vector) if vector else 0
        except Exception: vector = None
        if not vector:
            from fastembed import TextEmbedding
            vector = list(TextEmbedding("BAAI/bge-small-en-v1.5").embed([req.content]))[0].tolist(); embed_size = 384
        try:
            await client.create_collection(req.collection, vectors_config=VectorParams(size=embed_size, distance=Distance.COSINE))
        except Exception: pass
        await client.upsert(req.collection, [PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"content":req.content,**req.metadata})])
        return {"status":"ingested","collection":req.collection}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/knowledge/search")
async def search_kb(query: str = Query(...), collection: str = Query("research"), limit: int = Query(5)):
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=settings.qdrant_url)
        try:
            embs = await ollama_manager.embed(settings.ollama_default_embedding, query)
            vector = embs[0] if embs else None
        except Exception: vector = None
        if not vector:
            from fastembed import TextEmbedding
            vector = list(TextEmbedding("BAAI/bge-small-en-v1.5").embed([query]))[0].tolist()
        results = await client.search(collection, vector, limit=limit, with_payload=True)
        return [{"score":round(r.score,3),"payload":r.payload} for r in results]
    except Exception as e: return {"results":[],"error":str(e)}

@app.get("/knowledge/collections")
async def list_collections():
    try:
        from qdrant_client import AsyncQdrantClient
        c = AsyncQdrantClient(url=settings.qdrant_url)
        cols = await c.get_collections()
        return [{"name":col.name} for col in cols.collections]
    except Exception as e: return {"collections":[],"error":str(e)}

# WebSocket
@app.websocket("/ws/{conn_id}")
async def ws_endpoint(ws: WebSocket, conn_id: str):
    await ws_manager.connect(conn_id, ws)
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")
            if action == "run":
                agent = registry.get(data.get("agent_key","assistant"))
                if not agent:
                    await ws_manager.send(conn_id, {"event_type":"error","content":"Agent not found"}); continue
                sid = data.get("session_id", str(uuid.uuid4()))
                async def run_it(a=agent, s=sid, t=data.get("task","")):
                    async for evt in a.stream(task=t, session_id=s):
                        await ws_manager.send(conn_id, evt.model_dump(mode="json"))
                asyncio.create_task(run_it())
            elif action == "ollama_chat":
                model = data.get("model", settings.ollama_default_model)
                async def chat_it(m=model, msgs=data.get("messages",[])):
                    async for chunk in ollama_manager.chat(m, msgs, stream=True):
                        await ws_manager.send(conn_id, {"type":"ollama_chunk","data":chunk})
                asyncio.create_task(chat_it())
            elif action == "ping":
                await ws_manager.send(conn_id, {"event_type":"pong","ts":time.time()})
    except WebSocketDisconnect: await ws_manager.disconnect(conn_id)

# Stats
@app.get("/stats")
async def stats():
    os = await ollama_manager.get_stats()
    return {"agents":{"count":len(registry.keys()),"available":registry.keys()},
            "ollama":os,"mcp":{"servers":len(mcp_manager.servers),"tools":len(mcp_manager.list_all_tools())},
            "mode":"hybrid" if (settings.anthropic_api_key or settings.openai_api_key) else "local","timestamp":time.time()}

@app.get("/platform/info")
async def info():
    return {"name":"AgentFlow","version":"2.0.0","agents":registry.list_agents(),
            "capabilities":["local_llm","cloud_llm","multi_agent","code_execution","web_search","rag","mcp","streaming"],
            "llm_providers":{"local":bool(ollama_manager._get_best_host()),"anthropic":bool(settings.anthropic_api_key),
                             "openai":bool(settings.openai_api_key),"gemini":bool(settings.gemini_api_key)}}



# ════════════════════════════════════════════════════════════
# CLOUD PROVIDER ENDPOINTS
# ════════════════════════════════════════════════════════════

class AddCloudProviderReq(BaseModel):
    provider_id: str; name: str; base_url: str; api_key: str

@app.get("/cloud/providers")
async def list_cloud_providers():
    """List all cloud providers with config and health."""
    return cloud_manager.get_status_snapshot()

@app.post("/cloud/providers/check")
async def check_cloud_providers():
    """Health-check all configured providers concurrently."""
    results = await cloud_manager.check_all()
    return {pid: {"configured":s.configured,"healthy":s.healthy,"latency_ms":round(s.latency_ms,1),"error":s.error,"model_count":s.model_count} for pid,s in results.items()}

@app.post("/cloud/providers/{provider_id}/check")
async def check_single_provider(provider_id: str):
    """Health-check a single provider."""
    try:
        from core.cloud_manager import ProviderType; ptype = ProviderType(provider_id)
    except ValueError: raise HTTPException(400, f"Unknown provider: {provider_id}")
    s = await cloud_manager.check_provider(ptype)
    return {"provider":provider_id,"configured":s.configured,"healthy":s.healthy,"latency_ms":round(s.latency_ms,1),"error":s.error}

@app.post("/cloud/providers/custom")
async def add_custom_provider(req: AddCloudProviderReq):
    """Register a custom OpenAI-compatible endpoint."""
    cloud_manager.add_custom_provider(req.provider_id, req.name, req.base_url, req.api_key)
    return {"registered": req.provider_id}

@app.get("/cloud/models")
async def list_cloud_models(provider: Optional[str]=Query(None), only_configured: bool=Query(False), category: Optional[str]=Query(None)):
    """List cloud models, filtered by provider or category."""
    models = cloud_manager.get_all_cloud_models(only_configured=only_configured)
    if provider: models = [m for m in models if m.provider.value == provider]
    if category: models = [m for m in models if m.category == category]
    configured_providers = {p.value for p in cloud_manager.configured_providers()}
    return [{"id":m.id,"name":m.name,"provider":m.provider.value,
             "provider_name":PROVIDER_META.get(m.provider.value,{}).get("name",m.provider.value),
             "provider_icon":PROVIDER_META.get(m.provider.value,{}).get("icon","🤖"),
             "provider_color":PROVIDER_META.get(m.provider.value,{}).get("color","from-gray-500 to-gray-600"),
             "context_length":m.context_length,"cost_label":m.cost_label,
             "input_cost":m.input_cost_per_1m,"output_cost":m.output_cost_per_1m,
             "category":m.category,"tags":m.tags,"description":m.description,
             "is_latest":m.is_latest,"configured":m.provider.value in configured_providers} for m in models]

@app.get("/cloud/routing")
async def get_routing_table():
    """Full model routing table."""
    return cloud_manager.get_routing_table()

@app.post("/cloud/test")
async def test_cloud_model(model: str=Query(...), prompt: str=Query("Hello, respond in 5 words.")):
    """Test a cloud model end-to-end."""
    from agents.base import build_llm
    from langchain_core.messages import HumanMessage
    try:
        llm = build_llm(model, temperature=0, streaming=False)
        r = await llm.ainvoke([HumanMessage(content=prompt)])
        return {"model":model,"response":r.content,"success":True}
    except Exception as e:
        return {"model":model,"error":str(e),"success":False}

@app.get("/models/all")
async def list_all_models():
    """Unified model list: Ollama local + all cloud providers."""
    ollama_models = await ollama_manager.list_models()
    configured_cloud = {p.value for p in cloud_manager.configured_providers()}
    local = [{"id":m.name,"name":m.name,"source":"ollama","provider":"ollama",
              "provider_name":"Ollama","provider_icon":"🦙","provider_color":"from-green-500 to-teal-500",
              "size":m.size_label,"family":m.family,"host_url":m.host_url,
              "category":"local","configured":True,"cost_label":"Free (local)"} for m in ollama_models]
    cloud = [{"id":m.id,"name":m.name,"source":"cloud","provider":m.provider.value,
              "provider_name":PROVIDER_META.get(m.provider.value,{}).get("name",m.provider.value),
              "provider_icon":PROVIDER_META.get(m.provider.value,{}).get("icon","🤖"),
              "provider_color":PROVIDER_META.get(m.provider.value,{}).get("color","from-gray-500 to-gray-600"),
              "category":m.category,"context_length":m.context_length,"cost_label":m.cost_label,
              "tags":m.tags,"description":m.description,"is_latest":m.is_latest,
              "configured":m.provider.value in configured_cloud} for m in cloud_manager.get_all_cloud_models()]
    return {"local":local,"cloud":cloud,"total":len(local)+len(cloud)}

if __name__ == "__main__":
    import uvicorn; uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
