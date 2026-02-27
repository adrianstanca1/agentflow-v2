"""
AgentFlow v2 — Standalone Demo Server
Runs the full API + serves the React build in a single process.
Works without Docker, Ollama, or any external services.
API keys in .env unlock cloud models on the fly.
"""
import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import structlog
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

logger = structlog.get_logger()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
FRONTEND_DIST = HERE / "frontend" / "dist"

# ── Env ────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(HERE / ".env")

OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")
GROQ_KEY       = os.getenv("GROQ_API_KEY", "")
GEMINI_KEY     = os.getenv("GEMINI_API_KEY", "")
TOGETHER_KEY   = os.getenv("TOGETHER_API_KEY", "")
MISTRAL_KEY    = os.getenv("MISTRAL_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── Ollama helpers ─────────────────────────────────────────────────────────────
async def ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False

async def ollama_models() -> List[Dict]:
    try:
        async with httpx.AsyncClient(timeout=4.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                return r.json().get("models", [])
    except Exception:
        pass
    return []

async def ollama_running() -> List[Dict]:
    try:
        async with httpx.AsyncClient(timeout=4.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/ps")
            if r.status_code == 200:
                return r.json().get("models", [])
    except Exception:
        pass
    return []

async def ollama_pull_stream(model_name: str) -> AsyncGenerator[Dict, None]:
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", f"{OLLAMA_URL}/api/pull",
                            json={"name": model_name, "stream": True}) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    total = data.get("total", 0) or 0
                    completed = data.get("completed", 0) or 0
                    percent = (completed / total * 100) if total > 0 else 0
                    yield {"status": data.get("status",""), "percent": round(percent, 1),
                           "total": total, "completed": completed, "model": model_name}
                except Exception:
                    pass

# ── Cloud provider catalog (inline — no file deps) ─────────────────────────────
CLOUD_CATALOG = [
    # Anthropic
    {"id":"claude-opus-4-5","name":"Claude Opus 4.5","provider":"anthropic","provider_name":"Anthropic","provider_icon":"🟠","provider_color":"from-orange-500 to-amber-500","context_length":200000,"input_cost":15.0,"output_cost":75.0,"category":"reasoning","tags":["powerful","latest"],"description":"Most capable Claude model","is_latest":True},
    {"id":"claude-sonnet-4-5","name":"Claude Sonnet 4.5","provider":"anthropic","provider_name":"Anthropic","provider_icon":"🟠","provider_color":"from-orange-500 to-amber-500","context_length":200000,"input_cost":3.0,"output_cost":15.0,"category":"chat","tags":["fast","balanced"],"description":"Best balance of speed and intelligence","is_latest":True},
    {"id":"claude-haiku-4-5-20251001","name":"Claude Haiku 4.5","provider":"anthropic","provider_name":"Anthropic","provider_icon":"🟠","provider_color":"from-orange-500 to-amber-500","context_length":200000,"input_cost":0.25,"output_cost":1.25,"category":"chat","tags":["fast","cheap"],"description":"Fastest, most affordable Claude","is_latest":False},
    # OpenAI
    {"id":"gpt-4o","name":"GPT-4o","provider":"openai","provider_name":"OpenAI","provider_icon":"🟢","provider_color":"from-green-500 to-emerald-500","context_length":128000,"input_cost":5.0,"output_cost":15.0,"category":"chat","tags":["vision","fast"],"description":"OpenAI multimodal flagship","is_latest":True},
    {"id":"gpt-4o-mini","name":"GPT-4o Mini","provider":"openai","provider_name":"OpenAI","provider_icon":"🟢","provider_color":"from-green-500 to-emerald-500","context_length":128000,"input_cost":0.15,"output_cost":0.6,"category":"chat","tags":["cheap","fast"],"description":"Fast and affordable","is_latest":False},
    {"id":"o4-mini","name":"o4-mini","provider":"openai","provider_name":"OpenAI","provider_icon":"🟢","provider_color":"from-green-500 to-emerald-500","context_length":200000,"input_cost":1.1,"output_cost":4.4,"category":"reasoning","tags":["reasoning","fast"],"description":"Fast reasoning model","is_latest":True},
    # Groq
    {"id":"llama-3.3-70b-versatile","name":"Llama 3.3 70B","provider":"groq","provider_name":"Groq","provider_icon":"⚡","provider_color":"from-yellow-400 to-orange-400","context_length":128000,"input_cost":0.59,"output_cost":0.79,"category":"chat","tags":["fast","open"],"description":"Fast open-source via Groq","is_latest":True},
    {"id":"llama-3.1-8b-instant","name":"Llama 3.1 8B Instant","provider":"groq","provider_name":"Groq","provider_icon":"⚡","provider_color":"from-yellow-400 to-orange-400","context_length":128000,"input_cost":0.05,"output_cost":0.08,"category":"chat","tags":["ultra-fast","cheap"],"description":"Sub-100ms responses","is_latest":False},
    {"id":"deepseek-r1-distill-llama-70b","name":"DeepSeek R1 70B","provider":"groq","provider_name":"Groq","provider_icon":"⚡","provider_color":"from-yellow-400 to-orange-400","context_length":128000,"input_cost":0.75,"output_cost":0.99,"category":"reasoning","tags":["reasoning","fast"],"description":"Open reasoning model via Groq","is_latest":False},
    # Gemini
    {"id":"gemini-2.0-flash","name":"Gemini 2.0 Flash","provider":"gemini","provider_name":"Google AI","provider_icon":"🔵","provider_color":"from-blue-500 to-cyan-500","context_length":1048576,"input_cost":0.1,"output_cost":0.4,"category":"chat","tags":["fast","vision","latest"],"description":"Google's latest fast model","is_latest":True},
    {"id":"gemini-1.5-pro","name":"Gemini 1.5 Pro","provider":"gemini","provider_name":"Google AI","provider_icon":"🔵","provider_color":"from-blue-500 to-cyan-500","context_length":2097152,"input_cost":3.5,"output_cost":10.5,"category":"chat","tags":["long-ctx","vision"],"description":"2M context window","is_latest":False},
    # Together
    {"id":"meta-llama/Llama-3.3-70B-Instruct-Turbo","name":"Llama 3.3 70B Turbo","provider":"together","provider_name":"Together AI","provider_icon":"🟣","provider_color":"from-violet-500 to-purple-500","context_length":131072,"input_cost":0.88,"output_cost":0.88,"category":"chat","tags":["open","fast"],"description":"Meta's latest, fast","is_latest":True},
    {"id":"deepseek-ai/DeepSeek-R1","name":"DeepSeek R1","provider":"together","provider_name":"Together AI","provider_icon":"🟣","provider_color":"from-violet-500 to-purple-500","context_length":163840,"input_cost":3.0,"output_cost":7.0,"category":"reasoning","tags":["reasoning","open"],"description":"Open reasoning model","is_latest":True},
    {"id":"Qwen/Qwen2.5-Coder-32B-Instruct","name":"Qwen2.5 Coder 32B","provider":"together","provider_name":"Together AI","provider_icon":"🟣","provider_color":"from-violet-500 to-purple-500","context_length":32768,"input_cost":0.8,"output_cost":0.8,"category":"code","tags":["coding","open"],"description":"Best open coding model","is_latest":True},
    # Mistral
    {"id":"mistral-large-latest","name":"Mistral Large","provider":"mistral","provider_name":"Mistral AI","provider_icon":"🔴","provider_color":"from-red-500 to-pink-500","context_length":128000,"input_cost":2.0,"output_cost":6.0,"category":"chat","tags":["powerful","european"],"description":"Mistral's flagship","is_latest":True},
    {"id":"codestral-latest","name":"Codestral","provider":"mistral","provider_name":"Mistral AI","provider_icon":"🔴","provider_color":"from-red-500 to-pink-500","context_length":256000,"input_cost":0.2,"output_cost":0.6,"category":"code","tags":["coding","256k"],"description":"Best-in-class coding model","is_latest":True},
]

POPULAR_OLLAMA_MODELS = [
    {"name":"llama3.2:latest","category":"general","size":"2GB","description":"Meta's Llama 3.2 — best small general model","tags":["fast","recommended"]},
    {"name":"llama3.1:8b","category":"general","size":"4.7GB","description":"Llama 3.1 8B — excellent all-rounder","tags":["balanced","popular"]},
    {"name":"qwen2.5-coder:7b","category":"coding","size":"4.7GB","description":"Best open-source coding model","tags":["code","recommended"]},
    {"name":"deepseek-r1:8b","category":"reasoning","size":"4.7GB","description":"Local reasoning model","tags":["reasoning"]},
    {"name":"mistral:7b","category":"general","size":"4.1GB","description":"Mistral 7B — efficient and capable","tags":["popular"]},
    {"name":"phi4:latest","category":"general","size":"8.9GB","description":"Microsoft Phi-4 — strong reasoning","tags":["microsoft","reasoning"]},
    {"name":"nomic-embed-text","category":"embedding","size":"274MB","description":"High-quality text embeddings","tags":["embedding","recommended"]},
    {"name":"llava:7b","category":"vision","size":"4.7GB","description":"Vision-language model","tags":["vision","multimodal"]},
]

PROVIDER_META = {
    "anthropic":  {"name":"Anthropic",   "icon":"🟠","color":"from-orange-500 to-amber-500",  "docs":"https://docs.anthropic.com",    "key_env":"ANTHROPIC_API_KEY"},
    "openai":     {"name":"OpenAI",      "icon":"🟢","color":"from-green-500 to-emerald-500", "docs":"https://platform.openai.com",   "key_env":"OPENAI_API_KEY"},
    "groq":       {"name":"Groq",        "icon":"⚡","color":"from-yellow-400 to-orange-400", "docs":"https://console.groq.com",      "key_env":"GROQ_API_KEY"},
    "gemini":     {"name":"Google AI",   "icon":"🔵","color":"from-blue-500 to-cyan-500",     "docs":"https://ai.google.dev",         "key_env":"GEMINI_API_KEY"},
    "together":   {"name":"Together AI", "icon":"🟣","color":"from-violet-500 to-purple-500", "docs":"https://docs.together.ai",      "key_env":"TOGETHER_API_KEY"},
    "mistral":    {"name":"Mistral AI",  "icon":"🔴","color":"from-red-500 to-pink-500",      "docs":"https://docs.mistral.ai",       "key_env":"MISTRAL_API_KEY"},
    "openrouter": {"name":"OpenRouter",  "icon":"🔗","color":"from-slate-500 to-gray-500",    "docs":"https://openrouter.ai/docs",    "key_env":"OPENROUTER_API_KEY"},
}

KEY_MAP = {
    "anthropic":  ANTHROPIC_KEY,
    "openai":     OPENAI_KEY,
    "groq":       GROQ_KEY,
    "gemini":     GEMINI_KEY,
    "together":   TOGETHER_KEY,
    "mistral":    MISTRAL_KEY,
    "openrouter": OPENROUTER_KEY,
}

AGENT_TYPES = [
    {"id":"assistant",   "name":"Assistant",    "description":"General purpose: questions, explanations, analysis","type":"assistant",   "model":"auto","prefer_local":False,"tools":["web_search","knowledge_base_search"]},
    {"id":"coding",      "name":"Coding Agent", "description":"Code generation, debugging, testing — runs code in sandbox","type":"coding",     "model":"auto","prefer_local":False,"tools":["execute_code","write_file","read_file","run_shell","web_search"]},
    {"id":"research",    "name":"Research Agent","description":"Deep web research, academic papers, knowledge synthesis","type":"research",   "model":"auto","prefer_local":False,"tools":["web_search","fetch_url","search_arxiv","knowledge_base_search"]},
    {"id":"data_analyst","name":"Data Analyst",  "description":"Statistical analysis, SQL queries, business insights","type":"data_analyst","model":"auto","prefer_local":False,"tools":["analyze_data","sql_query","execute_code"]},
    {"id":"devops",      "name":"DevOps Agent",  "description":"Docker, Kubernetes, CI/CD, infrastructure automation","type":"devops",     "model":"auto","prefer_local":False,"tools":["docker_command","generate_dockerfile","generate_k8s","run_shell"]},
    {"id":"writer",      "name":"Writer Agent",  "description":"Articles, blog posts, marketing copy, technical docs","type":"writer",     "model":"auto","prefer_local":False,"tools":["web_search","fetch_url"]},
    {"id":"sql",         "name":"SQL Agent",     "description":"SQL queries, schema design, query optimization","type":"sql",        "model":"auto","prefer_local":False,"tools":["sql_query","analyze_data","execute_code"]},
    {"id":"qa",          "name":"QA Agent",      "description":"Test generation, code review, quality assurance","type":"qa",         "model":"auto","prefer_local":False,"tools":["generate_tests","execute_code","read_file"]},
]

# ── LLM routing ────────────────────────────────────────────────────────────────
def _detect_provider(model_id: str) -> Optional[str]:
    if ":" in model_id or model_id.startswith("ollama/"):
        return "ollama"
    if model_id.startswith("claude-"):   return "anthropic"
    if model_id.startswith(("gpt-","o1-","o3","o4-","text-embedding")): return "openai"
    if model_id.startswith("gemini-"):   return "gemini"
    if model_id.startswith(("codestral","mistral-","pixtral")): return "mistral"
    if any(model_id.startswith(p) for p in ["llama-3","deepseek-r1-distill","gemma2-","mixtral-"]): return "groq"
    if "/" in model_id:
        org = model_id.split("/")[0].lower()
        if any(org.startswith(k) for k in ["meta-llama","deepseek-ai","qwen","mistralai","togethercomputer"]): return "together"
    return None

def _build_llm(model: str, temperature: float = 0.7, tools: list = None):
    """Build an LLM that routes directly to the right provider."""
    from langchain_openai import ChatOpenAI

    provider = _detect_provider(model)

    if provider == "ollama":
        model_name = model.replace("ollama/", "")
        llm = ChatOpenAI(model=model_name, temperature=temperature, base_url=f"{OLLAMA_URL}/v1", api_key="ollama")

    elif provider == "anthropic" and ANTHROPIC_KEY:
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model, temperature=temperature, anthropic_api_key=ANTHROPIC_KEY)
        except ImportError:
            llm = ChatOpenAI(model=model, temperature=temperature, base_url="https://api.anthropic.com/v1", api_key=ANTHROPIC_KEY)

    elif provider == "groq" and GROQ_KEY:
        llm = ChatOpenAI(model=model, temperature=temperature, base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)

    elif provider == "together" and TOGETHER_KEY:
        llm = ChatOpenAI(model=model, temperature=temperature, base_url="https://api.together.xyz/v1", api_key=TOGETHER_KEY)

    elif provider == "mistral" and MISTRAL_KEY:
        llm = ChatOpenAI(model=model, temperature=temperature, base_url="https://api.mistral.ai/v1", api_key=MISTRAL_KEY)

    elif provider == "openai" and OPENAI_KEY:
        llm = ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_KEY)

    elif provider == "gemini" and GEMINI_KEY:
        # Use via OpenAI-compat endpoint
        llm = ChatOpenAI(model=model, temperature=temperature,
                         base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                         api_key=GEMINI_KEY)
    else:
        # No key or unknown — try Ollama with whatever model string was given
        llm = ChatOpenAI(model=model, temperature=temperature, base_url=f"{OLLAMA_URL}/v1", api_key="ollama")

    if tools:
        return llm.bind_tools(tools)
    return llm

async def _run_agent_stream(model: str, task: str, agent_key: str, session_id: str) -> AsyncGenerator[str, None]:
    """Simple streaming agent runner — yields SSE data lines."""
    from langchain_core.messages import HumanMessage, SystemMessage

    SYSTEM_PROMPTS = {
        "coding":      "You are an expert software engineer. Write clean, tested, production-quality code. Think step-by-step.",
        "research":    "You are a world-class research analyst. Synthesize information from multiple sources. Cite everything.",
        "data_analyst":"You are an expert data analyst. Turn raw data into actionable insights with clear visualizations.",
        "devops":      "You are a senior DevOps engineer. Design production infrastructure following best practices.",
        "writer":      "You are a versatile, talented writer. Create compelling, well-structured content.",
        "sql":         "You are an expert SQL and database engineer. Write optimized, well-explained queries.",
        "qa":          "You are a senior QA engineer. Write comprehensive tests covering edge cases and error conditions.",
        "assistant":   "You are a highly capable AI assistant. Be concise, accurate, and genuinely helpful.",
    }

    system_prompt = SYSTEM_PROMPTS.get(agent_key, SYSTEM_PROMPTS["assistant"])
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=task)]

    yield f"data: {json.dumps({'event_type': 'started', 'content': f'Running {agent_key} agent...', 'agent_id': agent_key, 'agent_name': agent_key, 'session_id': session_id})}\n\n"

    try:
        llm = _build_llm(model, temperature=0.7)
        full_content = ""
        async for chunk in llm.astream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                full_content += token
                yield f"data: {json.dumps({'event_type': 'stream_token', 'content': token, 'agent_id': agent_key, 'agent_name': agent_key, 'session_id': session_id})}\n\n"

        yield f"data: {json.dumps({'event_type': 'output', 'content': full_content, 'agent_id': agent_key, 'agent_name': agent_key, 'session_id': session_id})}\n\n"
        yield f"data: {json.dumps({'event_type': 'complete', 'content': full_content, 'agent_id': agent_key, 'agent_name': agent_key, 'session_id': session_id})}\n\n"

    except Exception as e:
        err = str(e)
        logger.error("Agent run failed", error=err, model=model)
        yield f"data: {json.dumps({'event_type': 'error', 'content': err, 'agent_id': agent_key, 'agent_name': agent_key, 'session_id': session_id})}\n\n"

# ── App ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    online = await ollama_available()
    cloud_keys = [k for k, v in KEY_MAP.items() if v]
    logger.info("AgentFlow v2 starting",
                ollama=online, cloud_providers=cloud_keys,
                mode="hybrid" if cloud_keys else ("local" if online else "demo"))
    yield

app = FastAPI(title="AgentFlow v2", version="2.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Request models ─────────────────────────────────────────────────────────────
class RunAgentReq(BaseModel):
    agent_key: str = "assistant"; task: str
    session_id: Optional[str] = None; stream: bool = True
    model_override: Optional[str] = None; prefer_local: bool = False

class PullModelReq(BaseModel):
    model_name: str; host_url: Optional[str] = None

class AddHostReq(BaseModel):
    url: str; name: Optional[str] = None; host_type: str = "remote"; api_key: Optional[str] = None

class SwitchModelReq(BaseModel):
    model: str; agent_key: Optional[str] = None; prefer_local: bool = False

class IngestReq(BaseModel):
    content: str; collection: str = "research"; metadata: Dict[str, Any] = Field(default_factory=dict)

class AddMCPReq(BaseModel):
    name: str; transport: str = "stdio"; command: Optional[str] = None
    args: List[str] = Field(default_factory=list); url: Optional[str] = None

# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    online = await ollama_available()
    configured = [k for k, v in KEY_MAP.items() if v]
    return {
        "status": "healthy", "version": "2.0.0",
        "agents": [a["id"] for a in AGENT_TYPES],
        "ollama": {"healthy_hosts": 1 if online else 0, "total_hosts": 1, "total_models": len(await ollama_models())},
        "mode": "hybrid" if configured else ("local" if online else "demo"),
        "timestamp": time.time(),
    }

# ── Agents ─────────────────────────────────────────────────────────────────────
@app.get("/agents")
async def list_agents():
    online = await ollama_available()
    configured = [k for k, v in KEY_MAP.items() if v]
    default_model = (
        await ollama_models()
    )
    dm = default_model[0].get("name", "llama3.2:latest") if default_model else (
        "claude-haiku-4-5-20251001" if ANTHROPIC_KEY else
        "llama-3.1-8b-instant" if GROQ_KEY else
        "gpt-4o-mini" if OPENAI_KEY else "demo-mode"
    )
    return [{**a, "model": dm} for a in AGENT_TYPES]


@app.post("/agents/run")
async def run_agent(req: RunAgentReq):
    session_id = req.session_id or str(uuid.uuid4())
    # Resolve model
    if req.model_override:
        model = req.model_override
    else:
        # Pick best available model
        ollama_list = await ollama_models()
        if req.prefer_local and ollama_list:
            model = ollama_list[0]["name"]
        elif ANTHROPIC_KEY:
            model = "claude-haiku-4-5-20251001"
        elif GROQ_KEY:
            model = "llama-3.3-70b-versatile"
        elif OPENAI_KEY:
            model = "gpt-4o-mini"
        elif ollama_list:
            model = ollama_list[0]["name"]
        else:
            # No model available at all
            err_msg = "No models available. Install Ollama and pull a model, or add a cloud API key (.env)."
            if not req.stream:
                raise HTTPException(503, err_msg)
            async def no_model():
                evt = {"event_type":"error","content":err_msg,"agent_id":req.agent_key,"agent_name":req.agent_key,"session_id":session_id}
                yield f"data: {json.dumps(evt)}\n\n"
            return EventSourceResponse(no_model())

    if not req.stream:
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = _build_llm(model, temperature=0.7)
        result = await llm.ainvoke([HumanMessage(content=req.task)])
        return {"output": result.content, "model": model, "session_id": session_id, "status": "completed"}

    return EventSourceResponse(_run_agent_stream(model, req.task, req.agent_key, session_id))


@app.post("/agents/switch-model")
async def switch_model(req: SwitchModelReq):
    return {"message": f"Model preference set to {req.model}", "note": "Applies to next request"}


# ── Ollama ─────────────────────────────────────────────────────────────────────
@app.get("/ollama/hosts")
async def ollama_hosts():
    online = await ollama_available()
    models = await ollama_models()
    return {"hosts": [{"url": OLLAMA_URL, "name": "Local", "type": "local", "is_healthy": online,
                       "latency_ms": 0, "model_count": len(models), "gpu_count": 0}],
            "total_hosts": 1, "healthy_hosts": 1 if online else 0,
            "total_models": len(models)}

@app.post("/ollama/hosts")
async def add_ollama_host(req: AddHostReq):
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{req.url}/api/tags")
            healthy = r.status_code == 200
    except Exception:
        healthy = False
    return {"url": req.url, "healthy": healthy, "models": 0}

@app.delete("/ollama/hosts/{host_url:path}")
async def remove_host(host_url: str):
    return {"removed": host_url}

@app.post("/ollama/hosts/{host_url:path}/check")
async def check_host(host_url: str):
    return {"url": host_url, "healthy": await ollama_available(), "latency_ms": 0, "models": 0}

@app.get("/ollama/models")
async def list_ollama_models(host_url: Optional[str] = Query(None)):
    models = await ollama_models()
    return [{"name": m.get("name",""), "size": _human_size(m.get("size",0)),
             "size_bytes": m.get("size",0), "family": m.get("details",{}).get("family",""),
             "host_url": OLLAMA_URL, "digest": m.get("digest","")[:12]} for m in models]

@app.get("/ollama/models/running")
async def running_models():
    return await ollama_running()

@app.get("/ollama/models/popular")
async def popular_models():
    return POPULAR_OLLAMA_MODELS

@app.post("/ollama/models/pull")
async def pull_model(req: PullModelReq):
    if not await ollama_available():
        async def err():
            yield {"data": json.dumps({"status": "error", "error": "Ollama not running"}), "event": "progress"}
        return EventSourceResponse(err())

    async def stream():
        async for p in ollama_pull_stream(req.model_name):
            yield {"data": json.dumps(p), "event": "progress"}
    return EventSourceResponse(stream())

@app.delete("/ollama/models/{model_name:path}")
async def delete_model(model_name: str, host_url: Optional[str] = Query(None)):
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.delete(f"{OLLAMA_URL}/api/delete", json={"name": model_name})
            return {"deleted": model_name, "ok": r.status_code == 200}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/ollama/models/{model_name:path}/info")
async def model_info(model_name: str):
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{OLLAMA_URL}/api/show", json={"name": model_name})
            return r.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/ollama/chat")
async def ollama_chat(req: dict):
    model = req.get("model", "llama3.2:latest")
    messages = req.get("messages", [])
    stream = req.get("stream", True)
    if not stream:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{OLLAMA_URL}/api/chat", json={"model": model, "messages": messages, "stream": False})
            return r.json()
    async def gen():
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", f"{OLLAMA_URL}/api/chat", json={"model": model, "messages": messages, "stream": True}) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield {"data": line, "event": "chunk"}
    return EventSourceResponse(gen())

# ── Cloud providers ────────────────────────────────────────────────────────────
def _configured(provider_id: str) -> bool:
    return bool(KEY_MAP.get(provider_id))

@app.get("/cloud/providers")
async def list_cloud_providers():
    return [
        {"id": pid, "name": meta["name"], "icon": meta["icon"], "color": meta["color"],
         "docs": meta["docs"], "configured": _configured(pid),
         "healthy": None, "latency_ms": None, "error": None, "model_count": sum(1 for m in CLOUD_CATALOG if m["provider"] == pid)}
        for pid, meta in PROVIDER_META.items()
    ]

@app.post("/cloud/providers/check")
async def check_all_providers():
    """Ping each configured provider to validate keys."""
    results = {}
    async with httpx.AsyncClient(timeout=8.0) as client:
        for pid, key in KEY_MAP.items():
            if not key:
                results[pid] = {"configured": False, "healthy": False}
                continue
            try:
                ping_urls = {
                    "anthropic": ("POST","https://api.anthropic.com/v1/messages", {"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"}, {"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}),
                    "openai":    ("GET","https://api.openai.com/v1/models", {"Authorization":f"Bearer {key}"}, None),
                    "groq":      ("GET","https://api.groq.com/openai/v1/models", {"Authorization":f"Bearer {key}"}, None),
                    "gemini":    ("GET",f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", {}, None),
                    "together":  ("GET","https://api.together.xyz/v1/models", {"Authorization":f"Bearer {key}"}, None),
                    "mistral":   ("GET","https://api.mistral.ai/v1/models", {"Authorization":f"Bearer {key}"}, None),
                    "openrouter":("GET","https://openrouter.ai/api/v1/models", {"Authorization":f"Bearer {key}"}, None),
                }
                if pid not in ping_urls:
                    results[pid] = {"configured": True, "healthy": None}
                    continue
                method, url, headers, body = ping_urls[pid]
                start = time.time()
                if method == "GET":
                    r = await client.get(url, headers=headers)
                else:
                    r = await client.post(url, headers=headers, json=body)
                latency = (time.time() - start) * 1000
                healthy = r.status_code in (200, 400)  # 400 = valid key, bad request
                results[pid] = {"configured": True, "healthy": healthy, "latency_ms": round(latency, 1)}
            except Exception as e:
                results[pid] = {"configured": True, "healthy": False, "error": str(e)[:80]}
    return results

@app.post("/cloud/providers/{provider_id}/check")
async def check_single_provider(provider_id: str):
    all_results = await check_all_providers()
    r = all_results.get(provider_id, {"configured": False, "healthy": False})
    return {"provider": provider_id, **r}

@app.get("/cloud/models")
async def list_cloud_models(provider: Optional[str] = Query(None), only_configured: bool = Query(False), category: Optional[str] = Query(None)):
    configured = set(k for k, v in KEY_MAP.items() if v)
    models = CLOUD_CATALOG
    if provider:
        models = [m for m in models if m["provider"] == provider]
    if only_configured:
        models = [m for m in models if m["provider"] in configured]
    if category:
        models = [m for m in models if m["category"] == category]
    return [{**m,
             "configured": m["provider"] in configured,
             "cost_label": f"${m['input_cost']:.2f}/${m['output_cost']:.2f}/1M" if m["input_cost"] > 0 else "Free"
             } for m in models]

@app.get("/cloud/routing")
async def cloud_routing():
    configured = set(k for k, v in KEY_MAP.items() if v)
    return {m["id"]: {"provider": m["provider"], "configured": m["provider"] in configured} for m in CLOUD_CATALOG}

@app.post("/cloud/test")
async def test_cloud_model(model: str = Query(...), prompt: str = Query("Say hi in exactly 3 words.")):
    try:
        from langchain_core.messages import HumanMessage
        llm = _build_llm(model, temperature=0)
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        return {"model": model, "response": result.content, "success": True}
    except Exception as e:
        return {"model": model, "error": str(e), "success": False}

@app.get("/models/all")
async def all_models():
    configured = set(k for k, v in KEY_MAP.items() if v)
    raw = await ollama_models()
    local = [{"id": m.get("name",""), "name": m.get("name",""), "source": "ollama", "provider": "ollama",
              "provider_name": "Ollama", "provider_icon": "🦙", "provider_color": "from-green-500 to-teal-500",
              "size": _human_size(m.get("size",0)), "family": m.get("details",{}).get("family",""),
              "host_url": OLLAMA_URL, "category": "local", "configured": True, "cost_label": "Free (local)"} for m in raw]
    cloud = [{**m, "configured": m["provider"] in configured,
              "cost_label": f"${m['input_cost']:.2f}/${m['output_cost']:.2f}/1M" if m.get("input_cost",0) > 0 else "Free"
              } for m in CLOUD_CATALOG]
    return {"local": local, "cloud": cloud, "total": len(local) + len(cloud)}

# ── MCP stubs ──────────────────────────────────────────────────────────────────
MCP_POPULAR = [
    {"id":"filesystem","name":"Filesystem","icon":"📁","transport":"stdio","category":"system","description":"Read and write files","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/tmp"]},
    {"id":"github","name":"GitHub","icon":"🐙","transport":"stdio","category":"dev","description":"Repos, issues, PRs, code search","command":"npx","args":["-y","@modelcontextprotocol/server-github"],"env_keys":["GITHUB_TOKEN"]},
    {"id":"brave-search","name":"Brave Search","icon":"🦁","transport":"stdio","category":"search","description":"Web search via Brave API","command":"npx","args":["-y","@modelcontextprotocol/server-brave-search"],"env_keys":["BRAVE_API_KEY"]},
    {"id":"postgres","name":"PostgreSQL","icon":"🐘","transport":"stdio","category":"data","description":"Query databases","command":"npx","args":["-y","@modelcontextprotocol/server-postgres"],"env_keys":["DATABASE_URL"]},
    {"id":"puppeteer","name":"Puppeteer","icon":"🤖","transport":"stdio","category":"browser","description":"Browser automation, scraping","command":"npx","args":["-y","@modelcontextprotocol/server-puppeteer"]},
    {"id":"slack","name":"Slack","icon":"💬","transport":"stdio","category":"comms","description":"Send messages, list channels","command":"npx","args":["-y","@modelcontextprotocol/server-slack"],"env_keys":["SLACK_BOT_TOKEN"]},
    {"id":"google-drive","name":"Google Drive","icon":"📁","transport":"stdio","category":"storage","description":"Read files from Google Drive","command":"npx","args":["-y","@modelcontextprotocol/server-gdrive"],"env_keys":["GOOGLE_CREDENTIALS"]},
    {"id":"sequential-thinking","name":"Sequential Thinking","icon":"🧠","transport":"stdio","category":"reasoning","description":"Enhanced step-by-step reasoning","command":"npx","args":["-y","@modelcontextprotocol/server-sequential-thinking"]},
]
_mcp_servers: List[Dict] = []

@app.get("/mcp/servers")
async def list_mcp_servers(): return _mcp_servers

@app.get("/mcp/servers/popular")
async def popular_mcp_servers(): return MCP_POPULAR

@app.post("/mcp/servers")
async def add_mcp_server(req: AddMCPReq):
    srv = {"id": str(uuid.uuid4()), "name": req.name, "transport": req.transport,
           "status": "connected", "tool_count": 0, "tools": []}
    _mcp_servers.append(srv)
    return srv

@app.get("/mcp/tools")
async def list_mcp_tools(): return []

# ── Knowledge base stubs ───────────────────────────────────────────────────────
_kb: List[Dict] = []

@app.post("/knowledge/ingest")
async def ingest_doc(req: IngestReq):
    _kb.append({"content": req.content, "collection": req.collection, **req.metadata})
    return {"status": "ingested", "collection": req.collection, "total": len(_kb)}

@app.get("/knowledge/search")
async def search_kb(query: str = Query(...), collection: str = Query("research"), limit: int = Query(5)):
    q = query.lower()
    results = [{"score": 0.9, "payload": d} for d in _kb if q in d.get("content","").lower()][:limit]
    return results

@app.get("/knowledge/collections")
async def kb_collections():
    cols = list(set(d.get("collection","research") for d in _kb)) or ["research"]
    return [{"name": c} for c in cols]

# ── Stats ──────────────────────────────────────────────────────────────────────
@app.get("/stats")
async def stats():
    online = await ollama_available()
    configured = [k for k, v in KEY_MAP.items() if v]
    return {"agents": {"count": len(AGENT_TYPES), "available": [a["id"] for a in AGENT_TYPES]},
            "ollama": {"healthy_hosts": 1 if online else 0, "total_hosts": 1, "total_models": len(await ollama_models())},
            "mcp": {"servers": len(_mcp_servers), "tools": 0},
            "mode": "hybrid" if configured else ("local" if online else "demo"),
            "timestamp": time.time()}

@app.get("/platform/info")
async def platform_info():
    return {"name":"AgentFlow","version":"2.0.0","agents":AGENT_TYPES,
            "capabilities":["local_llm","cloud_llm","multi_agent","streaming","websocket","mcp","rag"],
            "llm_providers":{k: bool(v) for k,v in KEY_MAP.items()}}

# ── WebSocket ──────────────────────────────────────────────────────────────────
ws_clients: Dict[str, WebSocket] = {}

@app.websocket("/ws/{conn_id}")
async def ws_endpoint(ws: WebSocket, conn_id: str):
    await ws.accept(); ws_clients[conn_id] = ws
    try:
        while True:
            data = await ws.receive_json()
            if data.get("action") == "ping":
                await ws.send_json({"event_type": "pong", "ts": time.time()})
            elif data.get("action") == "run":
                model_list = await ollama_models()
                model = (model_list[0]["name"] if model_list else
                         "claude-haiku-4-5-20251001" if ANTHROPIC_KEY else
                         "llama-3.1-8b-instant" if GROQ_KEY else "gpt-4o-mini")
                sid = data.get("session_id", str(uuid.uuid4()))
                async def run_ws(m=model, t=data.get("task",""), k=data.get("agent_key","assistant"), s=sid):
                    async for chunk in _run_agent_stream(m, t, k, s):
                        if chunk.startswith("data: "):
                            try:
                                await ws.send_json(json.loads(chunk[6:]))
                            except Exception:
                                pass
                asyncio.create_task(run_ws())
    except WebSocketDisconnect:
        ws_clients.pop(conn_id, None)



# ── Helpers ────────────────────────────────────────────────────────────────────
def _human_size(n: int) -> str:
    for unit in ["B","KB","MB","GB"]:
        if n < 1024: return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


# ════════════════════════════════════════════════════════════
# OPENCLAW ENDPOINTS
# ════════════════════════════════════════════════════════════

class OpenClawRunReq(BaseModel):
    task: str
    model: Optional[str] = None
    cwd: str = "."
    session_id: Optional[str] = None
    stream: bool = True

# In-memory sessions (prod would use Redis)
_openclaw_sessions: Dict[str, Any] = {}

@app.post("/openclaw/run")
async def openclaw_run(req: OpenClawRunReq):
    """Run an OpenClaw agentic coding task with streaming SSE."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "openclaw"))
    
    try:
        from openclaw.core import AgentSession, list_models as oc_list_models
        from openclaw.config import detect_best_model
    except ImportError:
        async def err():
            msg = json.dumps({"event_type":"error","content":"OpenClaw not installed. Run: pip install -e openclaw/"})
            yield f"data: {msg}\n\n"
        return EventSourceResponse(err())

    sid = req.session_id or str(uuid.uuid4())
    
    # Resolve model
    model = req.model
    if not model:
        available = await oc_list_models(OLLAMA_URL)
        model = detect_best_model(available) or "qwen2.5-coder:7b"
    
    # Get or create session
    if sid not in _openclaw_sessions:
        cwd = str(Path(req.cwd).resolve()) if req.cwd != "." else os.getcwd()
        _openclaw_sessions[sid] = AgentSession(model=model, cwd=cwd)
    
    session: AgentSession = _openclaw_sessions[sid]
    session.model = model

    async def stream_run():
        queue: asyncio.Queue = asyncio.Queue()

        def on_token(token: str):
            queue.put_nowait({"event_type": "stream_token", "content": token, "session_id": sid})

        def on_tool(name: str, args: dict, result):
            if result is None:
                queue.put_nowait({"event_type": "tool_call", "content": {"tool": name, "args": args}, "session_id": sid})
            else:
                queue.put_nowait({"event_type": "tool_result", "content": {"tool": name, "result": result[:500]}, "session_id": sid})

        async def runner():
            try:
                result = await session.run(req.task, on_token=on_token, on_tool=on_tool)
                queue.put_nowait({"event_type": "complete", "content": result, "session_id": sid})
            except Exception as e:
                queue.put_nowait({"event_type": "error", "content": str(e), "session_id": sid})
            queue.put_nowait(None)  # sentinel

        asyncio.create_task(runner())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return EventSourceResponse(stream_run())


@app.get("/openclaw/sessions")
async def list_openclaw_sessions():
    return [{"session_id": sid, "model": s.model, "cwd": s.cwd, "turns": len(s.messages)//2}
            for sid, s in _openclaw_sessions.items()]


@app.delete("/openclaw/sessions/{session_id}")
async def clear_openclaw_session(session_id: str):
    if session_id in _openclaw_sessions:
        del _openclaw_sessions[session_id]
    return {"cleared": session_id}


@app.get("/openclaw/models")
async def openclaw_models():
    """Models suitable for coding tasks."""
    available = await ollama_models()
    available_names = {m.get("name","") for m in available}
    coding_priority = [
        "qwen2.5-coder:32b","deepseek-coder-v2:16b","qwen2.5-coder:14b",
        "deepseek-r1:14b","qwen2.5-coder:7b","deepseek-r1:8b","codestral:22b",
        "llama3.3:70b","llama3.1:8b","phi4:latest","mistral:7b","llama3.2:latest",
    ]
    result = []
    for m in coding_priority:
        result.append({"name": m, "installed": m in available_names,
                        "recommended": m in {"qwen2.5-coder:7b","qwen2.5-coder:14b","deepseek-r1:8b"}})
    # Add any other installed models not in priority list
    for m in available_names:
        if not any(r["name"] == m for r in result):
            result.append({"name": m, "installed": True, "recommended": False})
    return result


# ── OpenClaw PTY WebSocket Terminal ───────────────────────────────────────────
@app.websocket("/ws/terminal/{session_id}")
async def terminal_ws(ws: WebSocket, session_id: str,
                      cwd: str = "/", model: str = "qwen2.5-coder:7b"):
    await ws.accept()
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "openclaw"))
    try:
        from openclaw.terminal import pty_websocket_handler
        await pty_websocket_handler(ws, session_id, cwd or os.getcwd(), model)
    except ImportError as e:
        await ws.send_json({"type": "error", "message": f"Terminal not available: {e}"})
    except Exception as e:
        try: await ws.send_json({"type": "error", "message": str(e)})
        except: pass


@app.get("/openclaw/session/{session_id}/files")
async def session_files(session_id: str, path: str = "."):
    """List files for the file explorer panel."""
    cwd = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    if not os.path.exists(cwd):
        raise HTTPException(404, "Path not found")
    items = []
    skip = {"node_modules", "__pycache__", ".git", ".venv", "dist", "build"}
    try:
        for entry in sorted(os.scandir(cwd), key=lambda e: (e.is_file(), e.name.lower())):
            if entry.name.startswith(".") or entry.name in skip:
                continue
            stat = entry.stat()
            items.append({
                "name": entry.name,
                "path": entry.path,
                "is_dir": entry.is_dir(),
                "size": stat.st_size if entry.is_file() else 0,
                "modified": stat.st_mtime,
            })
    except PermissionError:
        pass
    return {"path": cwd, "items": items}


@app.get("/openclaw/session/{session_id}/read")
async def read_file_api(session_id: str, path: str):
    """Read a file for the editor panel."""
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    if os.path.getsize(path) > 500_000:
        raise HTTPException(413, "File too large (>500KB)")
    try:
        content = open(path, errors="replace").read()
        ext = os.path.splitext(path)[1].lstrip(".")
        return {"path": path, "content": content, "extension": ext,
                "lines": content.count("\n") + 1}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Serve React frontend ───────────────────────────────────────────────────────
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # API paths handled above; everything else → index.html
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return HTMLResponse("<h1>AgentFlow v2</h1><p>Frontend not built. Run: cd frontend && npm run build</p>")
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n{'='*56}")
    print(f"  AgentFlow v2  —  http://localhost:{port}")
    print(f"  API Docs      —  http://localhost:{port}/docs")
    print(f"{'='*56}")
    online_providers = [k for k, v in KEY_MAP.items() if v]
    if online_providers:
        print(f"  Cloud keys   : {', '.join(online_providers)}")
    else:
        print("  Mode         : Local only (add keys to .env for cloud)")
    print(f"{'='*56}\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
