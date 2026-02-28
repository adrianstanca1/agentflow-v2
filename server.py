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

ENV_FILE   = HERE / ".env"

# ── Ollama host registry — supports multiple hosts, persisted in .env ──────────
_DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_DEFAULT_OLLAMA_KEY = os.getenv("OLLAMA_API_KEY", "")  # optional bearer token

def _load_ollama_hosts() -> List[Dict]:
    """Load Ollama host list from .env (JSON-encoded OLLAMA_HOSTS)."""
    load_dotenv(ENV_FILE, override=True)
    raw = os.getenv("OLLAMA_HOSTS", "")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    # Fall back to single legacy URL
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    key = os.getenv("OLLAMA_API_KEY", "")
    return [{"url": url, "name": "Default", "api_key": key, "enabled": True}]

def _save_ollama_hosts(hosts: List[Dict]):
    _write_env({"OLLAMA_HOSTS": json.dumps(hosts)})

def _get_primary_host() -> Dict:
    hosts = _load_ollama_hosts()
    enabled = [h for h in hosts if h.get("enabled", True)]
    return enabled[0] if enabled else {"url": "http://localhost:11434", "api_key": ""}

# Primary URL for backward compat
@property
def OLLAMA_URL_prop(self): return _get_primary_host()["url"]
OLLAMA_URL = _get_primary_host()["url"]  # snapshot at startup; use _get_primary_host() dynamically

def _ollama_headers(host: Dict = None) -> Dict:
    h = host or _get_primary_host()
    key = h.get("api_key", "")
    return {"Authorization": f"Bearer {key}"} if key else {}

async def _check_ollama_host(url: str, api_key: str = "") -> Dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=4.0) as c:
            r = await c.get(f"{url}/api/tags", headers=headers)
            latency = round((time.time()-start)*1000, 1)
            if r.status_code == 200:
                models = r.json().get("models", [])
                return {"healthy": True, "latency_ms": latency, "model_count": len(models), "models": models}
            return {"healthy": False, "latency_ms": latency, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"healthy": False, "latency_ms": 0, "error": str(e)[:80]}

# ── Dynamic key store — reads from .env, can be updated at runtime ─────────────
_KEY_NAMES = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "groq":       "GROQ_API_KEY",
    "gemini":     "GEMINI_API_KEY",
    "together":   "TOGETHER_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

def _load_keys() -> Dict[str, str]:
    """Load current API keys from env (re-reads .env each call so updates take effect)."""
    load_dotenv(ENV_FILE, override=True)
    return {pid: os.getenv(name, "") for pid, name in _KEY_NAMES.items()}

def get_key(provider: str) -> str:
    return _load_keys().get(provider, "")

def _get_key_map() -> Dict[str, str]:
    return _load_keys()

# Keep backward compat aliases (resolved lazily)
def _key(p): return get_key(p)
ANTHROPIC_KEY  = property(lambda self: get_key("anthropic"))
OPENAI_KEY     = property(lambda self: get_key("openai"))
GROQ_KEY       = property(lambda self: get_key("groq"))
GEMINI_KEY     = property(lambda self: get_key("gemini"))
TOGETHER_KEY   = property(lambda self: get_key("together"))
MISTRAL_KEY    = property(lambda self: get_key("mistral"))
OPENROUTER_KEY = property(lambda self: get_key("openrouter"))

def _write_env(updates: Dict[str, str]):
    """Write / update keys in the .env file without destroying existing content."""
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text().splitlines()
    else:
        lines = []

    for env_var, value in updates.items():
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{env_var}=") or line.startswith(f"{env_var} ="):
                lines[i] = f"{env_var}={value}"
                found = True
                break
        if not found:
            lines.append(f"{env_var}={value}")

    ENV_FILE.write_text("\n".join(lines) + "\n")
    # Reload into os.environ immediately
    load_dotenv(ENV_FILE, override=True)

# ── Ollama helpers ─────────────────────────────────────────────────────────────
async def ollama_available(host: Dict = None) -> bool:
    h = host or _get_primary_host()
    result = await _check_ollama_host(h["url"], h.get("api_key",""))
    return result["healthy"]

async def ollama_models(host: Dict = None) -> List[Dict]:
    h = host or _get_primary_host()
    headers = _ollama_headers(h)
    try:
        async with httpx.AsyncClient(timeout=4.0) as c:
            r = await c.get(f"{h['url']}/api/tags", headers=headers)
            if r.status_code == 200:
                return r.json().get("models", [])
    except Exception:
        pass
    return []

async def ollama_running(host: Dict = None) -> List[Dict]:
    h = host or _get_primary_host()
    headers = _ollama_headers(h)
    try:
        async with httpx.AsyncClient(timeout=4.0) as c:
            r = await c.get(f"{h['url']}/api/ps", headers=headers)
            if r.status_code == 200:
                return r.json().get("models", [])
    except Exception:
        pass
    return []

async def ollama_pull_stream(model_name: str, host: Dict = None) -> AsyncGenerator[Dict, None]:
    h = host or _get_primary_host()
    headers = _ollama_headers(h)
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", f"{h['url']}/api/pull",
                            headers=headers, json={"name": model_name, "stream": True}) as resp:
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

# KEY_MAP is now always computed live via _get_key_map()
KEY_MAP = _get_key_map()  # initial snapshot — use _get_key_map() for live values

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
    """Build an LLM that routes to the right provider using live API keys."""
    from langchain_openai import ChatOpenAI
    keys = _get_key_map()  # always fresh
    provider = _detect_provider(model)

    if provider == "ollama":
        h = _get_primary_host()
        ollama_key = h.get("api_key","") or "ollama"
        llm = ChatOpenAI(model=model.replace("ollama/",""), temperature=temperature,
                         base_url=f"{h['url']}/v1", api_key=ollama_key)
    elif provider == "anthropic" and keys.get("anthropic"):
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model, temperature=temperature,
                                anthropic_api_key=keys["anthropic"])
        except ImportError:
            llm = ChatOpenAI(model=model, temperature=temperature,
                             base_url="https://api.anthropic.com/v1", api_key=keys["anthropic"])
    elif provider == "groq" and keys.get("groq"):
        llm = ChatOpenAI(model=model, temperature=temperature,
                         base_url="https://api.groq.com/openai/v1", api_key=keys["groq"])
    elif provider == "together" and keys.get("together"):
        llm = ChatOpenAI(model=model, temperature=temperature,
                         base_url="https://api.together.xyz/v1", api_key=keys["together"])
    elif provider == "mistral" and keys.get("mistral"):
        llm = ChatOpenAI(model=model, temperature=temperature,
                         base_url="https://api.mistral.ai/v1", api_key=keys["mistral"])
    elif provider == "openai" and keys.get("openai"):
        llm = ChatOpenAI(model=model, temperature=temperature, api_key=keys["openai"])
    elif provider == "gemini":
        # Support both API key and Google OAuth2 credentials
        from dotenv import load_dotenv as _ld
        _ld(ENV_FILE, override=True)
        auth_method = os.getenv("GEMINI_AUTH_METHOD", "")
        if auth_method == "oauth" or (not keys.get("gemini") and os.getenv("GOOGLE_ACCESS_TOKEN")):
            # Use OAuth2 credentials via google-genai SDK wrapped in a thin LangChain adapter
            try:
                from google.oauth2.credentials import Credentials as GCreds
                from langchain_google_genai import ChatGoogleGenerativeAI
                import google.auth.transport.requests
                creds = GCreds(
                    token=os.getenv("GOOGLE_ACCESS_TOKEN",""),
                    refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN",""),
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=os.getenv("GOOGLE_CLIENT_ID",""),
                    client_secret=os.getenv("GOOGLE_CLIENT_SECRET",""),
                )
                llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, credentials=creds)
            except ImportError:
                # Fallback: use access token as bearer via OpenAI-compat endpoint
                token = os.getenv("GOOGLE_ACCESS_TOKEN","")
                llm = ChatOpenAI(model=model, temperature=temperature,
                                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                                 api_key=token or "oauth")
        elif keys.get("gemini"):
            llm = ChatOpenAI(model=model, temperature=temperature,
                             base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                             api_key=keys["gemini"])
        else:
            raise RuntimeError("Gemini not configured. Sign in with Google or add a GEMINI_API_KEY in Settings.")
    elif keys.get("openrouter"):
        llm = ChatOpenAI(model=model, temperature=temperature,
                         base_url="https://openrouter.ai/api/v1", api_key=keys["openrouter"])
    else:
        # Fallback to Ollama
        h = _get_primary_host()
        llm = ChatOpenAI(model=model, temperature=temperature,
                         base_url=f"{h['url']}/v1", api_key=h.get("api_key","") or "ollama")
    return llm.bind_tools(tools) if tools else llm

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

    yield f"data: {json.dumps({'event_type': 'started', 'content': f'Running {agent_key} agent...', 'agent_id': agent_key, 'agent_name': agent_key, 'session_id': session_id, 'model_used': model})}\n\n"

    try:
        llm = _build_llm(model, temperature=0.7)
        full_content = ""
        async for chunk in llm.astream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                full_content += token
                yield f"data: {json.dumps({'event_type': 'stream_token', 'content': token, 'agent_id': agent_key, 'agent_name': agent_key, 'session_id': session_id})}\n\n"

        yield f"data: {json.dumps({'event_type': 'output', 'content': full_content, 'agent_id': agent_key, 'agent_name': agent_key, 'session_id': session_id, 'model_used': model})}\n\n"
        yield f"data: {json.dumps({'event_type': 'complete', 'content': full_content, 'agent_id': agent_key, 'agent_name': agent_key, 'session_id': session_id, 'model_used': model})}\n\n"

    except Exception as e:
        err = str(e)
        logger.error("Agent run failed", error=err, model=model)
        yield f"data: {json.dumps({'event_type': 'error', 'content': err, 'agent_id': agent_key, 'agent_name': agent_key, 'session_id': session_id})}\n\n"

# ── App ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    online = await ollama_available()
    cloud_keys = [k for k, v in _get_key_map().items() if v]
    logger.info("AgentFlow v2 starting",
                ollama=online, cloud_providers=cloud_keys,
                mode="hybrid" if cloud_keys else ("local" if online else "demo"))
    yield

app = FastAPI(title="AgentFlow v2", version="2.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Auth routes (Google OAuth2, OpenAI setup) ──────────────────────────────────
try:
    from auth_providers import router as auth_router
    app.include_router(auth_router)
except ImportError as _auth_err:
    import logging
    logging.warning(f"auth_providers not loaded: {_auth_err}")

try:
    from github_integration import router as github_router
    app.include_router(github_router)
except ImportError as _gh_err:
    import logging
    logging.warning(f"github_integration not loaded: {_gh_err}")

try:
    from docker_integration import router as docker_router
    app.include_router(docker_router)
except ImportError as _dk_err:
    import logging
    logging.warning(f"docker_integration not loaded: {_dk_err}")

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
    configured = [k for k, v in _get_key_map().items() if v]
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
    configured = [k for k, v in _get_key_map().items() if v]
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
            # Try cloud providers via openclaw's detect_provider
            try:
                import sys as _s; _s.path.insert(0, str(HERE / "openclaw"))
                from openclaw.core import detect_provider, detect_model as _dm, _get_provider_config
                _prov = detect_provider()
                _cfg = _get_provider_config(_prov)
                if _cfg.get("resolved_api_key") and _prov != "ollama":
                    model = _dm(_prov)
                else:
                    raise ValueError("no cloud key")
            except Exception:
                err_msg = "No models available. Install Ollama and pull a model, or add a cloud API key in Settings."
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
    hosts = _load_ollama_hosts()
    result = []
    total_models = 0
    healthy = 0
    for h in hosts:
        check = await _check_ollama_host(h["url"], h.get("api_key",""))
        total_models += check.get("model_count", 0)
        if check["healthy"]: healthy += 1
        result.append({
            "url": h["url"], "name": h.get("name","Ollama"),
            "api_key": ("*" * 8) if h.get("api_key") else "",
            "has_key": bool(h.get("api_key")),
            "enabled": h.get("enabled", True),
            "is_healthy": check["healthy"],
            "latency_ms": check.get("latency_ms", 0),
            "model_count": check.get("model_count", 0),
            "error": check.get("error"),
        })
    return {"hosts": result, "total_hosts": len(result),
            "healthy_hosts": healthy, "total_models": total_models}

@app.post("/ollama/hosts")
async def add_ollama_host(req: AddHostReq):
    hosts = _load_ollama_hosts()
    # Avoid duplicates
    if any(h["url"] == req.url for h in hosts):
        raise HTTPException(400, f"Host {req.url} already exists")
    check = await _check_ollama_host(req.url, req.api_key or "")
    new_host = {"url": req.url, "name": req.name or req.url,
                "api_key": req.api_key or "", "enabled": True}
    hosts.append(new_host)
    _save_ollama_hosts(hosts)
    return {"url": req.url, "name": new_host["name"],
            "healthy": check["healthy"], "model_count": check.get("model_count",0),
            "error": check.get("error")}

@app.delete("/ollama/hosts/{host_url:path}")
async def remove_host(host_url: str):
    hosts = _load_ollama_hosts()
    original = len(hosts)
    hosts = [h for h in hosts if h["url"] != host_url]
    if len(hosts) < original:
        _save_ollama_hosts(hosts)
    return {"removed": host_url}

@app.put("/ollama/hosts/{host_url:path}")
async def update_host(host_url: str, req: AddHostReq):
    hosts = _load_ollama_hosts()
    for h in hosts:
        if h["url"] == host_url:
            if req.name: h["name"] = req.name
            if req.api_key is not None: h["api_key"] = req.api_key
            break
    else:
        raise HTTPException(404, "Host not found")
    _save_ollama_hosts(hosts)
    return {"updated": host_url}

@app.post("/ollama/hosts/{host_url:path}/check")
async def check_host(host_url: str):
    hosts = _load_ollama_hosts()
    host = next((h for h in hosts if h["url"] == host_url), {"url": host_url, "api_key": ""})
    result = await _check_ollama_host(host["url"], host.get("api_key",""))
    return {"url": host_url, **result}

@app.post("/ollama/hosts/{host_url:path}/set-primary")
async def set_primary_host(host_url: str):
    hosts = _load_ollama_hosts()
    # Move the selected host to front
    hosts = sorted(hosts, key=lambda h: 0 if h["url"]==host_url else 1)
    _save_ollama_hosts(hosts)
    return {"primary": host_url}

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
    if provider_id == "gemini":
        from dotenv import load_dotenv as _ld
        _ld(ENV_FILE, override=True)
        return bool(_get_key_map().get("gemini") or
                    (os.getenv("GEMINI_AUTH_METHOD") == "oauth" and os.getenv("GOOGLE_ACCESS_TOKEN")))
    return bool(_get_key_map().get(provider_id))

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
        for pid, key in _get_key_map().items():
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
    configured = set(k for k, v in _get_key_map().items() if v)
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
    configured = set(k for k, v in _get_key_map().items() if v)
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
    configured = set(k for k, v in _get_key_map().items() if v)
    primary = _get_primary_host()
    raw = await ollama_models(primary)
    local = [{"id": m.get("name",""), "name": m.get("name",""), "source": "ollama", "provider": "ollama",
              "provider_name": "Ollama", "provider_icon": "🦙", "provider_color": "from-green-500 to-teal-500",
              "size": _human_size(m.get("size",0)), "family": m.get("details",{}).get("family",""),
              "host_url": primary["url"], "category": "local", "configured": True, "cost_label": "Free (local)"} for m in raw]
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

@app.get("/knowledge/stats")
async def knowledge_stats():
    online = await ollama_available()
    collections = list(set(d.get("collection","research") for d in _kb)) or ["research"]
    return {
        "total_documents": len(_kb),
        "collections": collections,
        "vector_store": "in-memory (qdrant optional)",
        "status": "ready" if online else "degraded (Ollama offline)",
    }

# ── Conversation history (in-memory; swap for SQLite in production) ────────────
_conversations: list = []

class ConversationEntry(BaseModel):
    session_id: str; agent_key: str; task: str
    response: str = ""; model: str = ""; ts: str = ""

@app.get("/conversations")
async def list_conversations(limit: int = Query(50, ge=1, le=500),
                              agent_key: Optional[str] = Query(None)):
    convs = _conversations[-limit:]
    if agent_key:
        convs = [c for c in convs if c.get("agent_key") == agent_key]
    return list(reversed(convs))

@app.post("/conversations")
async def save_conversation(entry: ConversationEntry):
    import datetime
    record = entry.dict()
    record["ts"] = record.get("ts") or datetime.datetime.utcnow().isoformat()
    _conversations.append(record)
    if len(_conversations) > 1000:
        _conversations.pop(0)
    return {"saved": True, "id": len(_conversations) - 1}

@app.delete("/conversations")
async def clear_conversations():
    _conversations.clear()
    return {"cleared": True}

# ── Stats ──────────────────────────────────────────────────────────────────────
@app.get("/stats")
async def stats():
    online = await ollama_available()
    configured = [k for k, v in _get_key_map().items() if v]
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
# SETTINGS / API KEY MANAGEMENT ENDPOINTS
# ════════════════════════════════════════════════════════════

class SaveKeysReq(BaseModel):
    keys: Dict[str, str]  # provider_id -> api_key value

@app.get("/settings/keys")
async def get_api_keys():
    """Return current API keys — values masked for security."""
    keys = _get_key_map()
    result = {}
    for pid, meta in PROVIDER_META.items():
        raw = keys.get(pid, "")
        # Gemini may be connected via OAuth instead of API key
        if pid == "gemini":
            from dotenv import load_dotenv as _ld
            _ld(ENV_FILE, override=True)
            oauth_method = os.getenv("GEMINI_AUTH_METHOD","") == "oauth"
            oauth_email  = os.getenv("GOOGLE_USER_EMAIL","")
            oauth_name   = os.getenv("GOOGLE_USER_NAME","")
            is_configured = bool(raw) or oauth_method
            masked = (raw[:8]+"..."+raw[-4:]) if len(raw)>12 else ("*"*len(raw) if raw else "")
            result[pid] = {
                "provider": pid, "name": meta["name"], "icon": meta["icon"],
                "color": meta["color"], "docs": meta["docs"], "env_var": meta["key_env"],
                "configured": is_configured, "masked": masked, "key_length": len(raw),
                "auth_method": "oauth" if oauth_method else ("api_key" if raw else "none"),
                "oauth_email": oauth_email, "oauth_name": oauth_name,
                "get_key_url": "https://aistudio.google.com/app/apikey",
                "free_tier": True,
                "description": "Gemini 2.0 Flash, 1.5 Pro — sign in with Google (free) or use API key",
            }
            continue
        result[pid] = {
            "provider":    pid,
            "name":        meta["name"],
            "icon":        meta["icon"],
            "color":       meta["color"],
            "docs":        meta["docs"],
            "env_var":     meta["key_env"],
            "configured":  bool(raw),
            "masked":      (raw[:8] + "..." + raw[-4:]) if len(raw) > 12 else ("*" * len(raw) if raw else ""),
            "key_length":  len(raw),
            "get_key_url": {
                "anthropic":  "https://console.anthropic.com/settings/keys",
                "openai":     "https://platform.openai.com/api-keys",
                "groq":       "https://console.groq.com/keys",
                "gemini":     "https://aistudio.google.com/app/apikey",
                "together":   "https://api.together.xyz/settings/api-keys",
                "mistral":    "https://console.mistral.ai/api-keys/",
                "openrouter": "https://openrouter.ai/settings/keys",
            }.get(pid, "#"),
            "free_tier": pid in ("groq", "gemini", "together", "openrouter"),
            "description": {
                "anthropic":  "Claude models — Opus, Sonnet, Haiku",
                "openai":     "GPT-4o, o4-mini and all OpenAI models",
                "groq":       "Llama, DeepSeek at blazing speed — FREE tier",
                "gemini":     "Gemini 2.0 Flash, 1.5 Pro — FREE tier",
                "together":   "100+ open models at low cost — FREE $25 credit",
                "mistral":    "Mistral Large, Codestral",
                "openrouter": "300+ models from one key — FREE tier",
            }.get(pid, ""),
        }
    return result

@app.post("/settings/keys")
async def save_api_keys(req: SaveKeysReq):
    """Save one or more API keys to .env and reload immediately."""
    updates = {}
    saved = []
    for provider_id, key_value in req.keys.items():
        if provider_id not in _KEY_NAMES:
            continue
        env_var = _KEY_NAMES[provider_id]
        updates[env_var] = key_value.strip()
        saved.append(provider_id)

    if updates:
        _write_env(updates)

    # Return updated status
    keys = _get_key_map()
    return {
        "saved": saved,
        "configured": [k for k, v in keys.items() if v],
        "message": f"Saved {len(saved)} key(s). Active immediately — no restart needed.",
    }

@app.delete("/settings/keys/{provider_id}")
async def delete_api_key(provider_id: str):
    """Remove an API key."""
    if provider_id not in _KEY_NAMES:
        raise HTTPException(404, f"Unknown provider: {provider_id}")
    _write_env({_KEY_NAMES[provider_id]: ""})
    return {"deleted": provider_id, "message": "Key removed."}

@app.post("/settings/keys/{provider_id}/test")
async def test_api_key(provider_id: str):
    """Test an API key by making a minimal real request."""
    key = get_key(provider_id)
    if not key:
        return {"provider": provider_id, "success": False, "error": "No key configured",
                "latency_ms": 0}

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            test_configs = {
                "anthropic":  dict(method="POST", url="https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    json={"model": "claude-haiku-4-5-20251001", "max_tokens": 5, "messages": [{"role":"user","content":"hi"}]}),
                "openai":     dict(method="GET",  url="https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"}),
                "groq":       dict(method="GET",  url="https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {key}"}),
                "gemini":     dict(method="GET",  url=f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
                    headers={}),
                "together":   dict(method="GET",  url="https://api.together.xyz/v1/models",
                    headers={"Authorization": f"Bearer {key}"}),
                "mistral":    dict(method="GET",  url="https://api.mistral.ai/v1/models",
                    headers={"Authorization": f"Bearer {key}"}),
                "openrouter": dict(method="GET",  url="https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {key}"}),
            }
            cfg = test_configs.get(provider_id)
            if not cfg:
                return {"provider": provider_id, "success": False, "error": "No test available"}
            method = cfg.pop("method")
            if method == "GET":
                r = await c.get(**cfg)
            else:
                r = await c.post(**cfg)
            latency = round((time.time()-start)*1000, 1)
            ok = r.status_code in (200, 400)  # 400 = key valid, bad request params
            err_msg = None
            if not ok:
                try:
                    body = r.json()
                    err_msg = body.get("error", {}).get("message") or body.get("detail") or str(r.status_code)
                except Exception:
                    err_msg = f"HTTP {r.status_code}"
            return {"provider": provider_id, "success": ok, "latency_ms": latency,
                    "error": err_msg, "status_code": r.status_code}
    except Exception as e:
        latency = round((time.time()-start)*1000, 1)
        return {"provider": provider_id, "success": False, "error": str(e), "latency_ms": latency}

@app.get("/settings/ollama")
async def get_ollama_settings():
    primary = _get_primary_host()
    check = await _check_ollama_host(primary["url"], primary.get("api_key",""))
    return {
        "url": primary["url"],
        "has_key": bool(primary.get("api_key","")),
        "available": check["healthy"],
        "latency_ms": check.get("latency_ms", 0),
        "models": check.get("models", []),
        "model_count": check.get("model_count", 0),
        "error": check.get("error"),
        "all_hosts": _load_ollama_hosts(),
    }

@app.post("/settings/ollama")
async def save_ollama_settings(url: str = Query(...), api_key: str = Query("")):
    """Add or update primary Ollama host with optional API key."""
    hosts = _load_ollama_hosts()
    # Update existing or insert at front
    existing = next((h for h in hosts if h["url"]==url), None)
    if existing:
        if api_key: existing["api_key"] = api_key
        existing["enabled"] = True
    else:
        hosts.insert(0, {"url": url, "name": "Ollama", "api_key": api_key, "enabled": True})
    _save_ollama_hosts(hosts)
    check = await _check_ollama_host(url, api_key)
    return {"saved": True, "url": url, "available": check["healthy"],
            "model_count": check.get("model_count",0), "error": check.get("error")}

# ════════════════════════════════════════════════════════════
# OPENCLAW ENDPOINTS — unified provider routing
# ════════════════════════════════════════════════════════════

import sys as _sys
_sys.path.insert(0, str(HERE / "openclaw"))

class OpenClawRunReq(BaseModel):
    task: str
    model: Optional[str] = None
    provider: Optional[str] = None   # groq | openai | anthropic | ollama | together | mistral | gemini | openrouter
    cwd: str = "."
    session_id: Optional[str] = None
    stream: bool = True

# In-memory sessions
_openclaw_sessions: Dict[str, Any] = {}

def _oc_import():
    """Import OpenClaw core (deferred so missing openai SDK doesn't break startup)."""
    from openclaw.core import (
        AgentSession, PROVIDERS, detect_provider, detect_model,
        _get_provider_config, list_provider_models, check_provider,
    )
    return AgentSession, PROVIDERS, detect_provider, detect_model, _get_provider_config, list_provider_models, check_provider


@app.post("/openclaw/run")
async def openclaw_run(req: OpenClawRunReq):
    """Stream an OpenClaw agentic coding task using any configured provider."""
    try:
        AgentSession, PROVIDERS, detect_provider, detect_model, _get_provider_config, _, _ = _oc_import()
    except ImportError as e:
        async def err():
            yield f"data: {json.dumps({'event_type':'error','content':f'OpenClaw import error: {e}. Run: pip install openai'})}\n\n"
        return EventSourceResponse(err())

    sid = req.session_id or str(uuid.uuid4())
    cwd = str(Path(req.cwd).resolve()) if req.cwd not in (".", "") else str(Path.cwd())

    # Resolve provider — UI selection > env auto-detect
    provider = req.provider or detect_provider()
    if provider not in PROVIDERS:
        provider = detect_provider()

    # Resolve model — UI selection > provider default
    keys = _get_key_map()
    model = req.model or detect_model(provider)

    # Get or create session (keyed by sid)
    if sid not in _openclaw_sessions:
        _openclaw_sessions[sid] = AgentSession(model=model, provider=provider, cwd=cwd)
    session = _openclaw_sessions[sid]
    # Allow hot-swapping model/provider mid-session
    session.model = model
    session.provider = provider

    async def stream_run():
        queue: asyncio.Queue = asyncio.Queue()

        def on_token(tok: str):
            queue.put_nowait({"event_type":"stream_token","content":tok,"session_id":sid})

        def on_tool(name: str, args: dict, result):
            if result is None:
                queue.put_nowait({"event_type":"tool_call","content":{"tool":name,"args":args},"session_id":sid})
            else:
                queue.put_nowait({"event_type":"tool_result","content":{"tool":name,"result":str(result)[:800]},"session_id":sid})

        async def runner():
            try:
                result = await session.run(req.task, on_token=on_token, on_tool=on_tool)
                queue.put_nowait({"event_type":"complete","content":result,"session_id":sid})
            except Exception as e:
                queue.put_nowait({"event_type":"error","content":str(e),"session_id":sid})
            queue.put_nowait(None)

        asyncio.create_task(runner())
        while True:
            item = await queue.get()
            if item is None: break
            yield f"data: {json.dumps(item)}\n\n"

    return EventSourceResponse(stream_run())


@app.get("/openclaw/sessions")
async def list_openclaw_sessions():
    return [{"session_id": sid, "provider": s.provider, "model": s.model,
             "cwd": s.cwd, "turns": len([m for m in s.messages if m["role"]=="user"])}
            for sid, s in _openclaw_sessions.items()]


@app.delete("/openclaw/sessions/{session_id}")
async def clear_openclaw_session(session_id: str):
    if session_id in _openclaw_sessions:
        del _openclaw_sessions[session_id]
    return {"cleared": session_id}


@app.get("/openclaw/providers")
async def openclaw_providers():
    """All providers with configured status — used by the OpenClaw page model selector."""
    try:
        _, PROVIDERS, _, _, _get_provider_config, _, _ = _oc_import()
    except ImportError:
        return []
    keys = _get_key_map()
    result = []
    for pid, cfg in PROVIDERS.items():
        key_env = cfg.get("api_key_env","")
        if pid == "ollama":
            host = _get_primary_host()
            configured = True  # always listed; health determined separately
            url = host["url"]
        else:
            configured = bool(keys.get(pid,""))
            url = cfg.get("base_url","")
        result.append({
            "id": pid,
            "name": cfg["name"],
            "icon": cfg["icon"],
            "configured": configured,
            "free": cfg.get("free", False),
            "default_model": cfg.get("default_model",""),
            "url": url,
        })
    return result


@app.get("/openclaw/models")
async def openclaw_models(provider: Optional[str] = Query(None)):
    """Models for a provider — used to populate the model dropdown."""
    try:
        _, PROVIDERS, dp, _, _get_provider_config, list_provider_models, _ = _oc_import()
    except ImportError:
        return []
    p = provider or dp()
    models = await list_provider_models(p)
    # For Ollama, only return installed models
    if p == "ollama":
        installed = {m.get("name","") for m in await ollama_models()}
        return [{"name": m, "installed": True, "provider": p} for m in installed] or                [{"name": cfg, "installed": False, "provider": p} for cfg in [
                   "qwen2.5-coder:7b","llama3.2:latest","deepseek-r1:8b"]]
    return [{"name": m, "installed": True, "provider": p} for m in models]


@app.post("/openclaw/providers/{provider_id}/check")
async def check_openclaw_provider(provider_id: str):
    """Test a provider connection from the OpenClaw page."""
    try:
        _, _, _, _, _, _, check_provider = _oc_import()
        return await check_provider(provider_id)
    except ImportError as e:
        return {"ok": False, "error": str(e), "provider": provider_id}


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
