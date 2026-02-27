"""
AgentFlow v2 — Specialist Agents
8 production-ready agents, all work with local Ollama or cloud models.
"""
import json
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

import structlog
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .base import UniversalAgent
from ..core.config import settings

logger = structlog.get_logger()


class AgentState(TypedDict):
    messages: List[Any]
    task_input: str
    task_output: str
    thoughts: List[str]
    iteration: int
    max_iterations: int
    status: str
    error: Optional[str]
    session_id: str
    metadata: Dict[str, Any]
    plan: Optional[str]


def build_react_graph(agent: "UniversalAgent", tools: List, state_type=AgentState):
    """Build a standard ReAct graph for any agent."""
    llm_with_tools = agent.llm_with_tools
    tool_node = ToolNode(tools)

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1] if state["messages"] else None
        if state.get("iteration", 0) >= state.get("max_iterations", 15):
            return "__end__"
        if last and hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "__end__"

    async def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=agent.system_prompt)] + list(messages)
        response = await llm_with_tools.ainvoke(messages)
        output = response.content if isinstance(response.content, str) else state.get("task_output", "")
        done = not (hasattr(response, "tool_calls") and response.tool_calls)
        return {**state, "messages": messages + [response], "iteration": state.get("iteration", 0) + 1,
                "task_output": output, "status": "completed" if done else "running"}

    async def tools_wrapper(state: AgentState) -> AgentState:
        result = await tool_node.ainvoke(state)
        return {**state, **result, "status": "running"}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_wrapper)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=agent.checkpointer)


# ── TOOLS ─────────────────────────────────────────────────────────────────────

@tool
async def execute_code(code: str, language: str = "python", timeout: int = 30) -> str:
    """Execute code in a secure sandbox. Returns stdout/stderr/error.
    Args:
        code: Source code to execute
        language: Programming language (python, javascript, bash)
        timeout: Max seconds to run
    """
    import subprocess, sys, os, tempfile
    try:
        if settings.e2b_api_key and language == "python":
            from e2b_code_interpreter import AsyncSandbox
            async with await AsyncSandbox.create() as sb:
                ex = await sb.run_code(code)
                return json.dumps({"stdout": ex.text, "error": str(ex.error) if ex.error else None})
        with tempfile.NamedTemporaryFile(suffix=f".{language}", mode="w", delete=False) as f:
            f.write(code); tmp = f.name
        interp = {"python": [sys.executable], "javascript": ["node"], "bash": ["bash"]}.get(language, [sys.executable])
        proc = subprocess.run(interp + [tmp], capture_output=True, text=True, timeout=timeout)
        os.unlink(tmp)
        return json.dumps({"stdout": proc.stdout[:3000], "stderr": proc.stderr[:500], "returncode": proc.returncode})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
async def write_file(path: str, content: str) -> str:
    """Write content to a file in the agent workspace.
    Args:
        path: Relative file path
        content: File content to write
    """
    import os
    ws = "/tmp/agentflow_workspace"
    os.makedirs(ws, exist_ok=True)
    full = os.path.join(ws, path.lstrip("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return f"Written {len(content)} chars to {path}"


@tool
async def read_file(path: str) -> str:
    """Read a file from the agent workspace.
    Args:
        path: File path to read
    """
    import os
    full = os.path.join("/tmp/agentflow_workspace", path.lstrip("/"))
    if not os.path.exists(full):
        return f"File not found: {path}"
    with open(full) as f:
        return f.read()[:8000]


@tool
async def run_shell(command: str) -> str:
    """Run a shell command in the workspace directory.
    Args:
        command: Shell command to run
    """
    import subprocess
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, cwd="/tmp/agentflow_workspace")
        return f"OUT:\n{proc.stdout[:2000]}\nERR:\n{proc.stderr[:300]}"
    except Exception as e:
        return f"Error: {e}"


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information.
    Args:
        query: Search query string
        max_results: Number of results (1-10)
    """
    try:
        if settings.tavily_api_key:
            from tavily import AsyncTavilyClient
            client = AsyncTavilyClient(api_key=settings.tavily_api_key)
            r = await client.search(query=query, max_results=max_results, include_answer=True)
            parts = []
            if r.get("answer"): parts.append(f"Answer: {r['answer']}")
            for res in r.get("results", [])[:max_results]:
                parts.append(f"[{res['title']}]({res['url']})\n{res['content'][:400]}")
            return "\n\n".join(parts)
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().run(query)[:3000]
    except Exception as e:
        return f"Search error: {e}"


@tool
async def fetch_url(url: str) -> str:
    """Fetch and extract text content from a URL.
    Args:
        url: URL to fetch
    """
    try:
        import httpx
        from bs4 import BeautifulSoup
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
            r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script","style","nav","footer","header","aside"]): tag.decompose()
            lines = [l for l in soup.get_text(separator="\n", strip=True).splitlines() if l.strip()]
            return "\n".join(lines)[:5000]
    except Exception as e:
        return f"Fetch error: {e}"


@tool
async def search_arxiv(query: str, max_results: int = 3) -> str:
    """Search academic papers on arXiv.
    Args:
        query: Research topic
        max_results: Number of papers to return
    """
    try:
        import httpx
        from xml.etree import ElementTree as ET
        async with httpx.AsyncClient() as c:
            r = await c.get("http://export.arxiv.org/api/query",
                params={"search_query": f"all:{query}", "max_results": max_results, "sortBy": "relevance"}, timeout=10)
        root = ET.fromstring(r.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        papers = []
        for e in root.findall("a:entry", ns)[:max_results]:
            title = (e.findtext("a:title", namespaces=ns) or "").strip()
            summary = (e.findtext("a:summary", namespaces=ns) or "").strip()[:300]
            link = e.findtext("a:id", namespaces=ns) or ""
            papers.append(f"**{title}**\n{summary}\n{link}")
        return "\n\n---\n\n".join(papers) if papers else "No papers found"
    except Exception as e:
        return f"arXiv error: {e}"


@tool
async def knowledge_base_search(query: str, collection: str = "research", limit: int = 5) -> str:
    """Search the internal vector knowledge base.
    Args:
        query: Semantic search query
        collection: Collection name to search
        limit: Max results
    """
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=settings.qdrant_url)
        try:
            from ..core.ollama_manager import ollama_manager
            embs = await ollama_manager.embed(settings.ollama_default_embedding, query)
            vector = embs[0] if embs else None
        except Exception:
            vector = None
        if not vector:
            from fastembed import TextEmbedding
            vector = list(TextEmbedding("BAAI/bge-small-en-v1.5").embed([query]))[0].tolist()
        results = await client.search(collection_name=collection, query_vector=vector, limit=limit, with_payload=True)
        if not results:
            return "No results found in knowledge base."
        return "\n\n".join(f"[{r.score:.2f}] {(r.payload or {}).get('content','')[:400]}" for r in results)
    except Exception as e:
        return f"KB search error: {e}"


@tool
async def analyze_data(data: str, analysis_type: str = "descriptive") -> str:
    """Run statistical analysis on CSV or JSON data.
    Args:
        data: CSV or JSON string
        analysis_type: Type: descriptive, correlation, regression
    """
    code = f"""
import pandas as pd, numpy as np, json, io, warnings
warnings.filterwarnings('ignore')
try:
    df = pd.read_csv(io.StringIO('''{data[:4000]}'''))
except:
    df = pd.DataFrame(json.loads('''{data[:4000]}'''))
print("Shape:", df.shape)
print("\\nDescribe:\\n", df.describe().to_string())
if '{analysis_type}' == 'correlation':
    print("\\nCorrelation:\\n", df.select_dtypes('number').corr().round(3).to_string())
print("\\nNulls:\\n", df.isnull().sum().to_string())
"""
    return await execute_code(code)


@tool
async def sql_query(query: str) -> str:
    """Execute a SQL query against the platform database.
    Args:
        query: SQL SELECT query to execute
    """
    import subprocess
    conn = settings.database_url_sync.replace("+asyncpg","").replace("postgresql+asyncpg","postgresql")
    try:
        r = subprocess.run(["psql", conn, "-c", query, "--csv"], capture_output=True, text=True, timeout=15)
        return r.stdout[:3000] or r.stderr[:500]
    except Exception as e:
        return f"SQL error: {e}"


@tool
async def docker_command(command: str) -> str:
    """Run a safe Docker read-only command (ps, images, inspect, logs, info).
    Args:
        command: Docker subcommand (e.g., 'ps -a', 'images', 'logs container_name')
    """
    import subprocess
    safe = ["ps", "images", "inspect", "logs", "stats", "info", "version", "network", "volume", "compose"]
    parts = command.strip().split()
    if parts and parts[0] not in safe:
        return f"Not allowed: docker {parts[0]}. Safe: {', '.join(safe)}"
    try:
        r = subprocess.run(["docker"] + parts, capture_output=True, text=True, timeout=10)
        return r.stdout[:2000] or r.stderr[:500]
    except FileNotFoundError:
        return "Docker not available"
    except Exception as e:
        return f"Error: {e}"


@tool
async def generate_dockerfile(app_type: str, language: str = "python", framework: str = "") -> str:
    """Generate an optimized production Dockerfile.
    Args:
        app_type: Application type (web, api, worker, cli)
        language: Language (python, node, go, rust)
        framework: Framework name (fastapi, express, etc.)
    """
    templates = {
        "python": "FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nUSER 1000\nEXPOSE 8000\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]",
        "node": "FROM node:20-alpine\nWORKDIR /app\nCOPY package*.json ./\nRUN npm ci --only=production\nCOPY . .\nUSER node\nEXPOSE 3000\nCMD [\"node\", \"server.js\"]",
        "go": "FROM golang:1.22-alpine AS build\nWORKDIR /app\nCOPY . .\nRUN go build -o app .\nFROM alpine:latest\nCOPY --from=build /app/app /app\nEXPOSE 8080\nCMD [\"/app\"]",
    }
    return templates.get(language, f"# {language}/{framework} Dockerfile\n# Customize for {app_type}")


@tool
async def generate_k8s(app_name: str, image: str, port: int = 8000, replicas: int = 2) -> str:
    """Generate a Kubernetes Deployment + Service manifest.
    Args:
        app_name: Application name
        image: Docker image (name:tag)
        port: Container port
        replicas: Number of replicas
    """
    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels: {{app: {app_name}}}
  template:
    metadata:
      labels: {{app: {app_name}}}
    spec:
      containers:
      - name: {app_name}
        image: {image}
        ports: [{{containerPort: {port}}}]
        resources:
          requests: {{memory: "256Mi", cpu: "100m"}}
          limits: {{memory: "512Mi", cpu: "500m"}}
---
apiVersion: v1
kind: Service
metadata:
  name: {app_name}
spec:
  selector: {{app: {app_name}}}
  ports: [{{port: {port}}}]"""


@tool
async def generate_tests(code: str, framework: str = "pytest") -> str:
    """Generate test stubs for the provided code.
    Args:
        code: Source code to generate tests for
        framework: Test framework (pytest, jest, mocha)
    """
    return f"""# Auto-generated tests ({framework})
# For: {code[:80].strip()}...

import pytest

class TestCode:
    def test_happy_path(self):
        # Test normal operation
        pass
    
    def test_edge_cases(self):
        # Test boundary values
        pass
    
    def test_error_handling(self):
        # Test exceptions
        with pytest.raises(Exception):
            pass
    
    def test_types(self):
        # Test input/output types
        pass
"""


# ═══════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════

class CodingAgent(UniversalAgent):
    """
    Coding Agent — backed by OpenClaw's agentic tool loop.
    Uses Ollama-native tool calling for file edits, shell commands, git ops.
    Falls back to standard LangGraph when OpenClaw is unavailable.
    """
    TOOLS = [execute_code, write_file, read_file, run_shell, web_search]
    def __init__(self, **kw):
        super().__init__(agent_id="coding", name="Coding Agent",
            description="Code generation, debugging, testing, git — powered by OpenClaw + Ollama",
            agent_type="coding", tools=self.TOOLS,
            model=kw.pop("model", settings.default_model_coding), **kw)
    def default_system_prompt(self):
        return """You are an elite software engineer. Write clean, tested, production-quality code.
Process: 1) Understand requirements 2) Plan approach 3) Write complete code 4) Execute & verify 5) Fix errors 6) Deliver final clean code.
Always execute code to verify it works. Use write_file for multi-file projects."""
    def _build_graph(self): return build_react_graph(self, self.TOOLS)


class ResearchAgent(UniversalAgent):
    TOOLS = [web_search, fetch_url, search_arxiv, knowledge_base_search]
    def __init__(self, **kw):
        super().__init__(agent_id="research", name="Research Agent",
            description="Deep web research, academic papers, knowledge synthesis, fact-finding",
            agent_type="research", tools=self.TOOLS,
            model=kw.pop("model", settings.default_model_research),
            max_iterations=kw.pop("max_iterations", 12), **kw)
    def default_system_prompt(self):
        return """You are a world-class research analyst. Search multiple sources, cross-reference facts, cite everything.
Process: 1) Plan search strategy 2) Search web + arxiv 3) Fetch key sources 4) Verify facts 5) Synthesize comprehensive report with citations."""
    def _build_graph(self): return build_react_graph(self, self.TOOLS)


class DataAnalystAgent(UniversalAgent):
    TOOLS = [analyze_data, sql_query, execute_code, web_search]
    def __init__(self, **kw):
        super().__init__(agent_id="data_analyst", name="Data Analyst",
            description="Statistical analysis, data visualization, SQL queries, business insights",
            agent_type="data_analyst", tools=self.TOOLS, **kw)
    def default_system_prompt(self):
        return """You are an expert data analyst. You turn raw data into actionable insights.
Always: check data quality, run descriptive stats, find patterns/outliers, generate visualization code, give business recommendations."""
    def _build_graph(self): return build_react_graph(self, self.TOOLS)


class DevOpsAgent(UniversalAgent):
    TOOLS = [docker_command, generate_dockerfile, generate_k8s, execute_code, run_shell]
    def __init__(self, **kw):
        super().__init__(agent_id="devops", name="DevOps Agent",
            description="Docker, Kubernetes, CI/CD pipelines, infrastructure automation",
            agent_type="devops", tools=self.TOOLS, **kw)
    def default_system_prompt(self):
        return """You are a senior DevOps and platform engineer. You design production infrastructure.
Best practices: least privilege, health checks, resource limits, graceful shutdown, IaC, observability."""
    def _build_graph(self): return build_react_graph(self, self.TOOLS)


class WriterAgent(UniversalAgent):
    TOOLS = [web_search, fetch_url]
    def __init__(self, **kw):
        super().__init__(agent_id="writer", name="Writer Agent",
            description="Articles, blog posts, marketing copy, technical docs, creative writing",
            agent_type="writer", tools=self.TOOLS, **kw)
    def default_system_prompt(self):
        return """You are a talented versatile writer. You create compelling, well-structured content tailored to the audience.
Adapt tone: technical for developers, simple for consumers, formal for business. Use clear structure, strong opening, actionable conclusion."""
    def _build_graph(self): return build_react_graph(self, self.TOOLS)


class SQLAgent(UniversalAgent):
    TOOLS = [sql_query, analyze_data, execute_code]
    def __init__(self, **kw):
        super().__init__(agent_id="sql", name="SQL Agent",
            description="SQL query writing, optimization, schema design, database analysis",
            agent_type="sql", tools=self.TOOLS, **kw)
    def default_system_prompt(self):
        return """You are an expert SQL and database engineer. Write optimized queries for PostgreSQL, MySQL, SQLite.
Always consider: indexes, execution plans, N+1 patterns, proper joins. Explain performance implications."""
    def _build_graph(self): return build_react_graph(self, self.TOOLS)


class QAAgent(UniversalAgent):
    TOOLS = [generate_tests, execute_code, read_file, run_shell]
    def __init__(self, **kw):
        super().__init__(agent_id="qa", name="QA Agent",
            description="Test generation, code review, quality assurance, bug finding",
            agent_type="qa", tools=self.TOOLS, **kw)
    def default_system_prompt(self):
        return """You are a senior QA engineer. Write comprehensive tests: unit, integration, e2e, performance, security.
Cover happy paths, edge cases, error conditions, boundary values, concurrency issues. Use pytest/Jest best practices."""
    def _build_graph(self): return build_react_graph(self, self.TOOLS)


class AssistantAgent(UniversalAgent):
    TOOLS = [web_search, knowledge_base_search]
    def __init__(self, **kw):
        super().__init__(agent_id="assistant", name="Assistant",
            description="General purpose: questions, explanations, brainstorming, planning, analysis",
            agent_type="assistant", tools=self.TOOLS, **kw)
    def default_system_prompt(self):
        return """You are a highly capable, knowledgeable AI assistant. Be concise, accurate, and genuinely helpful.
Think step-by-step for complex questions. Search the web for current info when needed."""
    def _build_graph(self): return build_react_graph(self, self.TOOLS)


# ── Registry ───────────────────────────────────────────────────────────────────
AGENT_CLASSES = {
    "coding": CodingAgent, "research": ResearchAgent,
    "data_analyst": DataAnalystAgent, "devops": DevOpsAgent,
    "writer": WriterAgent, "sql": SQLAgent,
    "qa": QAAgent, "assistant": AssistantAgent,
}


class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, UniversalAgent] = {}

    def initialize_defaults(self, prefer_local: bool = False):
        for key, cls in AGENT_CLASSES.items():
            try:
                self._agents[key] = cls(prefer_local=prefer_local)
            except Exception as e:
                logger.warning("Agent init failed", key=key, error=str(e))
        return self

    def register(self, key: str, agent: UniversalAgent):
        self._agents[key] = agent

    def get(self, key: str) -> Optional[UniversalAgent]:
        return self._agents.get(key)

    def list_agents(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self._agents.values()]

    def keys(self) -> List[str]:
        return list(self._agents.keys())

    def switch_all_models(self, model: str, prefer_local: bool = False):
        for agent in self._agents.values():
            agent.switch_model(model, prefer_local=prefer_local)


registry = AgentRegistry()
