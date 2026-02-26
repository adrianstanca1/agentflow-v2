# AgentFlow v2

**The best agentic AI platform on the market — local-first, Ollama-powered, zero API keys required.**

```
 █████╗  ██████╗ ███████╗███╗  ██╗████████╗███████╗██╗      ██████╗ ██╗    ██╗
██╔══██╗██╔════╝ ██╔════╝████╗ ██║╚══██╔══╝██╔════╝██║     ██╔═══██╗██║    ██║
███████║██║  ███╗█████╗  ██╔██╗██║   ██║   █████╗  ██║     ██║   ██║██║ █╗ ██║
██╔══██║██║   ██║██╔══╝  ██║╚████║   ██║   ██╔══╝  ██║     ██║   ██║██║███╗██║
██║  ██║╚██████╔╝███████╗██║ ╚███║   ██║   ██║     ███████╗╚██████╔╝╚███╔███╔╝
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚══╝   ╚═╝   ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝  v2
```

## What makes it the best

| Feature | AgentFlow v2 |
|---|---|
| **Local-first** | 100% Ollama — zero API keys needed |
| **Multi-host** | Manage local + remote + cloud Ollama |
| **8 Agents** | Coding · Research · Data · DevOps · Writer · SQL · QA · Assistant |
| **Model Hub** | Browse, pull, delete 20+ curated models with live progress |
| **MCP Servers** | GitHub · Filesystem · Brave Search · Puppeteer · Postgres · Slack |
| **RAG** | Qdrant vector store + Ollama embeddings |
| **Streaming** | Real-time SSE + WebSocket for everything |
| **Hot-swap** | Switch model at runtime without restarting |
| **Cloud fallback** | Auto-detects OpenAI/Anthropic keys, uses cloud when available |
| **Observability** | Langfuse tracing, Temporal workflows |

## Quick Start

```bash
git clone https://github.com/yourorg/agentflow-v2
cd agentflow-v2
chmod +x setup.sh && ./setup.sh
```

Open: **http://localhost:3000**

## Requirements

- Docker + Docker Compose v2
- [Ollama](https://ollama.ai) installed locally (`brew install ollama` / Linux installer)
- 8GB RAM minimum, 16GB+ recommended for larger models

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Dashboard                       │
│  Chat │ Model Hub │ Agents │ MCP │ Knowledge │ Settings  │
└──────────────────────┬──────────────────────────────────┘
                       │ REST + SSE + WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend                         │
│  /agents  /ollama  /mcp  /knowledge  /ws                 │
└──┬────────────┬──────────────┬────────────┬─────────────┘
   │            │              │            │
┌──▼──┐  ┌─────▼─────┐  ┌─────▼──┐  ┌─────▼──────┐
│Ollama│  │  LangGraph │  │  MCP   │  │  Qdrant    │
│Multi-│  │  Agents    │  │ Servers│  │  + Ollama  │
│Host  │  │  8 types   │  │ 10+    │  │  Embeddings│
└──────┘  └────────────┘  └────────┘  └────────────┘
```

## Agents

| Agent | Best Model | Use Case |
|---|---|---|
| **Coding** | `qwen2.5-coder:7b` | Code gen, debugging, testing |
| **Research** | `llama3.1:8b` | Web search, papers, synthesis |
| **Data Analyst** | `llama3.1:8b` | Stats, SQL, visualization |
| **DevOps** | `llama3.2:3b` | Docker, K8s, CI/CD |
| **Writer** | `mistral:7b` | Articles, copy, docs |
| **SQL** | `qwen2.5-coder:7b` | Query optimization, schema design |
| **QA** | `llama3.2:3b` | Test generation, code review |
| **Assistant** | `llama3.2:3b` | General purpose |

## Running on GPU

```bash
# Enable GPU in docker-compose.yml
# Uncomment the `deploy.resources.reservations` block under the ollama service
docker compose --profile gpu up -d
```

## Adding Remote Ollama

1. Open **Model Hub → Hosts tab**
2. Enter the host URL: `http://192.168.1.100:11434`
3. Click **Connect**

Or via env:
```env
OLLAMA_REMOTE_HOSTS=[{"url":"http://192.168.1.100:11434","name":"Server"}]
```

## API

Full REST API at `http://localhost:8000/docs`

Key endpoints:
- `POST /agents/run` — run any agent (streaming SSE)
- `GET /ollama/models` — list installed models
- `POST /ollama/models/pull` — pull model with SSE progress
- `GET /ollama/hosts` — list all Ollama hosts
- `POST /mcp/servers` — add MCP server
- `POST /knowledge/ingest` — add to vector KB
- `WS /ws/{id}` — WebSocket real-time interface

## Stack

**Backend:** FastAPI · LangGraph · LiteLLM · Qdrant · PostgreSQL · Redis · Langfuse · Temporal  
**Frontend:** React 18 · TypeScript · Tailwind · Framer Motion · Vite  
**AI:** Ollama · LangChain · MCP Protocol · FastEmbed  
**Infra:** Docker Compose · Nginx · Open WebUI
