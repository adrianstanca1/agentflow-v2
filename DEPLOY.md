# AgentFlow v2 — Deployment Guide

## Quick Start (Development)

```bash
git clone https://github.com/adrianstanca1/agentflow-v2
cd agentflow-v2

# Install Python deps
pip install -r requirements.txt

# Build frontend
cd frontend && npm install && npm run build && cd ..

# Configure
cp .env.example .env
# Edit .env — add any API keys you want

# Run
python3 server.py
# → http://localhost:8000
```

---

## Docker Compose (Recommended for Production)

### Full stack (Ollama + Qdrant + LiteLLM + AgentFlow)

```bash
cp .env.example .env
# Edit .env — at minimum set SECRET_KEY

# Build and start
docker compose --profile agentflow up -d

# Or start just the infrastructure and run AgentFlow locally:
docker compose up -d ollama qdrant postgres redis
python3 server.py
```

### Services started:
| Service | Port | Description |
|---------|------|-------------|
| AgentFlow | 8000 | Main API + Frontend |
| Ollama | 11434 | Local LLM engine |
| Qdrant | 6333 | Vector database |
| PostgreSQL | 5432 | Relational DB |
| Redis | 6379 | Cache + queues |
| LiteLLM | 4000 | Cloud LLM gateway |
| Langfuse | 3001 | Observability |

---

## GitHub Integration

1. Go to **Settings → GitHub** in the UI  
   *or* add to `.env`:
   ```
   GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
   ```

2. Create a token at [github.com/settings/tokens/new](https://github.com/settings/tokens/new?scopes=repo,read:user)  
   Required scopes: `repo`, `read:user`

3. Features:
   - **Repo browser** — file tree, syntax-highlighted file viewer, branch switching
   - **Commits** — full commit history with author avatars
   - **Pull Requests** — open/closed PRs, diff viewer, review status
   - **Issues** — browsing with label colors
   - **AI Code Review** — click "AI Review" on any PR for instant analysis
   - **AI Draft PR** — generate PR title + body from branch commits
   - **Create PR** — from AI draft or manually

---

## Docker Integration

The backend needs access to the Docker socket:

### When running locally:
Docker CLI must be installed on the host. The integration uses `docker` CLI commands directly.

### When running in Docker:
The Docker socket must be mounted (already done in `docker-compose.yml`):
```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

### Features:
- **Containers** — list all containers, start/stop/restart/remove, view logs
- **Container detail** — ports, mounts, env, restart policy, live stats
- **Images** — browse local images, search Docker Hub, pull with live progress
- **Compose** — manage the AgentFlow stack: up/down/status from the UI
- **System** — volumes, networks, disk usage, prune unused resources

---

## Environment Variables

```bash
# ── Server ─────────────────────────────────────────────
SECRET_KEY=change_me_in_production
PORT=8000

# ── Ollama (local LLMs) ────────────────────────────────
OLLAMA_URL=http://localhost:11434

# ── Cloud providers (all optional) ────────────────────
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
TOGETHER_API_KEY=...
MISTRAL_API_KEY=...
OPENROUTER_API_KEY=sk-or-...

# ── GitHub integration ─────────────────────────────────
GITHUB_TOKEN=ghp_...

# ── Vector store (optional, falls back to in-memory) ──
QDRANT_URL=http://localhost:6333

# ── Observability (optional) ───────────────────────────
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=lf-pk-...
LANGFUSE_SECRET_KEY=lf-sk-...

# ── Google OAuth (for Gemini) ──────────────────────────
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
```

---

## Updating

```bash
git pull origin main
cd frontend && npm run build && cd ..
# Restart server
```

With Docker:
```bash
git pull origin main
cd frontend && npm run build && cd ..
docker compose --profile agentflow up -d --build agentflow
```

---

## Architecture

```
Browser
  └─ React SPA (frontend/dist)
       └─ FastAPI (server.py :8000)
            ├─ /agents/*      → LangGraph agents (8 types)
            ├─ /github/*      → GitHub API proxy (github_integration.py)
            ├─ /docker/*      → Docker CLI wrapper (docker_integration.py)
            ├─ /openclaw/*    → Universal LLM engine
            ├─ /knowledge/*   → Vector search (Qdrant or in-memory)
            ├─ /mcp/*         → Model Context Protocol servers
            ├─ /ollama/*      → Ollama model management
            ├─ /auth/*        → OAuth2 flows (Google, OpenAI)
            └─ /settings/*    → API key management
```
