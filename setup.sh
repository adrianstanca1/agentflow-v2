#!/usr/bin/env bash
set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'

echo ""
echo -e "${CYAN}${BOLD}"
echo " █████╗  ██████╗ ███████╗███╗  ██╗████████╗███████╗██╗      ██████╗ ██╗    ██╗"
echo "██╔══██╗██╔════╝ ██╔════╝████╗ ██║╚══██╔══╝██╔════╝██║     ██╔═══██╗██║    ██║"
echo "███████║██║  ███╗█████╗  ██╔██╗██║   ██║   █████╗  ██║     ██║   ██║██║ █╗ ██║"
echo "██╔══██║██║   ██║██╔══╝  ██║╚████║   ██║   ██╔══╝  ██║     ██║   ██║██║███╗██║"
echo "██║  ██║╚██████╔╝███████╗██║ ╚███║   ██║   ██║     ███████╗╚██████╔╝╚███╔███╔╝"
echo "╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚══╝   ╚═╝   ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝"
echo "                                                                          v2.0.0${RESET}"
echo ""

# Detect GPU
HAS_GPU=false
if command -v nvidia-smi &>/dev/null 2>&1; then
  HAS_GPU=true
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
  echo -e "${GREEN}✓ GPU detected: ${GPU_COUNT} NVIDIA GPU(s)${RESET}"
else
  echo -e "${YELLOW}⚠  No NVIDIA GPU — using CPU inference (slower)${RESET}"
fi

# Check deps
for cmd in docker ollama; do
  if ! command -v $cmd &>/dev/null; then
    echo "✗ $cmd not found. Please install it first."
    [ "$cmd" = "ollama" ] && echo "  → curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
  fi
done
echo -e "${GREEN}✓ Docker and Ollama found${RESET}"

# Create .env if missing
if [ ! -f .env ]; then
  cp .env.example .env 2>/dev/null || cat > .env << 'ENV'
POSTGRES_USER=agentflow
POSTGRES_PASSWORD=agentflow_secret_change_me
REDIS_PASSWORD=redis_secret_change_me
LANGFUSE_SECRET_KEY=lf-secret-change-me
LANGFUSE_PUBLIC_KEY=lf-public-change-me
OLLAMA_URL=http://host.docker.internal:11434
# Optional cloud API keys (leave empty for fully local mode)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
TAVILY_API_KEY=
ENV
  echo -e "${GREEN}✓ Created .env (edit to add API keys)${RESET}"
fi

# Start Ollama
echo ""
echo -e "${CYAN}Starting Ollama...${RESET}"
if [[ "$OSTYPE" == "darwin"* ]]; then
  pgrep -x ollama > /dev/null || (open -a Ollama 2>/dev/null || ollama serve &>/dev/null &)
else
  pgrep -x ollama > /dev/null || (ollama serve &>/dev/null & sleep 2)
fi
sleep 2

# Pull default models
echo -e "${CYAN}Pulling default models (this may take a while)...${RESET}"
MODELS=("llama3.2:latest" "nomic-embed-text")
if $HAS_GPU; then
  MODELS+=("qwen2.5-coder:7b" "llama3.1:8b")
fi
for model in "${MODELS[@]}"; do
  echo -n "  Pulling $model... "
  ollama pull "$model" &>/dev/null && echo -e "${GREEN}✓${RESET}" || echo -e "${YELLOW}skipped${RESET}"
done

# Launch services
echo ""
echo -e "${CYAN}Starting platform services...${RESET}"
COMPOSE_ARGS=""
$HAS_GPU && COMPOSE_ARGS="--profile gpu"
docker compose $COMPOSE_ARGS up -d --build 2>&1 | grep -E "Started|Created|Warning|Error" || true

# Wait for API
echo -n "Waiting for API to be ready"
for i in $(seq 1 30); do
  curl -sf http://localhost:8000/health > /dev/null 2>&1 && break
  echo -n "."; sleep 2
done
echo ""

# Check status
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
  echo ""
  echo -e "${GREEN}${BOLD}✅ AgentFlow v2 is running!${RESET}"
  echo ""
  echo -e "  ${CYAN}Dashboard:${RESET}   http://localhost:3000"
  echo -e "  ${CYAN}API Docs:${RESET}    http://localhost:8000/docs"
  echo -e "  ${CYAN}Langfuse:${RESET}    http://localhost:3001"
  echo -e "  ${CYAN}Open WebUI:${RESET}  http://localhost:8888"
  echo -e "  ${CYAN}LiteLLM:${RESET}     http://localhost:4000"
  echo ""
  echo -e "  Mode: $($HAS_GPU && echo 'GPU' || echo 'CPU') | Ollama: $(ollama list 2>/dev/null | tail -n+2 | wc -l | tr -d ' ') models"
  echo ""
else
  echo -e "${YELLOW}⚠  API not responding. Check logs: docker compose logs api${RESET}"
fi
