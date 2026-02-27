#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#   AgentFlow v2 + OpenClaw — One-Command Server Setup
#   Usage: bash setup.sh [--model qwen2.5-coder:7b] [--port 8000] [--no-ollama]
# ════════════════════════════════════════════════════════════════════════
set -e

R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m' C='\033[0;36m' B='\033[1m' N='\033[0m'
ok()   { echo -e "${G}✓${N} $*"; }
info() { echo -e "${C}▸${N} $*"; }
warn() { echo -e "${Y}⚠${N}  $*"; }
err()  { echo -e "${R}✗${N} $*"; exit 1; }
step() { echo -e "\n${B}${C}══ $* ══${N}"; }

INSTALL_DIR="/opt/agentflow"
PORT=8000
MODEL="qwen2.5-coder:7b"
INSTALL_OLLAMA=true
PULL_MODEL=true
CREATE_SERVICE=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)      MODEL="$2";           shift 2 ;;
    --port)       PORT="$2";            shift 2 ;;
    --dir)        INSTALL_DIR="$2";     shift 2 ;;
    --no-ollama)  INSTALL_OLLAMA=false; shift ;;
    --no-pull)    PULL_MODEL=false;     shift ;;
    --no-service) CREATE_SERVICE=false; shift ;;
    *) warn "Unknown arg: $1"; shift ;;
  esac
done

echo -e "${C}${B}"
echo "   AgentFlow v2 + OpenClaw — Local AI Platform"
echo -e "${N}"
info "Dir: $INSTALL_DIR | Port: $PORT | Model: $MODEL"

[[ $EUID -ne 0 ]] && err "Run as root: sudo bash setup.sh"

step "System Dependencies"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv nodejs npm curl git build-essential 2>/dev/null
ok "System packages ready"

step "AgentFlow Source"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  cd "$INSTALL_DIR" && git pull --quiet origin main && ok "Updated"
elif [[ -d "$INSTALL_DIR" ]]; then
  cd "$INSTALL_DIR" && ok "Using existing files"
else
  git clone --quiet https://github.com/adrianstanca1/agentflow-v2 "$INSTALL_DIR"
  cd "$INSTALL_DIR" && ok "Cloned"
fi

step "Python Environment"
cd "$INSTALL_DIR"
python3 -m venv .venv 2>/dev/null || true
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet \
  fastapi "uvicorn[standard]" sse-starlette pydantic "pydantic-settings" \
  httpx structlog python-dotenv rich aiofiles \
  langchain-core langchain-openai langchain-anthropic \
  langchain langchain-community langgraph 2>/dev/null
ok "Python packages installed"

step "Frontend Build"
if command -v node &>/dev/null; then
  cd "$INSTALL_DIR/frontend"
  npm install --silent --legacy-peer-deps 2>/dev/null
  npm run build --silent 2>/dev/null
  cd "$INSTALL_DIR"
  ok "React frontend built"
else
  warn "Node.js not found — API-only mode"
fi

if [[ "$INSTALL_OLLAMA" == "true" ]]; then
  step "Ollama"
  command -v ollama &>/dev/null || curl -fsSL https://ollama.ai/install.sh | sh 2>/dev/null
  pgrep -x ollama &>/dev/null || (systemctl start ollama 2>/dev/null || (ollama serve &>/tmp/ollama.log & sleep 3))
  ok "Ollama running"
  if [[ "$PULL_MODEL" == "true" ]]; then
    info "Pulling $MODEL..."
    timeout 600 ollama pull "$MODEL" 2>/dev/null && ok "Model ready" || warn "Pull timed out — run: ollama pull $MODEL"
  fi
fi

step "Configuration"
[[ -f .env ]] || cat > .env << ENV
OLLAMA_URL=http://localhost:11434
OPENCLAW_MODEL=$MODEL
PORT=$PORT
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GROQ_API_KEY=
GEMINI_API_KEY=
TOGETHER_API_KEY=
MISTRAL_API_KEY=
ENV
ok ".env ready"

if [[ "$CREATE_SERVICE" == "true" ]] && command -v systemctl &>/dev/null; then
  step "Systemd Service"
  cat > /etc/systemd/system/agentflow.service << SERVICE
[Unit]
Description=AgentFlow v2 + OpenClaw
After=network.target ollama.service
[Service]
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/.venv/bin/python3 $INSTALL_DIR/server.py
Restart=always
RestartSec=5
EnvironmentFile=$INSTALL_DIR/.env
Environment="PYTHONPATH=$INSTALL_DIR/openclaw"
[Install]
WantedBy=multi-user.target
SERVICE
  systemctl daemon-reload
  systemctl enable agentflow
  systemctl restart agentflow
  sleep 3
  systemctl is-active --quiet agentflow && ok "Service running" || warn "Check: journalctl -u agentflow -n 30"
fi

command -v ufw &>/dev/null && ufw allow "$PORT/tcp" &>/dev/null || true

PUBLIC_IP=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
echo ""
echo -e "${G}${B}✅ AgentFlow v2 + OpenClaw deployed!${N}"
echo ""
echo -e "  ${C}Dashboard:${N}  http://${PUBLIC_IP}:${PORT}"
echo -e "  ${C}API Docs:${N}   http://${PUBLIC_IP}:${PORT}/docs"
echo -e "  ${C}Manage:${N}     systemctl status agentflow"
echo -e "  ${C}Upgrade:${N}    cd $INSTALL_DIR && git pull && bash setup.sh"
echo ""
