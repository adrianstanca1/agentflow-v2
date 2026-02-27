"""OpenClaw — Local-first agentic coding CLI powered by Ollama."""
from .core import AgentSession, ToolExecutor, ollama_available, list_models
from .config import OpenClawConfig, detect_best_model
from .cli import main

__version__ = "1.0.0"
__all__ = ["AgentSession", "ToolExecutor", "OpenClawConfig", "main"]
