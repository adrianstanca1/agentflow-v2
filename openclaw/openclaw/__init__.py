"""OpenClaw — Agentic coding agent for any LLM provider."""
from .core import (
    AgentSession, ToolExecutor,
    PROVIDERS, detect_provider, detect_model,
    provider_chat, check_provider, list_provider_models,
    _get_provider_config,
)
from .config import OpenClawConfig, detect_best_model
from .cli import run as main

__version__ = "2.0.0"
__all__ = [
    "AgentSession", "ToolExecutor", "PROVIDERS",
    "detect_provider", "detect_model", "OpenClawConfig", "main",
]
