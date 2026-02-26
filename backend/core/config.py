"""
AgentFlow v2 — Core Configuration
Supports local + remote Ollama, multi-provider LLMs, and all platform services.
"""
import json
from functools import lru_cache
from typing import Any, Dict, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────
    app_name: str = "AgentFlow"
    app_version: str = "2.0.0"
    environment: str = "production"
    secret_key: str = "change_me_32_chars_minimum_please"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # ── Database ───────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://agentflow:agentflow_secret@postgres:5432/agentflow"
    database_url_sync: str = "postgresql://agentflow:agentflow_secret@postgres:5432/agentflow"

    # ── Redis ──────────────────────────────────────────────
    redis_url: str = "redis://:redis_secret@redis:6379"

    # ── Ollama ─────────────────────────────────────────────
    ollama_url: str = "http://ollama:11434"
    ollama_default_model: str = "llama3.2:latest"
    ollama_default_embedding: str = "nomic-embed-text"
    # JSON list of remote Ollama hosts: [{"url": "...", "name": "...", "api_key": "..."}]
    ollama_remote_hosts: str = "[]"

    @property
    def ollama_remote_hosts_list(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.ollama_remote_hosts)
        except Exception:
            return []

    # ── Auto-pull models on startup ─────────────────────────
    ollama_auto_pull: str = "llama3.2:latest,nomic-embed-text"

    @property
    def ollama_auto_pull_list(self) -> List[str]:
        return [m.strip() for m in self.ollama_auto_pull.split(",") if m.strip()]

    # ── Vector DB ──────────────────────────────────────────
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: Optional[str] = None

    # ── LLM Gateway ────────────────────────────────────────
    litellm_url: str = "http://litellm:4000"
    litellm_api_key: str = "sk-agentflow-master"

    # ── Direct LLM Keys ────────────────────────────────────
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    together_api_key: Optional[str] = None
    fireworks_api_key: Optional[str] = None

    # ── Observability ──────────────────────────────────────
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = "lf-pk-agentflow"
    langfuse_secret_key: str = "lf-sk-agentflow"

    # ── Temporal ───────────────────────────────────────────
    temporal_url: str = "temporal:7233"
    temporal_namespace: str = "default"

    # ── Kafka ──────────────────────────────────────────────
    kafka_bootstrap_servers: str = "kafka:9092"

    # ── Search ─────────────────────────────────────────────
    tavily_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None

    # ── Code Execution ─────────────────────────────────────
    e2b_api_key: Optional[str] = None
    daytona_api_key: Optional[str] = None

    # ── Browser Automation ─────────────────────────────────
    playwright_headless: bool = True

    # ── MCP ────────────────────────────────────────────────
    mcp_servers: str = "[]"  # JSON array of MCP server configs

    @property
    def mcp_servers_list(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.mcp_servers)
        except Exception:
            return []

    # ── Default Models ──────────────────────────────────────
    default_model_local: str = "llama3.2:latest"
    default_model_cloud: str = "claude-sonnet-4-6"
    default_model_coding: str = "qwen2.5-coder:7b"
    default_model_research: str = "llama3.1:8b"
    default_model_fast: str = "llama3.2:3b"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def is_development(self) -> bool:
        return self.environment in ("development", "dev")

    def get_llm_base_url(self, prefer_local: bool = False) -> str:
        """Get the best LLM base URL."""
        if prefer_local:
            return f"{self.ollama_url}/v1"
        return f"{self.litellm_url}/v1"

    def get_llm_api_key(self, prefer_local: bool = False) -> str:
        if prefer_local:
            return "ollama"
        return self.litellm_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

    # ── Additional cloud provider keys ─────────────────────
    mistral_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
