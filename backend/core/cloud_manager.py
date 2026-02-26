"""
AgentFlow v2 — Cloud Provider Manager
Treats cloud LLM providers (Anthropic, OpenAI, Groq, Gemini, Together, Mistral)
as first-class "hosts" alongside Ollama — unified interface for all model routing.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import structlog

from .config import settings

logger = structlog.get_logger()


# ── Enums ─────────────────────────────────────────────────────────────────────
class ProviderType(str, Enum):
    ANTHROPIC  = "anthropic"
    OPENAI     = "openai"
    GROQ       = "groq"
    GEMINI     = "gemini"
    TOGETHER   = "together"
    FIREWORKS  = "fireworks"
    MISTRAL    = "mistral"
    OPENROUTER = "openrouter"
    OLLAMA     = "ollama"       # cloud-hosted Ollama endpoint (e.g. Modal, Replicate)


# ── Model catalog ─────────────────────────────────────────────────────────────
@dataclass
class CloudModel:
    id: str                          # canonical ID sent to API
    name: str                        # display name
    provider: ProviderType
    context_length: int
    input_cost_per_1m: float         # USD per 1M tokens
    output_cost_per_1m: float
    category: str                    # chat | code | vision | embedding | reasoning
    tags: List[str] = field(default_factory=list)
    description: str = ""
    is_latest: bool = False          # highlight in hub

    @property
    def cost_label(self) -> str:
        if self.input_cost_per_1m == 0:
            return "Free"
        return f"${self.input_cost_per_1m:.2f}/${ self.output_cost_per_1m:.2f} / 1M"


CLOUD_CATALOG: List[CloudModel] = [
    # ── Anthropic ────────────────────────────────────────────────────────────
    CloudModel("claude-opus-4-5",            "Claude Opus 4.5",           ProviderType.ANTHROPIC, 200_000, 15.0, 75.0,  "reasoning", ["powerful","latest"], "Most capable Claude model", is_latest=True),
    CloudModel("claude-sonnet-4-5",          "Claude Sonnet 4.5",         ProviderType.ANTHROPIC, 200_000, 3.0,  15.0,  "chat",      ["fast","balanced"], "Best balance of speed and intelligence", is_latest=True),
    CloudModel("claude-haiku-4-5-20251001",  "Claude Haiku 4.5",          ProviderType.ANTHROPIC, 200_000, 0.25, 1.25,  "chat",      ["fast","cheap"], "Fastest, most affordable Claude"),
    CloudModel("claude-opus-4-0",            "Claude Opus 4",             ProviderType.ANTHROPIC, 200_000, 15.0, 75.0,  "reasoning", ["powerful"], "Previous generation flagship"),
    CloudModel("claude-sonnet-4-0",          "Claude Sonnet 4",           ProviderType.ANTHROPIC, 200_000, 3.0,  15.0,  "chat",      ["balanced"], "Previous gen balanced model"),

    # ── OpenAI ───────────────────────────────────────────────────────────────
    CloudModel("gpt-4o",                "GPT-4o",               ProviderType.OPENAI, 128_000, 5.0,  15.0,  "chat",      ["vision","fast"], "OpenAI multimodal flagship", is_latest=True),
    CloudModel("gpt-4o-mini",           "GPT-4o Mini",          ProviderType.OPENAI, 128_000, 0.15, 0.6,   "chat",      ["cheap","fast"], "Fast and affordable"),
    CloudModel("o3",                    "o3",                   ProviderType.OPENAI, 200_000, 10.0, 40.0,  "reasoning", ["reasoning","powerful"], "Advanced reasoning model"),
    CloudModel("o4-mini",               "o4-mini",              ProviderType.OPENAI, 200_000, 1.1,  4.4,   "reasoning", ["reasoning","fast"], "Fast reasoning model", is_latest=True),
    CloudModel("gpt-4-turbo",           "GPT-4 Turbo",          ProviderType.OPENAI, 128_000, 10.0, 30.0,  "chat",      ["powerful"], "Previous generation flagship"),

    # ── Groq (ultra-fast inference) ──────────────────────────────────────────
    CloudModel("llama-3.3-70b-versatile",    "Llama 3.3 70B",        ProviderType.GROQ, 128_000, 0.59, 0.79, "chat",      ["fast","open"], "Fast open-source via Groq", is_latest=True),
    CloudModel("llama-3.1-8b-instant",       "Llama 3.1 8B Instant", ProviderType.GROQ, 128_000, 0.05, 0.08, "chat",      ["ultra-fast","cheap"], "Sub-100ms responses"),
    CloudModel("deepseek-r1-distill-llama-70b", "DeepSeek R1 70B",   ProviderType.GROQ, 128_000, 0.75, 0.99, "reasoning", ["reasoning","fast"], "Open reasoning model via Groq"),
    CloudModel("gemma2-9b-it",               "Gemma2 9B",            ProviderType.GROQ, 8_192,   0.2,  0.2,  "chat",      ["open","google"], "Google's open model"),
    CloudModel("mixtral-8x7b-32768",         "Mixtral 8x7B",         ProviderType.GROQ, 32_768,  0.27, 0.27, "chat",      ["open","moe"], "Mixture of experts"),

    # ── Google Gemini ─────────────────────────────────────────────────────────
    CloudModel("gemini-2.0-flash",           "Gemini 2.0 Flash",     ProviderType.GEMINI, 1_048_576, 0.1,  0.4,  "chat",      ["fast","vision","latest"], "Google's latest fast model", is_latest=True),
    CloudModel("gemini-2.0-flash-thinking",  "Gemini 2.0 Thinking",  ProviderType.GEMINI, 32_768,    0.0,  0.0,  "reasoning", ["reasoning","free"], "Thinking model (free tier)"),
    CloudModel("gemini-1.5-pro",             "Gemini 1.5 Pro",       ProviderType.GEMINI, 2_097_152, 3.5,  10.5, "chat",      ["long-ctx","vision"], "2M context window"),
    CloudModel("gemini-1.5-flash",           "Gemini 1.5 Flash",     ProviderType.GEMINI, 1_048_576, 0.075,0.3,  "chat",      ["fast","cheap"], "Fast and efficient"),

    # ── Together AI (open models) ─────────────────────────────────────────────
    CloudModel("meta-llama/Llama-3.3-70B-Instruct-Turbo", "Llama 3.3 70B Turbo", ProviderType.TOGETHER, 131_072, 0.88, 0.88, "chat", ["open","fast"], "Meta's latest, fast", is_latest=True),
    CloudModel("deepseek-ai/DeepSeek-R1",    "DeepSeek R1",          ProviderType.TOGETHER, 163_840, 3.0,  7.0,  "reasoning", ["reasoning","open"], "Open reasoning model"),
    CloudModel("Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen2.5 Coder 32B", ProviderType.TOGETHER, 32_768, 0.8, 0.8, "code", ["coding","open"], "Best open coding model"),
    CloudModel("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B",   ProviderType.TOGETHER, 32_768,  0.2,  0.2,  "chat",      ["open","small"], "Fast small model"),

    # ── Mistral AI ────────────────────────────────────────────────────────────
    CloudModel("mistral-large-latest",       "Mistral Large",        ProviderType.MISTRAL, 128_000, 2.0,  6.0,  "chat",      ["powerful","european"], "Mistral's flagship", is_latest=True),
    CloudModel("codestral-latest",           "Codestral",            ProviderType.MISTRAL, 256_000, 0.2,  0.6,  "code",      ["coding","256k"], "Best-in-class coding model"),
    CloudModel("mistral-small-latest",       "Mistral Small",        ProviderType.MISTRAL, 128_000, 0.1,  0.3,  "chat",      ["cheap","fast"], "Efficient general model"),
]

# Provider metadata
PROVIDER_META: Dict[str, Dict] = {
    "anthropic":  {"name": "Anthropic",    "icon": "🟠", "color": "from-orange-500 to-amber-500",  "base_url": "https://api.anthropic.com", "docs": "https://docs.anthropic.com",    "models_endpoint": "/v1/models"},
    "openai":     {"name": "OpenAI",       "icon": "🟢", "color": "from-green-500 to-emerald-500", "base_url": "https://api.openai.com",    "docs": "https://platform.openai.com",   "models_endpoint": "/v1/models"},
    "groq":       {"name": "Groq",         "icon": "⚡", "color": "from-yellow-400 to-orange-400", "base_url": "https://api.groq.com",      "docs": "https://console.groq.com",      "models_endpoint": "/openai/v1/models"},
    "gemini":     {"name": "Google AI",    "icon": "🔵", "color": "from-blue-500 to-cyan-500",     "base_url": "https://generativelanguage.googleapis.com", "docs": "https://ai.google.dev", "models_endpoint": None},
    "together":   {"name": "Together AI",  "icon": "🟣", "color": "from-violet-500 to-purple-500", "base_url": "https://api.together.xyz",  "docs": "https://docs.together.ai",      "models_endpoint": "/v1/models"},
    "mistral":    {"name": "Mistral AI",   "icon": "🔴", "color": "from-red-500 to-pink-500",      "base_url": "https://api.mistral.ai",    "docs": "https://docs.mistral.ai",       "models_endpoint": "/v1/models"},
    "fireworks":  {"name": "Fireworks AI", "icon": "🎆", "color": "from-pink-500 to-rose-500",     "base_url": "https://api.fireworks.ai",  "docs": "https://fireworks.ai/docs",     "models_endpoint": "/inference/v1/models"},
    "openrouter": {"name": "OpenRouter",   "icon": "🔗", "color": "from-slate-500 to-gray-500",    "base_url": "https://openrouter.ai/api", "docs": "https://openrouter.ai/docs",    "models_endpoint": "/v1/models"},
}


# ── Provider Status ───────────────────────────────────────────────────────────
@dataclass
class ProviderStatus:
    provider: ProviderType
    configured: bool          # has API key
    healthy: bool             # key is valid (checked via test request)
    latency_ms: float = 0.0
    error: Optional[str] = None
    model_count: int = 0
    last_checked: float = 0.0


# ── Cloud Provider Manager ────────────────────────────────────────────────────
class CloudProviderManager:
    """
    Manages all cloud LLM providers as unified 'hosts'.
    Provides health checks, model listing, and routing alongside Ollama.
    """

    def __init__(self):
        self._statuses: Dict[str, ProviderStatus] = {}
        self._custom_providers: Dict[str, Dict] = {}   # user-added providers

    # ── API Key resolution ────────────────────────────────────────────────────
    def _get_key(self, provider: ProviderType) -> Optional[str]:
        key_map = {
            ProviderType.ANTHROPIC:  settings.anthropic_api_key,
            ProviderType.OPENAI:     settings.openai_api_key,
            ProviderType.GROQ:       settings.groq_api_key,
            ProviderType.GEMINI:     settings.gemini_api_key,
            ProviderType.TOGETHER:   settings.together_api_key,
            ProviderType.FIREWORKS:  settings.fireworks_api_key,
            ProviderType.MISTRAL:    settings.mistral_api_key,
            ProviderType.OPENROUTER: settings.openrouter_api_key,
        }
        return key_map.get(provider)

    def configured_providers(self) -> List[ProviderType]:
        """Return providers that have API keys set."""
        return [p for p in ProviderType if p != ProviderType.OLLAMA and self._get_key(p)]

    def is_configured(self, provider: ProviderType) -> bool:
        return bool(self._get_key(provider))

    # ── Health checks ─────────────────────────────────────────────────────────
    async def check_provider(self, provider: ProviderType) -> ProviderStatus:
        """Check if a cloud provider API key is valid."""
        key = self._get_key(provider)
        if not key:
            s = ProviderStatus(provider=provider, configured=False, healthy=False, error="No API key")
            self._statuses[provider.value] = s
            return s

        start = time.time()
        try:
            healthy = await self._ping_provider(provider, key)
            latency = (time.time() - start) * 1000
            models = self.get_models_for_provider(provider)
            s = ProviderStatus(provider=provider, configured=True, healthy=healthy, latency_ms=latency,
                               model_count=len(models), last_checked=time.time())
        except Exception as e:
            latency = (time.time() - start) * 1000
            s = ProviderStatus(provider=provider, configured=True, healthy=False,
                               latency_ms=latency, error=str(e)[:120], last_checked=time.time())

        self._statuses[provider.value] = s
        return s

    async def check_all(self) -> Dict[str, ProviderStatus]:
        """Check all configured providers concurrently."""
        providers = list(ProviderType)
        providers = [p for p in providers if p != ProviderType.OLLAMA]
        results = await asyncio.gather(*[self.check_provider(p) for p in providers], return_exceptions=True)
        return {p.value: r for p, r in zip(providers, results) if isinstance(r, ProviderStatus)}

    async def _ping_provider(self, provider: ProviderType, key: str) -> bool:
        """Lightweight ping to verify API key works."""
        async with httpx.AsyncClient(timeout=8.0) as client:
            if provider == ProviderType.ANTHROPIC:
                r = await client.post("https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]})
                return r.status_code in (200, 400)   # 400 = valid key but bad request still OK

            elif provider == ProviderType.OPENAI:
                r = await client.get("https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"})
                return r.status_code == 200

            elif provider == ProviderType.GROQ:
                r = await client.get("https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {key}"})
                return r.status_code == 200

            elif provider == ProviderType.GEMINI:
                r = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
                return r.status_code == 200

            elif provider in (ProviderType.TOGETHER, ProviderType.MISTRAL,
                               ProviderType.FIREWORKS, ProviderType.OPENROUTER):
                base_urls = {
                    ProviderType.TOGETHER:   "https://api.together.xyz/v1/models",
                    ProviderType.MISTRAL:    "https://api.mistral.ai/v1/models",
                    ProviderType.FIREWORKS:  "https://api.fireworks.ai/inference/v1/models",
                    ProviderType.OPENROUTER: "https://openrouter.ai/api/v1/models",
                }
                r = await client.get(base_urls[provider],
                    headers={"Authorization": f"Bearer {key}"})
                return r.status_code == 200

        return False

    # ── Model listing ─────────────────────────────────────────────────────────
    def get_models_for_provider(self, provider: ProviderType) -> List[CloudModel]:
        return [m for m in CLOUD_CATALOG if m.provider == provider]

    def get_all_cloud_models(self, only_configured: bool = False) -> List[CloudModel]:
        configured = set(p.value for p in self.configured_providers())
        if only_configured:
            return [m for m in CLOUD_CATALOG if m.provider.value in configured]
        return list(CLOUD_CATALOG)

    def find_model(self, model_id: str) -> Optional[CloudModel]:
        return next((m for m in CLOUD_CATALOG if m.id == model_id), None)

    # ── Routing ───────────────────────────────────────────────────────────────
    def resolve_provider(self, model_id: str) -> Optional[ProviderType]:
        """Given a model ID string, determine which provider it belongs to."""
        # Explicit prefix
        for prefix, provider in [
            ("claude-", ProviderType.ANTHROPIC),
            ("gpt-",    ProviderType.OPENAI),
            ("o1-",     ProviderType.OPENAI),
            ("o3",      ProviderType.OPENAI),
            ("o4-",     ProviderType.OPENAI),
            ("text-",   ProviderType.OPENAI),
            ("gemini-", ProviderType.GEMINI),
            ("codestral", ProviderType.MISTRAL),
            ("mistral-",  ProviderType.MISTRAL),
            ("llama-3.1-8b-instant",   ProviderType.GROQ),
            ("llama-3.3-70b-versatile", ProviderType.GROQ),
            ("deepseek-r1-distill",    ProviderType.GROQ),
            ("gemma2-",  ProviderType.GROQ),
            ("mixtral-", ProviderType.GROQ),
        ]:
            if model_id.startswith(prefix):
                return provider

        # Check catalog
        m = self.find_model(model_id)
        if m:
            return m.provider

        # Provider/model format
        if "/" in model_id:
            org = model_id.split("/")[0].lower()
            org_map = {"meta-llama": ProviderType.TOGETHER, "deepseek-ai": ProviderType.TOGETHER,
                       "qwen": ProviderType.TOGETHER, "mistralai": ProviderType.TOGETHER,
                       "google": ProviderType.GEMINI}
            for k, v in org_map.items():
                if org.startswith(k):
                    return v

        return None

    def is_cloud_model(self, model_id: str) -> bool:
        """Return True if this looks like a cloud model (not an Ollama model)."""
        if ":" in model_id:
            return False  # Ollama format: name:tag
        if model_id.startswith("ollama/"):
            return False
        return self.resolve_provider(model_id) is not None

    def get_litellm_model_id(self, model_id: str) -> str:
        """Convert to LiteLLM-compatible model ID for routing."""
        provider = self.resolve_provider(model_id)
        if not provider:
            return model_id
        prefix_map = {
            ProviderType.ANTHROPIC:  "",          # claude-* works natively
            ProviderType.OPENAI:     "",           # gpt-* works natively
            ProviderType.GROQ:       "groq/",
            ProviderType.GEMINI:     "gemini/",
            ProviderType.TOGETHER:   "together_ai/",
            ProviderType.MISTRAL:    "mistral/",
            ProviderType.FIREWORKS:  "fireworks_ai/",
            ProviderType.OPENROUTER: "openrouter/",
        }
        prefix = prefix_map.get(provider, "")
        if prefix and not model_id.startswith(prefix):
            return f"{prefix}{model_id}"
        return model_id

    def get_direct_client_kwargs(self, model_id: str) -> Dict[str, Any]:
        """
        Return kwargs for ChatOpenAI-compatible initialization
        that bypass LiteLLM and go directly to the provider's API.
        """
        provider = self.resolve_provider(model_id)
        if not provider:
            return {}

        key = self._get_key(provider)

        direct_endpoints: Dict[ProviderType, str] = {
            ProviderType.OPENAI:     "https://api.openai.com/v1",
            ProviderType.GROQ:       "https://api.groq.com/openai/v1",
            ProviderType.TOGETHER:   "https://api.together.xyz/v1",
            ProviderType.MISTRAL:    "https://api.mistral.ai/v1",
            ProviderType.FIREWORKS:  "https://api.fireworks.ai/inference/v1",
            ProviderType.OPENROUTER: "https://openrouter.ai/api/v1",
        }

        if provider == ProviderType.ANTHROPIC:
            # Use langchain-anthropic directly
            return {"__provider": "anthropic", "__key": key}

        if provider == ProviderType.GEMINI:
            # Use LiteLLM gateway for Gemini (needs key remapping)
            return {"__provider": "litellm", "__model": f"gemini/{model_id}"}

        base_url = direct_endpoints.get(provider)
        if base_url and key:
            return {"base_url": base_url, "api_key": key}

        return {}

    # ── Status snapshot ───────────────────────────────────────────────────────
    def get_status_snapshot(self) -> List[Dict]:
        """Return current status for all providers."""
        result = []
        for ptype in ProviderType:
            if ptype == ProviderType.OLLAMA:
                continue
            key = self._get_key(ptype)
            cached = self._statuses.get(ptype.value)
            meta = PROVIDER_META.get(ptype.value, {})
            models = self.get_models_for_provider(ptype)
            result.append({
                "id":          ptype.value,
                "name":        meta.get("name", ptype.value),
                "icon":        meta.get("icon", "🤖"),
                "color":       meta.get("color", "from-gray-500 to-gray-600"),
                "docs":        meta.get("docs", ""),
                "configured":  bool(key),
                "healthy":     cached.healthy if cached else None,
                "latency_ms":  cached.latency_ms if cached else None,
                "error":       cached.error if cached else None,
                "model_count": len(models),
                "last_checked": cached.last_checked if cached else None,
            })
        return result

    def add_custom_provider(self, provider_id: str, name: str, base_url: str, api_key: str):
        """Register a custom OpenAI-compatible provider endpoint."""
        self._custom_providers[provider_id] = {"name": name, "base_url": base_url, "api_key": api_key}
        logger.info("Custom provider added", id=provider_id, url=base_url)

    def get_routing_table(self) -> Dict[str, Dict]:
        """Return a full routing table: model_id → {provider, base_url, key, litellm_id}."""
        table = {}
        for model in CLOUD_CATALOG:
            provider = model.provider
            key = self._get_key(provider)
            kwargs = self.get_direct_client_kwargs(model.id)
            table[model.id] = {
                "provider":    provider.value,
                "configured":  bool(key),
                "litellm_id":  self.get_litellm_model_id(model.id),
                "base_url":    kwargs.get("base_url"),
                "has_key":     bool(kwargs.get("api_key") or kwargs.get("__key")),
            }
        return table


# Global singleton
cloud_manager = CloudProviderManager()
