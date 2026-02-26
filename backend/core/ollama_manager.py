"""
AgentFlow v2 — Ollama Manager
Full lifecycle management for local + remote Ollama instances.
Supports multi-host, GPU detection, model registry, auto-pull, health monitoring.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import structlog

logger = structlog.get_logger()


class OllamaHostType(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"
    CLOUD = "cloud"    # Ollama Cloud / hosted instances


@dataclass
class OllamaHost:
    url: str
    host_type: OllamaHostType = OllamaHostType.LOCAL
    name: str = ""
    api_key: Optional[str] = None          # For Ollama Cloud / authenticated remotes
    gpu_count: int = 0
    gpu_model: str = ""
    is_healthy: bool = False
    latency_ms: float = 0
    models: List[Dict[str, Any]] = field(default_factory=list)
    last_checked: float = 0

    def __post_init__(self):
        if not self.name:
            parsed = urlparse(self.url)
            self.name = parsed.netloc or parsed.path


@dataclass
class ModelInfo:
    name: str
    size_bytes: int
    digest: str
    modified_at: str
    host_url: str
    details: Dict[str, Any] = field(default_factory=dict)
    is_running: bool = False
    context_length: int = 4096

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def size_label(self) -> str:
        gb = self.size_gb
        if gb >= 1:
            return f"{gb:.1f}GB"
        return f"{self.size_bytes / (1024**2):.0f}MB"

    @property
    def family(self) -> str:
        """Extract model family from name."""
        n = self.name.lower().split(":")[0]
        for fam in ["llama", "qwen", "deepseek", "mistral", "phi", "gemma", "codellama", "nomic", "mxbai"]:
            if fam in n:
                return fam
        return "other"


# ── Popular models registry ────────────────────────────────────────────────────
POPULAR_MODELS = [
    # Coding
    {"name": "qwen2.5-coder:7b",      "category": "coding",   "size": "4.7GB",  "description": "Best coding model for its size. Handles Python, JS, Go, Rust perfectly.", "tags": ["coding", "fast"]},
    {"name": "qwen2.5-coder:32b",     "category": "coding",   "size": "19GB",   "description": "Near-frontier coding quality, open weights.", "tags": ["coding", "powerful"]},
    {"name": "codellama:13b",         "category": "coding",   "size": "7.4GB",  "description": "Meta's dedicated code model. Strong at code completion.", "tags": ["coding"]},
    {"name": "deepseek-coder-v2",     "category": "coding",   "size": "9GB",    "description": "DeepSeek's coding specialist with excellent reasoning.", "tags": ["coding", "reasoning"]},
    # General purpose
    {"name": "llama3.2:3b",           "category": "general",  "size": "2.0GB",  "description": "Fastest local model. Great for quick tasks.", "tags": ["fast", "small"]},
    {"name": "llama3.2:latest",       "category": "general",  "size": "2.0GB",  "description": "Meta's latest efficient model.", "tags": ["fast"]},
    {"name": "llama3.1:8b",           "category": "general",  "size": "4.7GB",  "description": "Excellent all-rounder with tool use support.", "tags": ["tools", "balanced"]},
    {"name": "llama3.1:70b",          "category": "general",  "size": "40GB",   "description": "Near-GPT-4 quality, fully open.", "tags": ["powerful", "large"]},
    {"name": "qwen2.5:7b",            "category": "general",  "size": "4.7GB",  "description": "Strong multilingual support, good reasoning.", "tags": ["multilingual", "balanced"]},
    {"name": "qwen2.5:32b",           "category": "general",  "size": "19GB",   "description": "Powerful reasoning with 128K context.", "tags": ["reasoning", "long-context"]},
    # Reasoning
    {"name": "deepseek-r1:8b",        "category": "reasoning","size": "4.9GB",  "description": "Chain-of-thought reasoning model. Thinks before answering.", "tags": ["reasoning", "thinking"]},
    {"name": "deepseek-r1:32b",       "category": "reasoning","size": "19GB",   "description": "Strong reasoning, near o1-level on benchmarks.", "tags": ["reasoning", "powerful"]},
    {"name": "phi4:latest",           "category": "reasoning","size": "9.1GB",  "description": "Microsoft's Phi-4. Exceptional reasoning per parameter.", "tags": ["reasoning", "efficient"]},
    # Embeddings
    {"name": "nomic-embed-text",      "category": "embedding","size": "274MB",  "description": "Best open-source text embeddings. 8K context.", "tags": ["embedding", "rag"]},
    {"name": "mxbai-embed-large",     "category": "embedding","size": "670MB",  "description": "High-quality embeddings for RAG pipelines.", "tags": ["embedding", "rag"]},
    {"name": "bge-m3",                "category": "embedding","size": "1.2GB",  "description": "Multilingual embeddings. Best for diverse corpora.", "tags": ["embedding", "multilingual"]},
    # Vision
    {"name": "llava:13b",             "category": "vision",   "size": "8.0GB",  "description": "Multimodal vision-language model.", "tags": ["vision", "multimodal"]},
    {"name": "moondream2",            "category": "vision",   "size": "1.7GB",  "description": "Tiny but capable vision model.", "tags": ["vision", "small"]},
    {"name": "minicpm-v:8b",          "category": "vision",   "size": "5.5GB",  "description": "Strong OCR and document understanding.", "tags": ["vision", "ocr"]},
]


# ── Ollama Manager ─────────────────────────────────────────────────────────────
class OllamaManager:
    """
    Manages multiple Ollama instances (local + remote + cloud).
    Provides unified API for model management, inference routing, and health monitoring.
    """

    def __init__(self):
        self.hosts: Dict[str, OllamaHost] = {}
        self._client = httpx.AsyncClient(timeout=120.0)
        self._health_task: Optional[asyncio.Task] = None

    def add_host(
        self,
        url: str,
        name: Optional[str] = None,
        host_type: OllamaHostType = OllamaHostType.LOCAL,
        api_key: Optional[str] = None,
    ) -> OllamaHost:
        """Register an Ollama host."""
        host = OllamaHost(
            url=url.rstrip("/"),
            host_type=host_type,
            name=name or url,
            api_key=api_key,
        )
        self.hosts[url] = host
        logger.info("Ollama host registered", url=url, type=host_type)
        return host

    def _get_headers(self, host: OllamaHost) -> Dict[str, str]:
        """Build auth headers for a host."""
        headers = {"Content-Type": "application/json"}
        if host.api_key:
            headers["Authorization"] = f"Bearer {host.api_key}"
        return headers

    # ── Health & Discovery ──────────────────────────────────────────────────
    async def check_health(self, host: OllamaHost) -> bool:
        """Check if an Ollama host is reachable."""
        start = time.time()
        try:
            resp = await self._client.get(
                f"{host.url}/api/tags",
                headers=self._get_headers(host),
                timeout=5.0,
            )
            host.latency_ms = (time.time() - start) * 1000
            host.is_healthy = resp.status_code == 200
            host.last_checked = time.time()

            if host.is_healthy:
                data = resp.json()
                host.models = data.get("models", [])

            # Try to get GPU info
            await self._detect_gpu(host)
            return host.is_healthy
        except Exception as e:
            host.is_healthy = False
            host.last_checked = time.time()
            logger.debug("Ollama host unhealthy", url=host.url, error=str(e))
            return False

    async def _detect_gpu(self, host: OllamaHost):
        """Detect GPU info from running models or system info."""
        try:
            resp = await self._client.get(
                f"{host.url}/api/ps",
                headers=self._get_headers(host),
                timeout=3.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                models_running = data.get("models", [])
                if models_running:
                    first = models_running[0]
                    size_vram = first.get("size_vram", 0)
                    host.gpu_count = 1 if size_vram > 0 else 0
        except Exception:
            pass

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered hosts."""
        results = {}
        tasks = [self.check_health(host) for host in self.hosts.values()]
        statuses = await asyncio.gather(*tasks, return_exceptions=True)
        for host, status in zip(self.hosts.values(), statuses):
            results[host.url] = bool(status) if not isinstance(status, Exception) else False
        return results

    async def start_health_monitor(self, interval: int = 30):
        """Start background health monitoring."""
        async def _monitor():
            while True:
                await self.health_check_all()
                await asyncio.sleep(interval)
        self._health_task = asyncio.create_task(_monitor())

    # ── Model Management ────────────────────────────────────────────────────
    async def list_models(self, host_url: Optional[str] = None) -> List[ModelInfo]:
        """List all models across all hosts (or a specific host)."""
        models = []
        hosts = [self.hosts[host_url]] if host_url and host_url in self.hosts else list(self.hosts.values())

        for host in hosts:
            if not host.is_healthy:
                continue
            try:
                resp = await self._client.get(
                    f"{host.url}/api/tags",
                    headers=self._get_headers(host),
                )
                data = resp.json()
                for m in data.get("models", []):
                    models.append(ModelInfo(
                        name=m["name"],
                        size_bytes=m.get("size", 0),
                        digest=m.get("digest", ""),
                        modified_at=m.get("modified_at", ""),
                        host_url=host.url,
                        details=m.get("details", {}),
                        context_length=m.get("details", {}).get("context_length", 4096),
                    ))
            except Exception as e:
                logger.error("Failed to list models", host=host.url, error=str(e))

        return models

    async def list_running(self, host_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """List currently running/loaded models."""
        running = []
        hosts = [self.hosts[host_url]] if host_url and host_url in self.hosts else list(self.hosts.values())
        for host in hosts:
            if not host.is_healthy:
                continue
            try:
                resp = await self._client.get(f"{host.url}/api/ps", headers=self._get_headers(host))
                data = resp.json()
                for m in data.get("models", []):
                    m["host_url"] = host.url
                    running.append(m)
            except Exception:
                pass
        return running

    async def pull_model(
        self,
        model_name: str,
        host_url: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Pull a model with streaming progress updates.
        Yields progress dicts: {status, digest, total, completed, percent}
        """
        target_url = host_url or self._get_best_host()
        if not target_url:
            yield {"status": "error", "error": "No healthy Ollama hosts available"}
            return

        host = self.hosts.get(target_url)
        if not host:
            yield {"status": "error", "error": f"Host {target_url} not found"}
            return

        logger.info("Pulling model", model=model_name, host=target_url)

        try:
            async with self._client.stream(
                "POST",
                f"{host.url}/api/pull",
                json={"name": model_name, "stream": True},
                headers=self._get_headers(host),
                timeout=None,  # Pull can take a long time
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        percent = (completed / total * 100) if total > 0 else 0
                        yield {
                            "status": data.get("status", ""),
                            "digest": data.get("digest", ""),
                            "total": total,
                            "completed": completed,
                            "percent": round(percent, 1),
                            "model": model_name,
                            "host": target_url,
                        }
                    except json.JSONDecodeError:
                        continue

            # Refresh model list
            await self.check_health(host)
            yield {"status": "success", "model": model_name, "percent": 100}

        except Exception as e:
            logger.error("Pull failed", model=model_name, error=str(e))
            yield {"status": "error", "error": str(e), "model": model_name}

    async def delete_model(self, model_name: str, host_url: Optional[str] = None) -> bool:
        """Delete a model from a host."""
        target_url = host_url or self._find_model_host(model_name)
        if not target_url:
            return False
        host = self.hosts.get(target_url)
        if not host:
            return False
        try:
            resp = await self._client.delete(
                f"{host.url}/api/delete",
                json={"name": model_name},
                headers=self._get_headers(host),
            )
            await self.check_health(host)
            return resp.status_code == 200
        except Exception as e:
            logger.error("Delete failed", model=model_name, error=str(e))
            return False

    async def copy_model(self, source: str, destination: str, host_url: Optional[str] = None) -> bool:
        """Copy/rename a model."""
        target_url = host_url or self._find_model_host(source)
        if not target_url:
            return False
        host = self.hosts.get(target_url)
        try:
            resp = await self._client.post(
                f"{host.url}/api/copy",
                json={"source": source, "destination": destination},
                headers=self._get_headers(host),
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def show_model_info(self, model_name: str, host_url: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed model information (modelfile, parameters, template)."""
        target_url = host_url or self._find_model_host(model_name)
        if not target_url:
            return {}
        host = self.hosts.get(target_url)
        try:
            resp = await self._client.post(
                f"{host.url}/api/show",
                json={"name": model_name},
                headers=self._get_headers(host),
            )
            return resp.json()
        except Exception:
            return {}

    async def create_modelfile(
        self,
        model_name: str,
        modelfile: str,
        host_url: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create a custom model from a Modelfile."""
        target_url = host_url or self._get_best_host()
        if not target_url:
            yield {"status": "error", "error": "No healthy hosts"}
            return
        host = self.hosts[target_url]
        try:
            async with self._client.stream(
                "POST",
                f"{host.url}/api/create",
                json={"name": model_name, "modelfile": modelfile, "stream": True},
                headers=self._get_headers(host),
                timeout=None,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue
        except Exception as e:
            yield {"status": "error", "error": str(e)}

    # ── Inference ───────────────────────────────────────────────────────────
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        host_url: Optional[str] = None,
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Raw generation with streaming."""
        target_url = host_url or self._find_model_host(model) or self._get_best_host()
        if not target_url:
            yield "Error: No Ollama hosts available"
            return

        host = self.hosts[target_url]
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options or {},
        }
        if system:
            payload["system"] = system

        try:
            async with self._client.stream(
                "POST",
                f"{host.url}/api/generate",
                json=payload,
                headers=self._get_headers(host),
                timeout=None,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("response"):
                                yield data["response"]
                            if data.get("done"):
                                break
                        except Exception:
                            continue
        except Exception as e:
            yield f"Error: {str(e)}"

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        host_url: Optional[str] = None,
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """OpenAI-compatible chat with streaming."""
        target_url = host_url or self._find_model_host(model) or self._get_best_host()
        if not target_url:
            yield {"error": "No Ollama hosts available"}
            return

        host = self.hosts[target_url]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options or {},
        }
        if tools:
            payload["tools"] = tools

        try:
            async with self._client.stream(
                "POST",
                f"{host.url}/api/chat",
                json=payload,
                headers=self._get_headers(host),
                timeout=None,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            yield data
                            if data.get("done"):
                                break
                        except Exception:
                            continue
        except Exception as e:
            yield {"error": str(e), "done": True}

    async def embed(
        self,
        model: str,
        input_text: str | List[str],
        host_url: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embeddings using Ollama embedding models."""
        target_url = host_url or self._find_model_host(model) or self._get_best_host()
        if not target_url:
            return []

        host = self.hosts[target_url]
        texts = input_text if isinstance(input_text, list) else [input_text]

        try:
            resp = await self._client.post(
                f"{host.url}/api/embed",
                json={"model": model, "input": texts},
                headers=self._get_headers(host),
            )
            data = resp.json()
            return data.get("embeddings", [])
        except Exception as e:
            logger.error("Embedding failed", error=str(e))
            return []

    # ── Routing ─────────────────────────────────────────────────────────────
    def _get_best_host(self) -> Optional[str]:
        """Get the URL of the fastest healthy host."""
        healthy = [(h.latency_ms or 9999, h.url) for h in self.hosts.values() if h.is_healthy]
        if not healthy:
            return None
        return sorted(healthy)[0][1]

    def _find_model_host(self, model_name: str) -> Optional[str]:
        """Find which host has a specific model."""
        base_name = model_name.split(":")[0].lower()
        for host in self.hosts.values():
            if not host.is_healthy:
                continue
            for m in host.models:
                if base_name in m.get("name", "").lower():
                    return host.url
        return None

    def get_openai_base_url(self, host_url: Optional[str] = None) -> str:
        """Get OpenAI-compatible base URL for LangChain/LiteLLM integration."""
        url = host_url or self._get_best_host() or "http://localhost:11434"
        return f"{url}/v1"

    # ── Stats ───────────────────────────────────────────────────────────────
    async def get_stats(self) -> Dict[str, Any]:
        """Get aggregate stats across all hosts."""
        all_models = await self.list_models()
        running = await self.list_running()
        healthy_hosts = sum(1 for h in self.hosts.values() if h.is_healthy)

        return {
            "total_hosts": len(self.hosts),
            "healthy_hosts": healthy_hosts,
            "total_models": len(all_models),
            "running_models": len(running),
            "hosts": [
                {
                    "url": h.url,
                    "name": h.name,
                    "type": h.host_type,
                    "is_healthy": h.is_healthy,
                    "latency_ms": round(h.latency_ms, 1),
                    "model_count": len(h.models),
                    "gpu_count": h.gpu_count,
                    "gpu_model": h.gpu_model,
                }
                for h in self.hosts.values()
            ],
        }

    async def close(self):
        if self._health_task:
            self._health_task.cancel()
        await self._client.aclose()


# ── Global instance ────────────────────────────────────────────────────────────
ollama_manager = OllamaManager()


def initialize_ollama(hosts: Optional[List[Dict[str, Any]]] = None):
    """Initialize Ollama manager with configured hosts."""
    from .config import settings

    # Always add local Ollama
    ollama_manager.add_host(
        url=settings.ollama_url,
        name="Local Ollama",
        host_type=OllamaHostType.LOCAL,
    )

    # Add remote hosts from config
    for h in (hosts or settings.ollama_remote_hosts_list):
        ollama_manager.add_host(
            url=h.get("url", ""),
            name=h.get("name", "Remote"),
            host_type=OllamaHostType.REMOTE,
            api_key=h.get("api_key"),
        )

    logger.info("Ollama manager initialized", hosts=len(ollama_manager.hosts))
    return ollama_manager
