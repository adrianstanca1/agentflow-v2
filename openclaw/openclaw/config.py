"""
OpenClaw — Config System
Persistent settings, model profiles, per-project .openclaw config.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


CONFIG_DIR = Path.home() / ".config" / "openclaw"
GLOBAL_CONFIG = CONFIG_DIR / "config.json"
PROJECT_CONFIG = ".openclaw.json"


@dataclass
class ModelProfile:
    name: str
    context_window: int = 32768
    temperature: float = 0.2
    system_addendum: str = ""  # Extra system prompt text for this model


# Recommended Ollama models with their ideal settings
RECOMMENDED_MODELS: Dict[str, ModelProfile] = {
    "qwen2.5-coder:7b": ModelProfile("qwen2.5-coder:7b", 32768, 0.2),
    "qwen2.5-coder:14b": ModelProfile("qwen2.5-coder:14b", 32768, 0.2),
    "qwen2.5-coder:32b": ModelProfile("qwen2.5-coder:32b", 32768, 0.2),
    "deepseek-coder-v2:16b": ModelProfile("deepseek-coder-v2:16b", 65536, 0.2),
    "deepseek-r1:8b": ModelProfile("deepseek-r1:8b", 32768, 0.3,
        system_addendum="Think step by step before coding."),
    "deepseek-r1:14b": ModelProfile("deepseek-r1:14b", 65536, 0.3,
        system_addendum="Think step by step before coding."),
    "llama3.1:8b": ModelProfile("llama3.1:8b", 32768, 0.3),
    "llama3.2:latest": ModelProfile("llama3.2:latest", 32768, 0.3),
    "llama3.3:70b": ModelProfile("llama3.3:70b", 65536, 0.2),
    "codestral:22b": ModelProfile("codestral:22b", 32768, 0.1),
    "phi4:latest": ModelProfile("phi4:latest", 16384, 0.2),
    "gemma2:9b": ModelProfile("gemma2:9b", 8192, 0.3),
    "mistral:7b": ModelProfile("mistral:7b", 32768, 0.3),
}


@dataclass
class OpenClawConfig:
    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    default_model: str = "qwen2.5-coder:7b"
    fallback_model: str = "llama3.2:latest"
    context_window: int = 32768
    temperature: float = 0.2

    # Behaviour
    max_iterations: int = 20
    auto_approve_tools: bool = True       # False = ask before each tool
    confirm_destructive: bool = True      # Ask before git reset --hard etc
    stream_output: bool = True
    show_tool_results: bool = True
    compact_tool_output: bool = False     # True = one-line tool summaries

    # Files
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", "node_modules", ".git",
        "*.log", "*.lock", "dist/", "build/", ".venv",
    ])
    max_file_size_kb: int = 500           # Skip files larger than this in context

    # History
    history_file: str = str(Path.home() / ".openclaw_history")
    max_history: int = 1000

    # UI
    theme: str = "monokai"               # Rich syntax theme
    show_banner: bool = True

    @classmethod
    def load(cls, project_dir: Optional[str] = None) -> "OpenClawConfig":
        """Load config: defaults → global → project-level."""
        cfg = cls()

        # Global config
        if GLOBAL_CONFIG.exists():
            try:
                data = json.loads(GLOBAL_CONFIG.read_text())
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            except Exception:
                pass

        # Project config
        if project_dir:
            project_cfg = Path(project_dir) / PROJECT_CONFIG
            if project_cfg.exists():
                try:
                    data = json.loads(project_cfg.read_text())
                    for k, v in data.items():
                        if hasattr(cfg, k):
                            setattr(cfg, k, v)
                except Exception:
                    pass

        # Env overrides
        if os.getenv("OLLAMA_URL"):
            cfg.ollama_url = os.getenv("OLLAMA_URL")
        if os.getenv("OPENCLAW_MODEL"):
            cfg.default_model = os.getenv("OPENCLAW_MODEL")
        if os.getenv("OPENCLAW_CTX"):
            cfg.context_window = int(os.getenv("OPENCLAW_CTX"))

        return cfg

    def save_global(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        GLOBAL_CONFIG.write_text(json.dumps(data, indent=2))

    def save_project(self, project_dir: str):
        p = Path(project_dir) / PROJECT_CONFIG
        data = asdict(self)
        p.write_text(json.dumps(data, indent=2))

    def get_model_profile(self, model: str) -> ModelProfile:
        return RECOMMENDED_MODELS.get(model, ModelProfile(model, self.context_window, self.temperature))

    def to_display(self) -> str:
        lines = [
            f"  ollama_url:      {self.ollama_url}",
            f"  default_model:   {self.default_model}",
            f"  context_window:  {self.context_window:,}",
            f"  temperature:     {self.temperature}",
            f"  max_iterations:  {self.max_iterations}",
            f"  auto_approve:    {self.auto_approve_tools}",
            f"  stream_output:   {self.stream_output}",
            f"  theme:           {self.theme}",
        ]
        return "\n".join(lines)


def get_recommended_models() -> List[str]:
    return list(RECOMMENDED_MODELS.keys())


def detect_best_model(available: List[str]) -> Optional[str]:
    """Pick the best available coding model from what's installed."""
    priority = [
        "qwen2.5-coder:32b",
        "deepseek-coder-v2:16b",
        "qwen2.5-coder:14b",
        "deepseek-r1:14b",
        "qwen2.5-coder:7b",
        "deepseek-r1:8b",
        "codestral:22b",
        "llama3.3:70b",
        "llama3.1:8b",
        "phi4:latest",
        "mistral:7b",
        "llama3.2:latest",
    ]
    available_set = set(available)
    for model in priority:
        if model in available_set:
            return model
    # Fuzzy match — any qwen or coder
    for m in available:
        if any(kw in m.lower() for kw in ["coder", "qwen", "deepseek", "codestral"]):
            return m
    return available[0] if available else None
