"""
OpenClaw — Agentic Coding Engine
Works with ANY OpenAI-compatible provider: Ollama, Groq, OpenAI, Anthropic,
Together, Mistral, Gemini, OpenRouter. Same tool-calling loop everywhere.
"""
from __future__ import annotations

import asyncio
import difflib
import fnmatch
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    for _env in [Path(__file__).resolve().parent.parent.parent / ".env",
                 Path.home() / ".config" / "openclaw" / ".env"]:
        if _env.exists():
            load_dotenv(_env, override=False)
            break
except ImportError:
    pass

# ── Provider registry ─────────────────────────────────────────────────────────
PROVIDERS: Dict[str, Dict] = {
    "ollama": {
        "name": "Ollama",
        "icon": "🦙",
        "base_url_env": "OLLAMA_URL",
        "base_url_default": "http://localhost:11434",
        "base_url_suffix": "/v1",
        "api_key_env": "OLLAMA_API_KEY",
        "api_key_default": "ollama",
        "free": True,
        "default_model": "qwen2.5-coder:7b",
    },
    "groq": {
        "name": "Groq",
        "icon": "⚡",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "free": True,
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "name": "OpenAI",
        "icon": "🟢",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "free": False,
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "name": "Anthropic",
        "icon": "🟠",
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "free": False,
        "default_model": "claude-haiku-4-5-20251001",
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
    "together": {
        "name": "Together AI",
        "icon": "🟣",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "free": True,
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "mistral": {
        "name": "Mistral AI",
        "icon": "🔴",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "free": False,
        "default_model": "mistral-large-latest",
    },
    "gemini": {
        "name": "Google Gemini",
        "icon": "🔵",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "free": True,
        "default_model": "gemini-2.0-flash",
    },
    "openrouter": {
        "name": "OpenRouter",
        "icon": "🔗",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "free": True,
        "default_model": "meta-llama/llama-3.3-70b-instruct",
    },
}


def _get_provider_config(provider: str) -> Dict:
    cfg = dict(PROVIDERS.get(provider, PROVIDERS["ollama"]))
    if "base_url_env" in cfg:
        raw = os.getenv(cfg["base_url_env"], cfg["base_url_default"])
        cfg["resolved_base_url"] = raw.rstrip("/") + cfg.get("base_url_suffix", "")
    else:
        cfg["resolved_base_url"] = cfg["base_url"]
    key_env = cfg.get("api_key_env", "")
    cfg["resolved_api_key"] = os.getenv(key_env, "") or cfg.get("api_key_default", "")
    return cfg


def detect_provider() -> str:
    explicit = os.getenv("OPENCLAW_PROVIDER", "")
    if explicit and explicit in PROVIDERS:
        return explicit
    for p in ["groq", "openai", "anthropic", "together", "mistral", "gemini", "openrouter"]:
        if os.getenv(PROVIDERS[p]["api_key_env"], ""):
            return p
    return "ollama"


def detect_model(provider: str) -> str:
    explicit = os.getenv("OPENCLAW_MODEL", "")
    if explicit:
        return explicit
    return PROVIDERS.get(provider, PROVIDERS["ollama"]).get("default_model", "llama3.2")


# ── Unified OpenAI-compatible client ─────────────────────────────────────────
def _make_client(provider: str):
    from openai import AsyncOpenAI
    cfg = _get_provider_config(provider)
    kwargs: Dict = {"api_key": cfg["resolved_api_key"], "base_url": cfg["resolved_base_url"]}
    if cfg.get("extra_headers"):
        kwargs["default_headers"] = cfg["extra_headers"]
    return AsyncOpenAI(**kwargs)


async def provider_chat(
    messages: List[Dict],
    model: str,
    provider: str = "ollama",
    tools: Optional[List[Dict]] = None,
    on_token=None,
    temperature: float = 0.2,
    max_tokens: int = 8192,
) -> Tuple[str, Optional[List[Dict]]]:
    client = _make_client(provider)
    kwargs: Dict[str, Any] = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens, "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    full_content = ""
    tool_calls_raw: Dict[int, Dict] = {}

    try:
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            if delta.content:
                full_content += delta.content
                if on_token:
                    on_token(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_raw:
                        tool_calls_raw[idx] = {"id": tc.id or f"call_{idx}",
                                                "type": "function",
                                                "function": {"name": "", "arguments": ""}}
                    r = tool_calls_raw[idx]
                    if tc.function:
                        if tc.function.name: r["function"]["name"] += tc.function.name
                        if tc.function.arguments: r["function"]["arguments"] += tc.function.arguments
                        if tc.id: r["id"] = tc.id
    except Exception as e:
        err = str(e)
        cfg = _get_provider_config(provider)
        if "401" in err or "authentication" in err.lower():
            raise RuntimeError(f"[{provider}] Auth failed — check your {cfg.get('api_key_env','API key')}.")
        if "connect" in err.lower():
            raise RuntimeError(f"[{provider}] Cannot reach {cfg['resolved_base_url']} — is it running?")
        raise

    return full_content, list(tool_calls_raw.values()) or None


async def check_provider(provider: str) -> Dict:
    cfg = _get_provider_config(provider)
    key = cfg["resolved_api_key"]
    if not key and provider != "ollama":
        return {"ok": False, "error": f"No {cfg.get('api_key_env','API key')} set", "provider": provider}
    try:
        import time
        start = time.time()
        models = await _make_client(provider).models.list()
        return {"ok": True, "latency_ms": round((time.time()-start)*1000),
                "provider": provider, "url": cfg["resolved_base_url"],
                "model_count": len(models.data)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:120], "provider": provider}


async def list_provider_models(provider: str) -> List[str]:
    try:
        models = await _make_client(provider).models.list()
        return sorted(m.id for m in models.data)
    except Exception:
        return PROVIDERS.get(provider, {}).get("fallback_models", [])


# ── Tools ─────────────────────────────────────────────────────────────────────
TOOLS = [
    {"type":"function","function":{"name":"read_file","description":"Read a file before editing. Always do this first.","parameters":{"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"},"end_line":{"type":"integer"}},"required":["path"]}}},
    {"type":"function","function":{"name":"write_file","description":"Create or overwrite a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"edit_file","description":"Replace an exact unique string in a file — prefer this over write_file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"old_str":{"type":"string"},"new_str":{"type":"string"}},"required":["path","old_str","new_str"]}}},
    {"type":"function","function":{"name":"run_command","description":"Execute a shell command for tests, builds, linting, installs.","parameters":{"type":"object","properties":{"command":{"type":"string"},"working_dir":{"type":"string"},"timeout":{"type":"integer"}},"required":["command"]}}},
    {"type":"function","function":{"name":"search_files","description":"Search for a pattern across files.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"file_pattern":{"type":"string"},"case_sensitive":{"type":"boolean"}},"required":["pattern"]}}},
    {"type":"function","function":{"name":"list_directory","description":"Show directory tree.","parameters":{"type":"object","properties":{"path":{"type":"string"},"depth":{"type":"integer"},"show_hidden":{"type":"boolean"}}}}},
    {"type":"function","function":{"name":"git_command","description":"Run any git subcommand.","parameters":{"type":"object","properties":{"subcommand":{"type":"string"}},"required":["subcommand"]}}},
    {"type":"function","function":{"name":"glob_files","description":"Find files by glob pattern.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}}},
    {"type":"function","function":{"name":"fetch_url","description":"Fetch a URL and return its text content.","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}},
]


# ── Tool executor ─────────────────────────────────────────────────────────────
class ToolExecutor:
    def __init__(self, cwd: str):
        self.cwd = Path(cwd).resolve()
        self._history: Dict[str, List[str]] = {}

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return (self.cwd / path).resolve() if not path.is_absolute() else path.resolve()

    def _snapshot(self, p: Path):
        try:
            self._history.setdefault(str(p), []).append(p.read_text(errors="replace"))
        except Exception:
            pass

    def undo_file(self, path: str) -> bool:
        key = str(self._resolve(path))
        h = self._history.get(key, [])
        if not h: return False
        self._resolve(path).write_text(h.pop())
        return True

    async def execute(self, name: str, args: Dict) -> str:
        fn = getattr(self, f"_{name}", None)
        if not fn:
            return f"Unknown tool: {name}"
        try:
            return await fn(**args)
        except TypeError as e:
            return f"Bad args for {name}: {e}"
        except Exception as e:
            return f"Error in {name}: {e}"

    async def _read_file(self, path: str, start_line: int = None, end_line: int = None) -> str:
        p = self._resolve(path)
        if not p.exists(): return f"Not found: {path}"
        lines = p.read_text(errors="replace").splitlines(keepends=True)
        if start_line or end_line:
            s, e = (start_line or 1) - 1, end_line or len(lines)
            lines = lines[s:e]
        start = start_line or 1
        return f"[{path}]\n" + "".join(f"{start+i:4d}│ {l}" for i, l in enumerate(lines))

    async def _write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if p.exists(): self._snapshot(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"✓ Written {path} ({content.count(chr(10))+1} lines)"

    async def _edit_file(self, path: str, old_str: str, new_str: str) -> str:
        p = self._resolve(path)
        if not p.exists(): return f"Not found: {path}"
        self._snapshot(p)
        txt = p.read_text(errors="replace")
        n = txt.count(old_str)
        if n == 0: return f"String not found in {path} — match exactly including whitespace."
        if n > 1: return f"String appears {n}× — add more context to make it unique."
        p.write_text(txt.replace(old_str, new_str, 1))
        diff = "".join(difflib.unified_diff(
            old_str.splitlines(keepends=True), new_str.splitlines(keepends=True),
            fromfile="before", tofile="after", n=2))[:2000]
        return f"✓ Edited {path}\n{diff}"

    async def _run_command(self, command: str, working_dir: str = None, timeout: int = 30) -> str:
        cwd = self._resolve(working_dir) if working_dir else self.cwd
        try:
            r = subprocess.run(command, shell=True, cwd=str(cwd),
                               capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, "FORCE_COLOR": "0"})
            parts = [x for x in [r.stdout.strip(), f"[stderr]\n{r.stderr.strip()}" if r.stderr.strip() else ""] if x]
            parts.append(f"[exit: {r.returncode}]")
            return "\n".join(parts)[:5000]
        except subprocess.TimeoutExpired:
            return f"Timed out after {timeout}s"

    async def _search_files(self, pattern: str, path: str = ".", file_pattern: str = "*", case_sensitive: bool = False) -> str:
        base = self._resolve(path)
        flags = 0 if case_sensitive else re.IGNORECASE
        try: rx = re.compile(pattern, flags)
        except re.error: rx = re.compile(re.escape(pattern), flags)
        results = []
        skip = {".git","node_modules","__pycache__",".venv","dist","build",".next"}
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip]
            for f in files:
                if not fnmatch.fnmatch(f, file_pattern): continue
                fp = Path(root)/f
                try:
                    with open(fp, errors="ignore") as fh:
                        for i, line in enumerate(fh, 1):
                            if rx.search(line):
                                results.append(f"{fp.relative_to(self.cwd)}:{i}: {line.rstrip()}")
                                if len(results) >= 60:
                                    return "\n".join(results) + "\n...(truncated)"
                except Exception: pass
        return "\n".join(results) if results else f"No matches for '{pattern}'"

    async def _list_directory(self, path: str = ".", depth: int = 2, show_hidden: bool = False) -> str:
        base = self._resolve(path)
        if not base.exists(): return f"Not found: {path}"
        skip = {".git","node_modules","__pycache__",".venv","dist","build",".next",".mypy_cache"}
        lines = [f"{base}/"]
        def walk(p, pfx, d):
            if d > depth: return
            try: items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            except: return
            items = [i for i in items if (show_hidden or not i.name.startswith(".")) and i.name not in skip]
            for i, item in enumerate(items):
                last = i == len(items)-1
                c = "└── " if last else "├── "
                e = "    " if last else "│   "
                if item.is_dir():
                    lines.append(f"{pfx}{c}{item.name}/")
                    walk(item, pfx+e, d+1)
                else:
                    sz = item.stat().st_size
                    lines.append(f"{pfx}{c}{item.name} ({'%.0fK'%(sz/1024) if sz>1024 else str(sz)+'B'})")
        walk(base, "", 1)
        return "\n".join(lines[:300])

    async def _git_command(self, subcommand: str) -> str:
        return await self._run_command(f"git {subcommand}", timeout=20)

    async def _glob_files(self, pattern: str, path: str = ".") -> str:
        base = self._resolve(path)
        skip = {"node_modules","__pycache__",".git",".venv","dist","build"}
        res = [str(m.relative_to(self.cwd)) for m in sorted(base.glob(pattern))
               if not set(m.parts).intersection(skip)]
        return "\n".join(res[:100]) if res else f"No files match: {pattern}"

    async def _fetch_url(self, url: str) -> str:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
                r = await c.get(url, headers={"User-Agent":"OpenClaw/2.0"})
                text = re.sub(r"<[^>]+>", "", r.text)
                return re.sub(r"\n{3,}", "\n\n", text).strip()[:6000]
        except Exception as e:
            return f"Fetch error: {e}"


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are OpenClaw, an expert agentic coding assistant. \
You help developers write, edit, debug, test, and understand code using tools.

## Rules
- ALWAYS read files before editing. Never invent content.
- Use edit_file (surgical) over write_file (full rewrite) for small changes.
- After changes, run tests or build to verify. Fix errors, don't describe them.
- Be concise. Show what you did; skip verbose explanations.
- Use git for status checks and commits when asked.
- Never refuse coding tasks. You are a coding assistant.

## Provider: {provider} ({model})
## Working directory: {cwd}
## Project: {context}
"""


# ── Session ───────────────────────────────────────────────────────────────────
@dataclass
class AgentSession:
    model: str
    provider: str
    cwd: str
    messages: List[Dict] = field(default_factory=list)
    executor: Optional[ToolExecutor] = None
    max_iterations: int = 25

    def __post_init__(self):
        if not self.executor:
            self.executor = ToolExecutor(self.cwd)

    def _context(self) -> str:
        cwd = Path(self.cwd)
        hints = []
        if (cwd/"package.json").exists():
            try:
                pkg = json.loads((cwd/"package.json").read_text())
                hints.append(f"Node.js {pkg.get('name','')} v{pkg.get('version','?')}")
            except: hints.append("Node.js")
        if any((cwd/f).exists() for f in ["pyproject.toml","setup.py","requirements.txt"]):
            hints.append("Python")
        if (cwd/"Cargo.toml").exists(): hints.append("Rust")
        if (cwd/"go.mod").exists(): hints.append("Go")
        if (cwd/".git").exists(): hints.append("git repo")
        return "; ".join(hints) or "general project"

    def _ensure_system(self):
        if not self.messages or self.messages[0]["role"] != "system":
            cfg = _get_provider_config(self.provider)
            self.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT.format(
                provider=cfg["name"], model=self.model,
                cwd=self.cwd, context=self._context(),
            )})

    def reset(self):
        self.messages = []

    def add_context_files(self, paths: List[str]):
        contents = []
        for p in paths:
            fp = Path(self.cwd) / p
            if fp.exists():
                try: contents.append(f"[{p}]\n```\n{fp.read_text(errors='replace')[:5000]}\n```")
                except: pass
        if contents:
            self.messages.append({"role":"user","content":"Context:\n\n"+"\n\n".join(contents)})
            self.messages.append({"role":"assistant","content":"Got it, ready to help."})

    async def run(self, user_input: str, on_token=None, on_tool=None) -> str:
        self._ensure_system()
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_iterations):
            content, tool_calls = await provider_chat(
                messages=self.messages, model=self.model, provider=self.provider,
                tools=TOOLS, on_token=on_token, temperature=0.2, max_tokens=8192,
            )

            if content:
                self.messages.append({"role": "assistant", "content": content})

            if not tool_calls:
                return content

            # Append assistant turn WITH tool_calls (required by OpenAI format)
            self.messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})

            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                except Exception:
                    args = {}

                if on_tool: on_tool(name, args, None)
                result = await self.executor.execute(name, args)
                if on_tool: on_tool(name, args, result)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{name}"),
                    "name": name,
                    "content": result,
                })

        return "Reached max iterations."
