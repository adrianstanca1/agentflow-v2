"""
OpenClaw — Local-First Agentic Coding CLI
Claude Code-style terminal agent powered by Ollama.
Zero API keys required. Runs entirely on your machine.
"""
from __future__ import annotations

import asyncio
import difflib
import fnmatch
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ── Ollama client ─────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OPENCLAW_MODEL", "qwen2.5-coder:7b")
CONTEXT_WINDOW = int(os.getenv("OPENCLAW_CTX", "32768"))


async def ollama_chat(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    tools: Optional[List[Dict]] = None,
    stream: bool = True,
) -> Tuple[str, Optional[Dict]]:
    """
    Send messages to Ollama. Returns (text_content, tool_call_or_None).
    Supports native Ollama tool use for models that support it (qwen2.5, llama3.1+).
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"num_ctx": CONTEXT_WINDOW, "temperature": 0.2},
    }
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=120.0) as client:
        if not stream:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message", {})
            return msg.get("content", ""), msg.get("tool_calls")

        # Streaming
        full_content = ""
        tool_calls = None
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    msg = chunk.get("message", {})
                    delta = msg.get("content", "")
                    full_content += delta
                    if delta:
                        yield_token(delta)
                    if msg.get("tool_calls"):
                        tool_calls = msg["tool_calls"]
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    pass
        return full_content, tool_calls


async def ollama_available(url: str = OLLAMA_URL) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def list_models(url: str = OLLAMA_URL) -> List[str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{url}/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


async def pull_model(model: str, url: str = OLLAMA_URL):
    """Pull a model with progress callback."""
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", f"{url}/api/pull", json={"name": model, "stream": True}) as resp:
            async for line in resp.aiter_lines():
                if line.strip():
                    try:
                        d = json.loads(line)
                        yield d
                    except json.JSONDecodeError:
                        pass


# Token streaming callback (overridden by UI)
_token_callback = None

def yield_token(token: str):
    if _token_callback:
        _token_callback(token)


def set_token_callback(fn):
    global _token_callback
    _token_callback = fn


# ── Tool definitions ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Use this to understand existing code before editing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to working directory"},
                    "start_line": {"type": "integer", "description": "Start line (1-indexed, optional)"},
                    "end_line": {"type": "integer", "description": "End line (1-indexed, optional)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or create a file with the given content. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing an exact string with new content. More precise than write_file for small changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_str": {"type": "string", "description": "Exact string to find and replace (must be unique in file)"},
                    "new_str": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command. Use for tests, builds, installs, git operations. Avoid destructive commands without confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "working_dir": {"type": "string", "description": "Working directory (optional, defaults to cwd)"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for text patterns in files. Returns file paths and matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search (default: cwd)"},
                    "file_pattern": {"type": "string", "description": "Glob pattern for files (e.g. '*.py', '*.ts')"},
                    "case_sensitive": {"type": "boolean", "description": "Case sensitive search (default false)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories. Shows the project structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: cwd)"},
                    "depth": {"type": "integer", "description": "Max depth to traverse (default 2)"},
                    "show_hidden": {"type": "boolean", "description": "Show hidden files (default false)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_command",
            "description": "Run git operations: status, diff, log, add, commit, branch, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subcommand": {"type": "string", "description": "Git subcommand and args (e.g. 'status', 'diff HEAD', 'log --oneline -10', 'add -A', 'commit -m \"fix: bug\"')"},
                },
                "required": ["subcommand"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. 'src/**/*.py', '*.json')"},
                    "path": {"type": "string", "description": "Base directory (default: cwd)"},
                },
                "required": ["pattern"],
            },
        },
    },
]


# ── Tool executor ──────────────────────────────────────────────────────────────
class ToolExecutor:
    def __init__(self, cwd: str, confirm_fn=None):
        self.cwd = Path(cwd).resolve()
        self.confirm_fn = confirm_fn  # async fn(action) -> bool
        self._file_history: Dict[str, List[str]] = {}  # path -> [snapshots]

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.cwd / p
        return p.resolve()

    def _snapshot(self, path: Path):
        """Save file snapshot for undo."""
        try:
            content = path.read_text(errors="replace")
            key = str(path)
            if key not in self._file_history:
                self._file_history[key] = []
            self._file_history[key].append(content)
        except Exception:
            pass

    async def execute(self, tool_name: str, args: Dict) -> str:
        handlers = {
            "read_file":      self._read_file,
            "write_file":     self._write_file,
            "edit_file":      self._edit_file,
            "run_command":    self._run_command,
            "search_files":   self._search_files,
            "list_directory": self._list_directory,
            "git_command":    self._git_command,
            "glob_files":     self._glob_files,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        try:
            return await handler(**args)
        except Exception as e:
            return f"Tool error ({tool_name}): {e}"

    async def _read_file(self, path: str, start_line: int = None, end_line: int = None) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"File not found: {path}"
        try:
            lines = p.read_text(errors="replace").splitlines(keepends=True)
            if start_line or end_line:
                s = (start_line or 1) - 1
                e = end_line or len(lines)
                lines = lines[s:e]
                prefix = f"[Lines {start_line or 1}-{end_line or len(lines)}: {path}]\n"
            else:
                prefix = f"[{path}]\n"
            # Add line numbers
            start = (start_line or 1)
            numbered = "".join(f"{start+i:4d}│ {l}" for i, l in enumerate(lines))
            return prefix + numbered
        except Exception as e:
            return f"Read error: {e}"

    async def _write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if p.exists():
            self._snapshot(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        lines = content.count("\n") + 1
        return f"✓ Written {path} ({lines} lines, {len(content)} bytes)"

    async def _edit_file(self, path: str, old_str: str, new_str: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"File not found: {path}"
        self._snapshot(p)
        content = p.read_text(errors="replace")
        count = content.count(old_str)
        if count == 0:
            # Show a helpful diff of what was expected vs what exists
            return f"String not found in {path}. Make sure to match exactly including whitespace/indentation."
        if count > 1:
            return f"String found {count} times in {path} — must be unique. Add more context around the string."
        new_content = content.replace(old_str, new_str, 1)
        p.write_text(new_content)
        # Show diff
        diff = list(difflib.unified_diff(
            old_str.splitlines(keepends=True),
            new_str.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
            n=2,
        ))
        diff_preview = "".join(diff[:30])
        return f"✓ Edited {path}\n{diff_preview}"

    async def _run_command(self, command: str, working_dir: str = None, timeout: int = 30) -> str:
        cwd = self._resolve(working_dir) if working_dir else self.cwd
        try:
            result = subprocess.run(
                command, shell=True, cwd=str(cwd),
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "FORCE_COLOR": "0"},
            )
            out = result.stdout.strip()
            err = result.stderr.strip()
            code = result.returncode
            parts = []
            if out: parts.append(out)
            if err: parts.append(f"[stderr]\n{err}")
            parts.append(f"[exit: {code}]")
            return "\n".join(parts)[:4000]
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s: {command}"
        except Exception as e:
            return f"Command error: {e}"

    async def _search_files(self, pattern: str, path: str = ".", file_pattern: str = "*", case_sensitive: bool = False) -> str:
        base = self._resolve(path)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            regex = re.compile(re.escape(pattern), flags)

        results = []
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "dist", "build", ".next"}

        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fname in files:
                if not fnmatch.fnmatch(fname, file_pattern):
                    continue
                fpath = Path(root) / fname
                try:
                    with open(fpath, "r", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                rel = fpath.relative_to(self.cwd)
                                results.append(f"{rel}:{i}: {line.rstrip()}")
                                if len(results) >= 50:
                                    results.append("... (truncated at 50 results)")
                                    return "\n".join(results)
                except Exception:
                    pass

        if not results:
            return f"No matches for '{pattern}'"
        return "\n".join(results)

    async def _list_directory(self, path: str = ".", depth: int = 2, show_hidden: bool = False) -> str:
        base = self._resolve(path)
        if not base.exists():
            return f"Directory not found: {path}"
        lines = [f"{base}/"]
        skip = {".git", "node_modules", "__pycache__", ".venv", "dist", "build", ".next", ".mypy_cache"}

        def _walk(p: Path, prefix: str, cur_depth: int):
            if cur_depth > depth:
                return
            try:
                items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            except PermissionError:
                return
            items = [i for i in items if show_hidden or not i.name.startswith(".")]
            for i, item in enumerate(items):
                if item.name in skip:
                    continue
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                if item.is_dir():
                    lines.append(f"{prefix}{connector}{item.name}/")
                    ext = "    " if is_last else "│   "
                    _walk(item, prefix + ext, cur_depth + 1)
                else:
                    size = item.stat().st_size
                    size_s = f"{size/1024:.0f}K" if size > 1024 else f"{size}B"
                    lines.append(f"{prefix}{connector}{item.name} ({size_s})")

        _walk(base, "", 1)
        return "\n".join(lines[:200])

    async def _git_command(self, subcommand: str) -> str:
        safe_prefixes = ["status", "diff", "log", "branch", "show", "blame", "ls-files",
                          "add", "commit", "checkout", "stash", "tag", "remote -v",
                          "fetch", "pull", "push", "merge", "rebase", "reset --soft",
                          "reset --mixed", "cherry-pick"]
        dangerous = ["reset --hard", "clean -f", "rm", "push --force"]
        for d in dangerous:
            if subcommand.strip().startswith(d):
                if self.confirm_fn:
                    ok = await self.confirm_fn(f"git {subcommand}")
                    if not ok:
                        return "Cancelled by user."
                break
        return await self._run_command(f"git {subcommand}", timeout=20)

    async def _glob_files(self, pattern: str, path: str = ".") -> str:
        base = self._resolve(path)
        matches = sorted(base.glob(pattern))
        skip = {"node_modules", "__pycache__", ".git", ".venv", "dist", "build"}
        results = []
        for m in matches:
            parts_set = set(m.parts)
            if not parts_set.intersection(skip):
                results.append(str(m.relative_to(self.cwd)))
        if not results:
            return f"No files match: {pattern}"
        return "\n".join(results[:100])


# ── Agent ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are OpenClaw, an expert agentic coding assistant running entirely locally via Ollama.
You help developers write, edit, debug, test, and refactor code.

## Rules
- Always READ files before editing them. Never guess file contents.
- For edits, prefer edit_file (precise) over write_file (full rewrite).
- Run tests after making changes to verify correctness.
- Use git_command to check status/diff and commit when asked.
- Break complex tasks into steps, executing tools one at a time.
- Be concise. Show what you did, not verbose explanations.
- When you encounter an error, fix it — don't just describe it.
- Never make up file contents. Always read first.

## Working directory
{cwd}

## Project context
{context}
"""


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None


@dataclass
class AgentSession:
    model: str
    cwd: str
    messages: List[Dict] = field(default_factory=list)
    executor: Optional[ToolExecutor] = None
    max_iterations: int = 20
    _token_cb: Any = None

    def __post_init__(self):
        if self.executor is None:
            self.executor = ToolExecutor(self.cwd)

    def _build_context(self) -> str:
        """Scan cwd for project type indicators."""
        cwd = Path(self.cwd)
        hints = []
        if (cwd / "package.json").exists():
            try:
                pkg = json.loads((cwd / "package.json").read_text())
                hints.append(f"Node.js project: {pkg.get('name','')} {pkg.get('version','')}")
                deps = list(pkg.get("dependencies", {}).keys())[:5]
                if deps: hints.append(f"Dependencies: {', '.join(deps)}")
            except Exception:
                hints.append("Node.js project")
        if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists():
            hints.append("Python project")
        if (cwd / "Cargo.toml").exists():
            hints.append("Rust project")
        if (cwd / "go.mod").exists():
            hints.append("Go project")
        if (cwd / "Dockerfile").exists():
            hints.append("Has Dockerfile")
        if (cwd / ".git").exists():
            hints.append("Git repository")
        return "; ".join(hints) if hints else "General project"

    def reset(self):
        self.messages = []

    def _init_system(self):
        if not self.messages or self.messages[0]["role"] != "system":
            context = self._build_context()
            system = SYSTEM_PROMPT.format(cwd=self.cwd, context=context)
            self.messages.insert(0, {"role": "system", "content": system})

    async def run(self, user_input: str, on_token=None, on_tool=None) -> str:
        """
        Run one turn of the agentic loop.
        on_token(str) - called for each streamed token
        on_tool(name, args, result) - called for each tool execution
        """
        set_token_callback(on_token)
        self._init_system()
        self.messages.append({"role": "user", "content": user_input})

        for iteration in range(self.max_iterations):
            content, tool_calls = await ollama_chat(
                self.messages, model=self.model, tools=TOOLS, stream=True
            )

            if content:
                self.messages.append({"role": "assistant", "content": content})

            # No tool calls → done
            if not tool_calls:
                set_token_callback(None)
                return content

            # Execute tool calls
            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

                if on_tool:
                    on_tool(tool_name, args, None)

                result = await self.executor.execute(tool_name, args)

                if on_tool:
                    on_tool(tool_name, args, result)

                # Add tool result to messages
                self.messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tc],
                })
                self.messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc.get("id", tool_name),
                    "name": tool_name,
                })

        set_token_callback(None)
        return "Reached max iterations."

    def add_file_context(self, paths: List[str]):
        """Pre-load files into context."""
        contents = []
        for p in paths:
            fp = Path(self.cwd) / p
            if fp.exists():
                try:
                    text = fp.read_text(errors="replace")
                    contents.append(f"[{p}]\n```\n{text[:4000]}\n```")
                except Exception:
                    pass
        if contents:
            self.messages.append({
                "role": "user",
                "content": "Here are the relevant files:\n\n" + "\n\n".join(contents),
            })
            self.messages.append({
                "role": "assistant",
                "content": "I've read the files. Ready to help.",
            })
