# 🦅 OpenClaw

**Local-first agentic coding CLI powered by Ollama.**  
Claude Code for your local LLMs — no API keys, no cloud, full control.

```
   ____  ____  _____ _   _  ____ _        ___   _    __
  / __ \|  _ \| ____| \ | |/ ___| |      / _ \ | |  / /
 | |  | | |_) |  _| |  \| | |   | |     / /_\ \| | / /
 | |__| |  __/| |___| |\  | |___| |___ / /___\ \ |/ /
  \____/|_|   |_____|_| \_|\____|_____/_/     \_\_/_/
```

---

## What is it?

OpenClaw is a terminal-based AI coding agent that runs entirely on your machine using [Ollama](https://ollama.ai). It can:

- 📖 **Read and understand** your entire codebase
- ✏️ **Edit files** with surgical precision (no hallucinated content)
- ⚡ **Run commands** — tests, builds, linters, git operations
- 🔍 **Search code** with regex across your project
- 🔀 **Manage git** — status, diff, commit, branch
- 🧠 **Remember context** across a session (full conversation history)

---

## Quick Start

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull a coding model (pick one)
ollama pull qwen2.5-coder:7b       # Recommended — fast, accurate
ollama pull qwen2.5-coder:14b      # Better quality, needs 10GB VRAM
ollama pull deepseek-r1:8b         # Reasoning model — great for complex tasks

# 3. Install OpenClaw
pip install httpx rich             # Dependencies only (no pip package yet)
cd agentflow-v2/openclaw
python -m openclaw                 # or: python -m openclaw.cli

# 4. Start coding
openclaw                           # Interactive mode in current directory
openclaw -d /path/to/project       # Specific directory
openclaw "write unit tests for utils.py"  # One-shot mode
```

---

## Model Recommendations

| Model | Size | Best For | Speed |
|-------|------|----------|-------|
| `qwen2.5-coder:7b` | 4.7GB | General coding, fast iteration | ⚡⚡⚡ |
| `qwen2.5-coder:14b` | 9GB | Better quality, complex tasks | ⚡⚡ |
| `qwen2.5-coder:32b` | 20GB | Best quality (needs GPU) | ⚡ |
| `deepseek-r1:8b` | 4.9GB | Step-by-step reasoning | ⚡⚡ |
| `deepseek-r1:14b` | 9GB | Deep analysis, architecture | ⚡⚡ |
| `deepseek-coder-v2:16b` | 9GB | Strong code generation | ⚡⚡ |
| `llama3.1:8b` | 4.7GB | General tasks + coding | ⚡⚡⚡ |
| `codestral:22b` | 14GB | Mistral's coding model | ⚡ |

> **No GPU?** `qwen2.5-coder:7b` runs fine on CPU (slower but works).

---

## Usage

### Interactive Mode (REPL)

```bash
openclaw                    # Use current directory
openclaw -d ~/myproject     # Specific directory
openclaw -m llama3.1:8b     # Override model
```

### One-shot Mode

```bash
openclaw "add error handling to auth.py"
openclaw "write a Dockerfile for this Node.js app"
openclaw "find all TODO comments and create GitHub issues format"
openclaw -d ~/myproject "what does this codebase do?"
```

### Slash Commands

```
/help              Show all commands
/models            List installed Ollama models
/model <name>      Switch model mid-session
/context <file>    Add a file to context (pre-read)
/clear             Reset conversation history
/undo              Undo last file edit
/diff              Show git diff --stat
/git               Quick git status
/cd <path>         Change working directory
/config            Show current settings
/exit              Exit
```

---

## Configuration

### Global config (`~/.config/openclaw/config.json`)

```json
{
  "default_model": "qwen2.5-coder:7b",
  "ollama_url": "http://localhost:11434",
  "context_window": 32768,
  "temperature": 0.2,
  "max_iterations": 20,
  "auto_approve_tools": true,
  "stream_output": true,
  "theme": "monokai"
}
```

### Per-project config (`.openclaw.json` in project root)

```json
{
  "default_model": "qwen2.5-coder:14b",
  "max_iterations": 30,
  "ignore_patterns": ["*.pyc", "node_modules", "dist/", "*.lock"]
}
```

### Environment variables

```bash
export OLLAMA_URL=http://192.168.1.100:11434   # Remote Ollama
export OPENCLAW_MODEL=qwen2.5-coder:14b         # Override model
export OPENCLAW_CTX=65536                        # Context window
```

---

## Tools OpenClaw Can Use

| Tool | Description |
|------|-------------|
| `read_file` | Read file with optional line range |
| `write_file` | Create/overwrite a file |
| `edit_file` | Replace exact string (surgical edits) |
| `run_command` | Shell commands, tests, builds |
| `search_files` | Regex search across files |
| `list_directory` | Tree view of project structure |
| `git_command` | All git operations |
| `glob_files` | Find files by glob pattern |

---

## Example Sessions

### Debug a failing test
```
> the test_auth tests are failing, figure out why and fix it

📁 list_directory(path=.)
⚡ run_command(command=python -m pytest tests/test_auth.py -v)
📖 read_file(path=tests/test_auth.py)
📖 read_file(path=src/auth.py)
✏️  edit_file(path=src/auth.py, ...)
⚡ run_command(command=python -m pytest tests/test_auth.py -v)
✓ All 8 tests pass.
```

### Add a feature
```
> add rate limiting to the /api/login endpoint, 5 attempts per minute per IP

📖 read_file(path=src/routes/auth.py)
🔍 search_files(pattern=from flask|import flask)
✍️  write_file(path=src/middleware/rate_limit.py)
✏️  edit_file(path=src/routes/auth.py, ...)
⚡ run_command(command=python -m pytest)
🔀 git_command(subcommand=add -A)
🔀 git_command(subcommand=commit -m "feat: rate limiting on login endpoint")
```

### Understand a codebase
```
> explain the architecture of this project and identify potential issues

📁 list_directory(path=., depth=3)
📖 read_file(path=README.md)
🔍 search_files(pattern=class |def , file_pattern=*.py)
...
```

---

## Integration with AgentFlow v2

OpenClaw is embedded in AgentFlow v2 as the **Coding Agent** backend.
It's also available as a standalone CLI tool.

```python
from openclaw import AgentSession

session = AgentSession(model="qwen2.5-coder:7b", cwd="/my/project")
result = await session.run("add type hints to all functions in utils.py")
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally (or on a remote server)
- `httpx` — async HTTP
- `rich` — beautiful terminal UI (optional, falls back gracefully)

---

## License

MIT — do whatever you want with it.
