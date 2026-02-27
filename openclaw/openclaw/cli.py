"""
OpenClaw CLI — agentic coding agent for any provider.
Usage:
  openclaw                              # auto-detect provider, interactive
  openclaw -p groq -m llama-3.3-70b    # specify provider + model
  openclaw "fix the bug in auth.py"    # one-shot task
  openclaw --list-providers            # show configured providers
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make sure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openclaw.core import (
    PROVIDERS, AgentSession, check_provider, detect_model,
    detect_provider, list_provider_models, _get_provider_config,
)
from openclaw.tui import RichUI, PlainUI


def _get_ui(plain: bool = False):
    if plain:
        return PlainUI()
    try:
        return RichUI()
    except Exception:
        return PlainUI()


async def cmd_list_providers(ui):
    ui.print_header()
    ui.info("Configured providers:\n")
    for pid, cfg in PROVIDERS.items():
        key_env = cfg.get("api_key_env", "")
        key = os.getenv(key_env, "") if key_env else "n/a"
        if pid == "ollama":
            url = os.getenv(cfg.get("base_url_env","OLLAMA_URL"), cfg.get("base_url_default",""))
            status = f"URL: {url}"
            configured = True  # always available if Ollama runs
        else:
            configured = bool(key)
            status = f"{'✓ key set' if configured else '✗ no key'}"
        icon = cfg["icon"]
        name = cfg["name"]
        default = cfg.get("default_model","")
        line = f"  {icon} {name:15} {status:25} default: {default}"
        if configured:
            ui.success(line)
        else:
            ui.dim(line)
    ui.info("\nSet keys in .env or via Settings in the web UI.")


async def cmd_check_provider(provider: str, ui):
    ui.info(f"Checking {PROVIDERS[provider]['name']}...")
    result = await check_provider(provider)
    if result["ok"]:
        ui.success(f"✓ Connected — {result.get('model_count',0)} models — {result.get('latency_ms',0)}ms")
    else:
        ui.error(f"✗ Failed: {result['error']}")


async def interactive_loop(session: AgentSession, ui):
    """REPL loop — reads tasks, runs agent, prints output."""
    import readline

    history_file = Path.home() / ".openclaw_history"
    try:
        readline.read_history_file(str(history_file))
    except FileNotFoundError:
        pass

    cfg = _get_provider_config(session.provider)
    ui.print_header(provider=cfg["name"], model=session.model, cwd=session.cwd)

    def _on_token(tok: str):
        ui.stream_token(tok)

    def _on_tool(name: str, args: dict, result):
        if result is None:
            ui.tool_start(name, args)
        else:
            ui.tool_result(name, result)

    last_task = ""

    while True:
        try:
            raw = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            ui.info("\nBye!")
            break

        if not raw:
            continue

        # Slash commands
        if raw.startswith("/"):
            parts = raw[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("exit", "quit", "q"):
                ui.info("Bye!")
                break
            elif cmd == "clear":
                session.reset()
                ui.success("Conversation cleared.")
            elif cmd == "help":
                ui.info("""Commands:
  /clear          — clear conversation history
  /provider <id>  — switch provider (groq, openai, ollama, ...)
  /model <name>   — switch model
  /models         — list models for current provider
  /context <file> — pre-load a file into context
  /undo <file>    — undo last edit to a file
  /cd <dir>       — change working directory
  /check          — test current provider connection
  /providers      — list all configured providers
  /exit           — quit""")
            elif cmd == "providers":
                await cmd_list_providers(ui)
            elif cmd == "check":
                await cmd_check_provider(session.provider, ui)
            elif cmd == "provider":
                if arg and arg in PROVIDERS:
                    session.provider = arg
                    session.model = detect_model(arg)
                    session.reset()
                    cfg = _get_provider_config(arg)
                    ui.success(f"Switched to {cfg['name']} / {session.model}")
                else:
                    ui.error(f"Unknown provider. Options: {', '.join(PROVIDERS.keys())}")
            elif cmd == "model":
                if arg:
                    session.model = arg
                    ui.success(f"Model set to {arg}")
                else:
                    ui.error("Usage: /model <name>")
            elif cmd == "models":
                ui.info(f"Fetching models for {session.provider}...")
                models = await list_provider_models(session.provider)
                for m in models[:30]:
                    ui.dim(f"  {m}")
                if len(models) > 30:
                    ui.dim(f"  ... and {len(models)-30} more")
            elif cmd == "context":
                if arg:
                    session.add_context_files([arg])
                    ui.success(f"Loaded {arg} into context.")
                else:
                    ui.error("Usage: /context <file>")
            elif cmd == "undo":
                if arg:
                    ok = session.executor.undo_file(arg)
                    ui.success(f"Undone: {arg}") if ok else ui.error(f"No undo history for {arg}")
                else:
                    ui.error("Usage: /undo <file>")
            elif cmd == "cd":
                target = Path(arg).expanduser().resolve() if arg else Path.home()
                if target.is_dir():
                    session.cwd = str(target)
                    session.executor.cwd = target
                    ui.success(f"cwd: {target}")
                else:
                    ui.error(f"Not a directory: {arg}")
            else:
                ui.error(f"Unknown command: /{cmd}  — type /help for commands")
            continue

        # Repeat last on !!
        if raw == "!!":
            if not last_task:
                ui.error("No previous task.")
                continue
            raw = last_task

        last_task = raw

        try:
            ui.user_msg(raw)
            await session.run(raw, on_token=_on_token, on_tool=_on_tool)
            ui.stream_end()
        except KeyboardInterrupt:
            ui.warning("\nInterrupted.")
        except Exception as e:
            ui.error(str(e))

    try:
        readline.write_history_file(str(history_file))
    except Exception:
        pass


async def main():
    parser = argparse.ArgumentParser(
        prog="openclaw",
        description="OpenClaw — agentic coding agent for any LLM provider",
    )
    parser.add_argument("task", nargs="?", help="One-shot task (omit for interactive mode)")
    parser.add_argument("-p", "--provider", default=None,
                        help=f"Provider: {', '.join(PROVIDERS.keys())} (auto-detected from .env)")
    parser.add_argument("-m", "--model", default=None, help="Model name")
    parser.add_argument("-d", "--dir", default=None, help="Working directory")
    parser.add_argument("--plain", action="store_true", help="Plain text output (no rich)")
    parser.add_argument("--no-banner", action="store_true", help="Skip banner")
    parser.add_argument("--list-providers", action="store_true", help="Show configured providers")
    parser.add_argument("--check", metavar="PROVIDER", help="Test a provider connection")

    args = parser.parse_args()
    ui = _get_ui(args.plain)
    cwd = str(Path(args.dir).expanduser().resolve()) if args.dir else os.getcwd()

    if args.list_providers:
        await cmd_list_providers(ui)
        return

    if args.check:
        await cmd_check_provider(args.check, ui)
        return

    # Resolve provider + model
    provider = args.provider or detect_provider()
    model = args.model or detect_model(provider)

    if provider not in PROVIDERS:
        ui.error(f"Unknown provider: {provider}. Use: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    session = AgentSession(model=model, provider=provider, cwd=cwd)

    if args.task:
        # One-shot mode
        cfg = _get_provider_config(provider)
        if not args.no_banner:
            ui.info(f"OpenClaw [{cfg['icon']} {cfg['name']} / {model}] — {cwd}")

        def _on_token(tok): ui.stream_token(tok)
        def _on_tool(name, a, r):
            if r is None: ui.tool_start(name, a)
            else: ui.tool_result(name, r)

        try:
            await session.run(args.task, on_token=_on_token, on_tool=_on_tool)
            ui.stream_end()
        except Exception as e:
            ui.error(str(e))
            sys.exit(1)
    else:
        await interactive_loop(session, ui)


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
