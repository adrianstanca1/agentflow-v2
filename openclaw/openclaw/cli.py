"""
OpenClaw — CLI Entry Point
Interactive REPL: model switching, file context, history, slash commands.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

from .config import OpenClawConfig, detect_best_model, get_recommended_models
from .core import AgentSession, list_models, ollama_available, pull_model
from .tui import make_ui


# ── Command history ────────────────────────────────────────────────────────────
try:
    import readline

    def setup_readline(history_file: str, max_history: int):
        readline.set_history_length(max_history)
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass

    def save_history(history_file: str):
        try:
            readline.write_history_file(history_file)
        except Exception:
            pass
except ImportError:
    def setup_readline(history_file, max_history): pass
    def save_history(history_file): pass


# ── Main REPL ─────────────────────────────────────────────────────────────────
async def run_repl(
    cwd: str,
    model: str,
    cfg: OpenClawConfig,
    initial_task: Optional[str] = None,
):
    ui = make_ui(cwd, model)

    if cfg.show_banner:
        ui.print_banner()

    session = AgentSession(model=model, cwd=cwd, max_iterations=cfg.max_iterations)
    setup_readline(cfg.history_file, cfg.max_history)

    last_cmd = ""
    streaming_active = False

    def on_token(token: str):
        nonlocal streaming_active
        if not streaming_active:
            ui.streaming_start()
            streaming_active = True
        ui.streaming_token(token)

    def on_tool(name: str, args: dict, result):
        ui.print_tool(name, args, result)

    async def handle_command(cmd: str) -> bool:
        """Handle slash commands. Returns True to continue, False to exit."""
        nonlocal model, cwd, streaming_active

        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command in ("/exit", "/quit", "/q"):
            ui.print_info("Bye! 👋")
            return False

        elif command == "/help":
            ui.print_help()

        elif command == "/clear":
            session.reset()
            ui.print_info("Conversation history cleared.")

        elif command == "/models":
            available = await list_models(cfg.ollama_url)
            if not available:
                ui.print_error("No models found. Is Ollama running?")
            else:
                if hasattr(ui, "print_models_table"):
                    ui.print_models_table(available, model)
                else:
                    for m in available:
                        mark = " ◀ active" if m == model else ""
                        ui.print_info(f"  {m}{mark}")

        elif command == "/model":
            if not arg:
                ui.print_info(f"Current model: {model}")
                return True
            available = await list_models(cfg.ollama_url)
            if arg not in available:
                # Check if it's in recommended and offer to pull
                recommended = get_recommended_models()
                if arg in recommended:
                    ui.print_info(f"Model '{arg}' not installed. Pulling...")
                    async for progress in pull_model(arg, cfg.ollama_url):
                        status = progress.get("status", "")
                        pct = progress.get("percent", 0)
                        if pct:
                            ui.print_info(f"  {status} {pct:.0f}%")
                        elif status:
                            ui.print_info(f"  {status}")
                else:
                    ui.print_error(f"Model '{arg}' not found. Run /models to see available models.")
                    return True
            old_model = model
            model = arg
            session.model = model
            if hasattr(ui, "update_model"):
                ui.update_model(model)
            if hasattr(ui, "print_model_switch"):
                ui.print_model_switch(old_model, model)
            else:
                ui.print_info(f"Switched to {model}")

        elif command == "/context":
            if not arg:
                ui.print_info("Usage: /context <filepath>")
                return True
            paths = arg.split()
            session.add_file_context(paths)
            ui.print_info(f"Added {len(paths)} file(s) to context.")

        elif command == "/diff":
            result = await session.executor._run_command(
                "git diff --stat HEAD 2>/dev/null || echo 'Not a git repo'"
            )
            ui.print_info(result)

        elif command == "/git":
            result = await session.executor._git_command("status -s")
            ui.print_info(result or "Nothing to show.")

        elif command == "/cd":
            if not arg:
                ui.print_info(f"Current: {cwd}")
                return True
            new_cwd = str(Path(cwd) / arg) if not Path(arg).is_absolute() else arg
            if Path(new_cwd).is_dir():
                old_cwd = cwd
                cwd = str(Path(new_cwd).resolve())
                session.cwd = cwd
                session.executor.cwd = Path(cwd)
                if hasattr(ui, "print_cwd_change"):
                    ui.print_cwd_change(old_cwd, cwd)
                    ui.update_cwd(cwd)
                else:
                    ui.print_info(f"Changed to {cwd}")
            else:
                ui.print_error(f"Directory not found: {new_cwd}")

        elif command == "/config":
            ui.print_info(cfg.to_display())

        elif command == "/undo":
            # Undo last file edit
            history = session.executor._file_history
            if not history:
                ui.print_info("Nothing to undo.")
                return True
            last_path = list(history.keys())[-1]
            snapshots = history[last_path]
            if snapshots:
                restored = snapshots.pop()
                Path(last_path).write_text(restored)
                ui.print_info(f"Restored {last_path}")
            else:
                ui.print_info("No more snapshots for that file.")

        elif command in ("/pull",):
            target = arg or cfg.default_model
            ui.print_info(f"Pulling {target}...")
            async for p in pull_model(target, cfg.ollama_url):
                status = p.get("status", "")
                pct = p.get("percent", 0)
                if pct:
                    ui.print_info(f"  {pct:.0f}% {status}")
                elif status and status != last_cmd:
                    ui.print_info(f"  {status}")
                    last_cmd = status

        else:
            ui.print_error(f"Unknown command: {command}. Type /help for commands.")

        return True

    # ── Initial task from CLI args ─────────────────────────────────────────────
    if initial_task:
        streaming_active = False
        try:
            await session.run(initial_task, on_token=on_token, on_tool=on_tool)
            if streaming_active:
                ui.streaming_end()
        except KeyboardInterrupt:
            ui.print_info("\nCancelled.")
        return

    # ── Interactive REPL ───────────────────────────────────────────────────────
    while True:
        try:
            streaming_active = False
            user_input = ui.user_prompt()
        except (KeyboardInterrupt, EOFError):
            ui.print_info("\nBye! 👋")
            break

        if not user_input:
            continue

        # Repeat last command
        if user_input == "!!":
            user_input = last_cmd
            if not user_input:
                continue
            ui.print_info(f"↑ {user_input}")

        last_cmd = user_input
        save_history(cfg.history_file)

        # Slash commands
        if user_input.startswith("/"):
            should_continue = await handle_command(user_input.lstrip("/").strip() if user_input == "/" else user_input)
            if not should_continue:
                break
            continue

        # Agent run
        try:
            await session.run(user_input, on_token=on_token, on_tool=on_tool)
            if streaming_active:
                ui.streaming_end()
        except KeyboardInterrupt:
            if streaming_active:
                ui.streaming_end()
            ui.print_info("\nCancelled. Press Ctrl+C again to exit.")
        except Exception as e:
            if streaming_active:
                ui.streaming_end()
            ui.print_error(str(e))

    save_history(cfg.history_file)


# ── Entrypoint ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="openclaw",
        description="OpenClaw — Local-first agentic coding CLI powered by Ollama",
    )
    parser.add_argument("task", nargs="*", help="Task to run non-interactively")
    parser.add_argument("-m", "--model", help="Model to use (e.g. qwen2.5-coder:7b)")
    parser.add_argument("-d", "--dir", default=".", help="Working directory")
    parser.add_argument("--ollama-url", help="Ollama URL (default: http://localhost:11434)")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--no-banner", action="store_true", help="Skip banner")
    parser.add_argument("--version", action="store_true", help="Show version")
    args = parser.parse_args()

    if args.version:
        print("OpenClaw 1.0.0")
        return

    cfg = OpenClawConfig.load(args.dir)

    if args.ollama_url:
        cfg.ollama_url = args.ollama_url
    if args.no_banner:
        cfg.show_banner = False

    async def _main():
        # Check Ollama
        if not await ollama_available(cfg.ollama_url):
            print(f"❌ Ollama not reachable at {cfg.ollama_url}")
            print("   Start it with: ollama serve")
            print("   Or set OLLAMA_URL to the correct address")
            sys.exit(1)

        # List models
        available = await list_models(cfg.ollama_url)

        if args.list_models:
            print("Available models:")
            for m in available:
                print(f"  {m}")
            return

        # Pick model
        if args.model:
            model = args.model
        elif cfg.default_model in available:
            model = cfg.default_model
        else:
            model = detect_best_model(available)
            if not model:
                print("❌ No models available. Pull one first:")
                print("   ollama pull qwen2.5-coder:7b")
                sys.exit(1)
            print(f"  Using detected model: {model}")

        cwd = str(Path(args.dir).resolve())
        initial_task = " ".join(args.task) if args.task else None

        await run_repl(cwd=cwd, model=model, cfg=cfg, initial_task=initial_task)

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
