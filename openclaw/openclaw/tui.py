"""
OpenClaw — Terminal UI
Rich-powered interactive terminal interface with streaming output,
syntax highlighting, tool execution panels, and vim-style shortcuts.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

# ── Try rich, fallback to plain ───────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


BANNER = r"""
   ____  ____  _____ _   _  ____ _        ___   _    __
  / __ \|  _ \| ____| \ | |/ ___| |      / _ \ | |  / /
 | |  | | |_) |  _| |  \| | |   | |     / /_\ \| | / /
 | |__| |  __/| |___| |\  | |___| |___ / /___\ \ |/ /
  \____/|_|   |_____|_| \_|\____|_____/_/     \_\_/_/
  Local-first agentic coding • Powered by Ollama
"""

HELP_TEXT = """
Commands:
  /help          Show this help
  /model <name>  Switch model (e.g. /model qwen2.5-coder:14b)
  /models        List available Ollama models
  /clear         Clear conversation history
  /context <f>   Add file to context (/context src/main.py)
  /undo          Undo last file edit
  /diff          Show recent file changes
  /git           Quick git status
  /cd <path>     Change working directory
  /config        Show current config
  /exit          Exit OpenClaw

Shortcuts:
  Ctrl+C         Cancel current generation
  Up/Down        Navigate command history
  !! at start    Repeat last command
"""

TOOL_ICONS = {
    "read_file":      "📖",
    "write_file":     "✍️ ",
    "edit_file":      "✏️ ",
    "run_command":    "⚡",
    "search_files":   "🔍",
    "list_directory": "📁",
    "git_command":    "🔀",
    "glob_files":     "🗂️ ",
}


class PlainUI:
    """Fallback UI when rich is not available."""

    def __init__(self, cwd: str, model: str):
        self.cwd = cwd
        self.model = model

    def print_banner(self):
        print(BANNER)
        print(f"  Model: {self.model}  |  CWD: {self.cwd}\n")

    def print_help(self):
        print(HELP_TEXT)

    def user_prompt(self) -> str:
        return input(f"\n[{Path(self.cwd).name}] > ").strip()

    def print_tool(self, name: str, args: dict, result: Optional[str]):
        icon = TOOL_ICONS.get(name, "🔧")
        if result is None:
            arg_str = ", ".join(f"{k}={repr(v)[:40]}" for k, v in args.items())
            print(f"\n{icon} {name}({arg_str})")
        else:
            preview = result[:200].replace("\n", "\\n")
            status = "✓" if "error" not in result.lower()[:20] else "✗"
            print(f"  {status} {preview}")

    def print_response(self, text: str):
        print(f"\n{text}")

    def print_error(self, msg: str):
        print(f"\n❌ {msg}")

    def print_info(self, msg: str):
        print(f"\n  {msg}")

    def streaming_start(self):
        pass

    def streaming_token(self, token: str):
        print(token, end="", flush=True)

    def streaming_end(self):
        print()


class RichUI:
    """Full-featured Rich terminal UI."""

    def __init__(self, cwd: str, model: str):
        self.cwd = cwd
        self.model = model
        self.console = Console(highlight=False)
        self._stream_buffer = ""
        self._live: Optional[Live] = None
        self._tool_log: List[str] = []

    def print_banner(self):
        self.console.print(
            Panel(
                f"[bold cyan]{BANNER}[/bold cyan]",
                border_style="cyan",
                padding=(0, 2),
            )
        )
        self._print_status_bar()

    def _print_status_bar(self):
        t = Table.grid(expand=True)
        t.add_column(justify="left")
        t.add_column(justify="center")
        t.add_column(justify="right")
        t.add_row(
            f"[dim]model:[/dim] [bold green]{self.model}[/bold green]",
            f"[dim]cwd:[/dim] [yellow]{self.cwd}[/yellow]",
            "[dim]type /help for commands[/dim]",
        )
        self.console.print(t)
        self.console.print()

    def print_help(self):
        self.console.print(
            Panel(HELP_TEXT, title="[bold]OpenClaw Help[/bold]", border_style="blue")
        )

    def user_prompt(self) -> str:
        cwd_short = Path(self.cwd).name
        try:
            return Prompt.ask(f"\n[bold cyan]{cwd_short}[/bold cyan] [dim]❯[/dim]")
        except (KeyboardInterrupt, EOFError):
            return "/exit"

    def print_tool(self, name: str, args: dict, result: Optional[str]):
        icon = TOOL_ICONS.get(name, "🔧")

        if result is None:
            # Tool starting
            arg_parts = []
            for k, v in args.items():
                if k == "content":
                    arg_parts.append(f"[dim]{k}=[{len(str(v))} chars][/dim]")
                else:
                    arg_parts.append(f"[dim]{k}=[/dim][yellow]{str(v)[:60]}[/yellow]")
            arg_str = "  ".join(arg_parts)
            self.console.print(f"\n  {icon} [bold]{name}[/bold]  {arg_str}")
        else:
            # Tool result
            is_error = any(w in result.lower()[:30] for w in ["error", "not found", "failed"])
            status_icon = "[red]✗[/red]" if is_error else "[green]✓[/green]"
            preview = result[:300]

            # Syntax highlight code output
            if name == "read_file" and len(result) > 50:
                # Extract extension from args
                path_arg = args.get("path", "")
                ext = Path(path_arg).suffix.lstrip(".") or "text"
                lang_map = {"py": "python", "ts": "typescript", "tsx": "tsx",
                            "js": "javascript", "rs": "rust", "go": "go",
                            "sh": "bash", "yml": "yaml", "yaml": "yaml",
                            "json": "json", "md": "markdown", "css": "css",
                            "html": "html", "sql": "sql", "toml": "toml"}
                lang = lang_map.get(ext, "text")
                lines = result.split("\n")
                code_lines = [l.split("│ ", 1)[-1] if "│" in l else l for l in lines[1:21]]
                code = "\n".join(code_lines)
                if code.strip():
                    self.console.print(
                        Panel(
                            Syntax(code, lang, theme="monokai", line_numbers=False),
                            title=f"[dim]{path_arg}[/dim]",
                            border_style="dim",
                            padding=(0, 1),
                        )
                    )
                    return

            elif name in ("write_file", "edit_file") and "✓" in preview:
                self.console.print(f"  {status_icon} [green]{preview.split(chr(10))[0]}[/green]")
                # Show diff if edit
                if "---" in result and "+++" in result:
                    diff_lines = []
                    for line in result.split("\n")[1:]:
                        if line.startswith("+") and not line.startswith("+++"):
                            diff_lines.append(f"[green]{line}[/green]")
                        elif line.startswith("-") and not line.startswith("---"):
                            diff_lines.append(f"[red]{line}[/red]")
                        elif line.startswith("@@"):
                            diff_lines.append(f"[cyan]{line}[/cyan]")
                        else:
                            diff_lines.append(f"[dim]{line}[/dim]")
                    if diff_lines:
                        self.console.print(
                            Panel("\n".join(diff_lines[:20]), border_style="dim", padding=(0, 1))
                        )
                return

            elif name == "run_command":
                exit_line = [l for l in result.split("\n") if "[exit:" in l]
                exit_code = int(exit_line[0].split(":")[1].rstrip("]")) if exit_line else -1
                icon_c = "[green]✓[/green]" if exit_code == 0 else "[red]✗[/red]"
                cmd_output = "\n".join(l for l in result.split("\n") if "[exit:" not in l)
                if cmd_output.strip():
                    self.console.print(
                        Panel(
                            f"[dim]{cmd_output[:800]}[/dim]",
                            title=f"{icon_c} exit:{exit_code}",
                            border_style="green" if exit_code == 0 else "red",
                            padding=(0, 1),
                        )
                    )
                else:
                    self.console.print(f"  {icon_c} [dim]exit:{exit_code}[/dim]")
                return

            self.console.print(f"  {status_icon} [dim]{preview[:200]}[/dim]")

    def streaming_start(self):
        self._stream_buffer = ""
        self.console.print("\n[bold cyan]OpenClaw[/bold cyan] [dim]▸[/dim]", end=" ")

    def streaming_token(self, token: str):
        self._stream_buffer += token
        # Print token directly
        self.console.print(token, end="")

    def streaming_end(self):
        self.console.print()  # newline after stream
        # Optionally re-render as markdown
        if len(self._stream_buffer) > 100 and ("```" in self._stream_buffer or "**" in self._stream_buffer):
            self.console.print()
            self.console.print(Rule(style="dim"))

    def print_response(self, text: str):
        if text and "```" in text or (text and "**" in text):
            self.console.print(Markdown(text))
        else:
            self.console.print(f"\n[bold cyan]OpenClaw[/bold cyan] [dim]▸[/dim] {text}")

    def print_error(self, msg: str):
        self.console.print(f"\n[red bold]✗ Error:[/red bold] {msg}")

    def print_info(self, msg: str):
        self.console.print(f"\n[dim]{msg}[/dim]")

    def print_model_switch(self, old: str, new: str):
        self.console.print(f"\n[dim]Model:[/dim] [red]{old}[/red] → [green]{new}[/green]")

    def print_models_table(self, models: List[str], current: str):
        t = Table(title="Ollama Models", box=box.SIMPLE, border_style="dim")
        t.add_column("Model", style="cyan")
        t.add_column("Status")
        for m in models:
            status = "[bold green]● active[/bold green]" if m == current else "[dim]○[/dim]"
            t.add_row(m, status)
        self.console.print(t)

    def print_thinking(self):
        return Progress(
            SpinnerColumn("dots"),
            TextColumn("[cyan]thinking...[/cyan]"),
            transient=True,
        )

    def print_cwd_change(self, old: str, new: str):
        self.console.print(f"\n[dim]cwd:[/dim] [yellow]{old}[/yellow] → [yellow]{new}[/yellow]")

    def update_model(self, model: str):
        self.model = model

    def update_cwd(self, cwd: str):
        self.cwd = cwd


def make_ui(cwd: str, model: str):
    return RichUI(cwd, model) if HAS_RICH else PlainUI(cwd, model)
