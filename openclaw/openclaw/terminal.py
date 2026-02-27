"""
OpenClaw Web Terminal — PTY WebSocket bridge
Runs openclaw (or any command) in a real PTY, streams I/O over WebSocket.
Powers the browser-based terminal on the OpenClaw page.
"""
from __future__ import annotations

import asyncio
import json
import os
import pty
import select
import struct
import termios
import fcntl
import signal
from typing import Dict, Optional


class PTYSession:
    """A single PTY session connected to a shell or command."""

    def __init__(self, session_id: str, cols: int = 220, rows: int = 50):
        self.session_id = session_id
        self.cols = cols
        self.rows = rows
        self.pid: Optional[int] = None
        self.fd: Optional[int] = None
        self.alive = False

    def start(self, cmd: str, cwd: str, env: Dict[str, str] = None):
        """Fork a PTY and start the command."""
        pid, fd = pty.fork()

        if pid == 0:
            # Child — exec the command
            try:
                # Set terminal size
                _set_winsize(pty.STDOUT_FILENO, self.rows, self.cols)
                run_env = {
                    **os.environ,
                    "TERM": "xterm-256color",
                    "COLORTERM": "truecolor",
                    "COLUMNS": str(self.cols),
                    "LINES": str(self.rows),
                    **(env or {}),
                }
                os.chdir(cwd)
                os.execve("/bin/bash", ["/bin/bash", "-c", cmd], run_env)
            except Exception as e:
                os.write(pty.STDOUT_FILENO, f"\r\nError: {e}\r\n".encode())
                os._exit(1)
        else:
            # Parent
            self.pid = pid
            self.fd = fd
            self.alive = True
            # Set non-blocking
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            _set_winsize(fd, self.rows, self.cols)

    def resize(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        if self.fd is not None:
            try:
                _set_winsize(self.fd, rows, cols)
            except Exception:
                pass

    def write(self, data: bytes):
        if self.fd is not None and self.alive:
            try:
                os.write(self.fd, data)
            except OSError:
                self.alive = False

    def read(self, n: int = 4096) -> Optional[bytes]:
        if self.fd is None or not self.alive:
            return None
        try:
            r, _, _ = select.select([self.fd], [], [], 0.02)
            if r:
                return os.read(self.fd, n)
        except OSError:
            self.alive = False
        return None

    def kill(self):
        if self.pid:
            try:
                os.kill(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self.fd:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.alive = False

    def check_alive(self) -> bool:
        if not self.alive or self.pid is None:
            return False
        try:
            result = os.waitpid(self.pid, os.WNOHANG)
            if result[0] != 0:
                self.alive = False
        except ChildProcessError:
            self.alive = False
        return self.alive


def _set_winsize(fd: int, rows: int, cols: int):
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


# Global session registry
_pty_sessions: Dict[str, PTYSession] = {}


def get_or_create_session(session_id: str, cols: int = 220, rows: int = 50) -> PTYSession:
    if session_id not in _pty_sessions:
        _pty_sessions[session_id] = PTYSession(session_id, cols, rows)
    return _pty_sessions[session_id]


def kill_session(session_id: str):
    if session_id in _pty_sessions:
        _pty_sessions[session_id].kill()
        del _pty_sessions[session_id]


async def pty_websocket_handler(websocket, session_id: str, cwd: str, model: str):
    """
    Handle a WebSocket connection for a PTY terminal session.
    Protocol (JSON messages):
      Client → Server:
        {"type": "input",  "data": "<base64 or raw>"}
        {"type": "resize", "cols": N, "rows": N}
        {"type": "start",  "cmd": "...", "cwd": "...", "model": "..."}
        {"type": "kill"}
      Server → Client:
        {"type": "output", "data": "<bytes as latin-1 str>"}
        {"type": "exit",   "code": N}
        {"type": "error",  "message": "..."}
    """
    session = get_or_create_session(session_id)

    # Start openclaw if not already running
    if not session.alive:
        python = "python3"
        openclaw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openclaw")
        cmd = f"cd {cwd!r} && {python} -m openclaw.cli -d {cwd!r} -m {model!r} --no-banner 2>&1"
        env = {
            "PYTHONPATH": openclaw_dir,
            "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434"),
            "OPENCLAW_MODEL": model,
        }
        session.start(cmd, cwd, env)
        await websocket.send_json({"type": "connected", "session_id": session_id})

    async def read_loop():
        """Read PTY output and send to WebSocket."""
        while session.alive:
            data = session.read()
            if data:
                try:
                    await websocket.send_json({
                        "type": "output",
                        "data": data.decode("utf-8", errors="replace"),
                    })
                except Exception:
                    break
            else:
                await asyncio.sleep(0.02)
        try:
            await websocket.send_json({"type": "exit", "code": 0})
        except Exception:
            pass

    read_task = asyncio.create_task(read_loop())

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type", "")

            if msg_type == "input":
                data = msg.get("data", "")
                session.write(data.encode("utf-8", errors="replace"))

            elif msg_type == "resize":
                cols = int(msg.get("cols", 220))
                rows = int(msg.get("rows", 50))
                session.resize(cols, rows)

            elif msg_type == "kill":
                session.kill()
                break

            elif msg_type == "start":
                # Restart with new command
                if session.alive:
                    session.kill()
                new_cwd = msg.get("cwd", cwd)
                new_model = msg.get("model", model)
                new_cmd = msg.get("cmd", f"python3 -m openclaw.cli -d {new_cwd!r} -m {new_model!r}")
                session.start(new_cmd, new_cwd)

    except Exception:
        pass
    finally:
        read_task.cancel()
        try:
            await read_task
        except asyncio.CancelledError:
            pass
