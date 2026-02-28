"""
AgentFlow — Docker Integration
Manages containers, images, volumes, and compose stacks via Docker socket.
Requires /var/run/docker.sock to be mounted (or DOCKER_HOST set).
"""
from __future__ import annotations
import asyncio
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/docker", tags=["docker"])


# ── Docker availability ────────────────────────────────────────────────────────
def _docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def _run(args: List[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a docker CLI command and return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            ["docker"] + args, capture_output=True, text=True,
            timeout=timeout, env={**os.environ}
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", "Docker not found. Is Docker installed?"

def _run_json(args: List[str]) -> Any:
    code, out, err = _run(args)
    if code != 0:
        raise HTTPException(500, f"Docker error: {err or 'unknown'}")
    if not out:
        return []
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # Try parsing JSONL (one JSON object per line)
        results = []
        for line in out.strip().splitlines():
            try:
                results.append(json.loads(line))
            except Exception:
                pass
        return results


# ── Status ─────────────────────────────────────────────────────────────────────
@router.get("/status")
async def docker_status():
    if not _docker_available():
        return {
            "available": False,
            "message": "Docker not available. Install Docker or mount /var/run/docker.sock",
        }
    code, out, err = _run(["info", "--format", "{{json .}}"])
    if code != 0:
        return {"available": False, "message": err}
    try:
        info = json.loads(out)
        return {
            "available": True,
            "version": info.get("ServerVersion"),
            "containers_running": info.get("ContainersRunning", 0),
            "containers_stopped": info.get("ContainersStopped", 0),
            "containers_paused": info.get("ContainersPaused", 0),
            "images": info.get("Images", 0),
            "memory_total": info.get("MemTotal", 0),
            "cpus": info.get("NCPU", 0),
            "os": info.get("OperatingSystem", ""),
            "arch": info.get("Architecture", ""),
            "driver": info.get("Driver", ""),
            "compose_available": _compose_available(),
        }
    except Exception as e:
        return {"available": True, "error": str(e)}

def _compose_available() -> bool:
    try:
        r = subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


# ── Containers ─────────────────────────────────────────────────────────────────
CONTAINER_FORMAT = (
    '{"id":"{{.ID}}","name":"{{.Names}}","image":"{{.Image}}",'
    '"status":"{{.Status}}","state":"{{.State}}","created":"{{.CreatedAt}}",'
    '"ports":"{{.Ports}}","size":"{{.Size}}"}'
)

@router.get("/containers")
async def list_containers(all: bool = Query(True)):
    args = ["ps", "--format", CONTAINER_FORMAT]
    if all:
        args.append("-a")
    code, out, err = _run(args)
    if code != 0:
        raise HTTPException(500, err)
    containers = []
    for line in out.splitlines():
        try:
            c = json.loads(line)
            c["name"] = c["name"].lstrip("/")
            containers.append(c)
        except Exception:
            pass
    return containers

@router.get("/containers/{container_id}")
async def inspect_container(container_id: str):
    data = _run_json(["inspect", container_id])
    if not data:
        raise HTTPException(404, "Container not found")
    c = data[0]
    cfg = c.get("Config", {})
    net = c.get("NetworkSettings", {})
    state = c.get("State", {})
    return {
        "id": c.get("Id", "")[:12],
        "name": c.get("Name", "").lstrip("/"),
        "image": cfg.get("Image"),
        "state": state.get("Status"),
        "running": state.get("Running"),
        "pid": state.get("Pid"),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
        "exit_code": state.get("ExitCode"),
        "env": cfg.get("Env", []),
        "cmd": cfg.get("Cmd", []),
        "entrypoint": cfg.get("Entrypoint", []),
        "ports": net.get("Ports", {}),
        "networks": list(net.get("Networks", {}).keys()),
        "mounts": [
            {"type": m.get("Type"), "source": m.get("Source"), "destination": m.get("Destination")}
            for m in c.get("Mounts", [])
        ],
        "restart_policy": c.get("HostConfig", {}).get("RestartPolicy", {}).get("Name"),
        "labels": cfg.get("Labels", {}),
    }

@router.post("/containers/{container_id}/start")
async def start_container(container_id: str):
    code, out, err = _run(["start", container_id])
    if code != 0:
        raise HTTPException(400, err)
    return {"started": True, "container": container_id}

@router.post("/containers/{container_id}/stop")
async def stop_container(container_id: str, timeout: int = Query(10)):
    code, out, err = _run(["stop", "-t", str(timeout), container_id])
    if code != 0:
        raise HTTPException(400, err)
    return {"stopped": True, "container": container_id}

@router.post("/containers/{container_id}/restart")
async def restart_container(container_id: str):
    code, out, err = _run(["restart", container_id])
    if code != 0:
        raise HTTPException(400, err)
    return {"restarted": True, "container": container_id}

@router.delete("/containers/{container_id}")
async def remove_container(container_id: str, force: bool = Query(False)):
    args = ["rm", container_id]
    if force:
        args = ["rm", "-f", container_id]
    code, out, err = _run(args)
    if code != 0:
        raise HTTPException(400, err)
    return {"removed": True, "container": container_id}

@router.get("/containers/{container_id}/logs")
async def get_logs(
    container_id: str,
    tail: int = Query(100, ge=10, le=10000),
    timestamps: bool = Query(False),
):
    args = ["logs", "--tail", str(tail), container_id]
    if timestamps:
        args.insert(2, "-t")
    code, out, err = _run(args, timeout=15)
    # Docker sends most log output to stderr
    combined = out + ("\n" + err if err else "")
    return {"logs": combined.strip(), "container": container_id}

@router.get("/containers/{container_id}/stats")
async def get_container_stats(container_id: str):
    """Get live CPU/memory/network stats (one shot)."""
    code, out, err = _run(["stats", "--no-stream", "--format",
        '{"name":"{{.Name}}","cpu":"{{.CPUPerc}}","mem":"{{.MemUsage}}",'
        '"mem_perc":"{{.MemPerc}}","net":"{{.NetIO}}","block":"{{.BlockIO}}","pids":"{{.PIDs}}"}',
        container_id])
    if code != 0:
        raise HTTPException(400, err)
    try:
        return json.loads(out)
    except Exception:
        return {"raw": out}

class RunContainerBody(BaseModel):
    image: str
    name: Optional[str] = None
    command: Optional[str] = None
    ports: Optional[Dict[str, str]] = None  # {"8080": "8080"}
    env: Optional[Dict[str, str]] = None
    volumes: Optional[Dict[str, str]] = None
    detach: bool = True
    remove_after: bool = False
    network: Optional[str] = None

@router.post("/containers/run")
async def run_container(body: RunContainerBody):
    args = ["run"]
    if body.detach:
        args.append("-d")
    if body.remove_after:
        args.append("--rm")
    if body.name:
        args += ["--name", body.name]
    if body.network:
        args += ["--network", body.network]
    if body.ports:
        for host, container in body.ports.items():
            args += ["-p", f"{host}:{container}"]
    if body.env:
        for k, v in body.env.items():
            args += ["-e", f"{k}={v}"]
    if body.volumes:
        for host, container in body.volumes.items():
            args += ["-v", f"{host}:{container}"]
    args.append(body.image)
    if body.command:
        args += body.command.split()
    code, out, err = _run(args, timeout=60)
    if code != 0:
        raise HTTPException(400, err or out)
    return {"started": True, "container_id": out[:12], "image": body.image}


# ── Images ─────────────────────────────────────────────────────────────────────
IMAGE_FORMAT = (
    '{"id":"{{.ID}}","repository":"{{.Repository}}","tag":"{{.Tag}}",'
    '"size":"{{.Size}}","created":"{{.CreatedAt}}"}'
)

@router.get("/images")
async def list_images():
    code, out, err = _run(["images", "--format", IMAGE_FORMAT])
    if code != 0:
        raise HTTPException(500, err)
    images = []
    for line in out.splitlines():
        try:
            images.append(json.loads(line))
        except Exception:
            pass
    return images

@router.post("/images/pull")
async def pull_image(image: str = Query(...)):
    """Pull an image (streaming progress via SSE)."""
    async def stream_pull():
        import shlex
        proc = await asyncio.create_subprocess_exec(
            "docker", "pull", image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                yield f"data: {json.dumps({'line': text})}\n\n"
        await proc.wait()
        success = proc.returncode == 0
        yield f"data: {json.dumps({'done': True, 'success': success})}\n\n"

    from sse_starlette.sse import EventSourceResponse
    return EventSourceResponse(stream_pull())

@router.delete("/images/{image_id:path}")
async def remove_image(image_id: str, force: bool = Query(False)):
    args = ["rmi", image_id]
    if force:
        args = ["rmi", "-f", image_id]
    code, out, err = _run(args)
    if code != 0:
        raise HTTPException(400, err)
    return {"removed": True, "image": image_id}

@router.get("/images/search")
async def search_images(q: str = Query(...), limit: int = Query(10)):
    code, out, err = _run(["search", "--format",
        '{"name":"{{.Name}}","description":"{{.Description}}","stars":"{{.StarCount}}","official":"{{.IsOfficial}}"}',
        "--limit", str(limit), q])
    if code != 0:
        raise HTTPException(500, err)
    results = []
    for line in out.splitlines():
        try:
            results.append(json.loads(line))
        except Exception:
            pass
    return results


# ── Volumes ────────────────────────────────────────────────────────────────────
@router.get("/volumes")
async def list_volumes():
    data = _run_json(["volume", "ls", "--format", "json"])
    return data if isinstance(data, list) else []

@router.delete("/volumes/{volume_name}")
async def remove_volume(volume_name: str):
    code, out, err = _run(["volume", "rm", volume_name])
    if code != 0:
        raise HTTPException(400, err)
    return {"removed": True, "volume": volume_name}


# ── Networks ───────────────────────────────────────────────────────────────────
@router.get("/networks")
async def list_networks():
    code, out, err = _run(["network", "ls", "--format",
        '{"id":"{{.ID}}","name":"{{.Name}}","driver":"{{.Driver}}","scope":"{{.Scope}}"}'])
    if code != 0:
        raise HTTPException(500, err)
    networks = []
    for line in out.splitlines():
        try:
            networks.append(json.loads(line))
        except Exception:
            pass
    return networks


# ── Docker Compose ─────────────────────────────────────────────────────────────
@router.get("/compose/stacks")
async def list_compose_stacks(compose_dir: str = Query(".")):
    """List compose stacks in a directory."""
    import glob
    patterns = [
        os.path.join(compose_dir, "docker-compose.yml"),
        os.path.join(compose_dir, "docker-compose.yaml"),
        os.path.join(compose_dir, "compose.yml"),
        os.path.join(compose_dir, "compose.yaml"),
        # Nested: project/docker-compose.yml
        os.path.join(compose_dir, "*/docker-compose.yml"),
        os.path.join(compose_dir, "*/docker-compose.yaml"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return [{"path": f, "name": os.path.basename(os.path.dirname(f) or f)} for f in files]

@router.post("/compose/up")
async def compose_up(
    compose_file: str = Query(...),
    services: str = Query(""),
    detach: bool = Query(True),
    build: bool = Query(False),
):
    args = ["-f", compose_file, "up"]
    if detach:
        args.append("-d")
    if build:
        args.append("--build")
    if services:
        args += services.split()

    async def stream():
        proc = await asyncio.create_subprocess_exec(
            "docker", "compose", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                yield f"data: {json.dumps({'line': text})}\n\n"
        await proc.wait()
        yield f"data: {json.dumps({'done': True, 'success': proc.returncode == 0})}\n\n"

    from sse_starlette.sse import EventSourceResponse
    return EventSourceResponse(stream())

@router.post("/compose/down")
async def compose_down(
    compose_file: str = Query(...),
    volumes: bool = Query(False),
):
    args = ["-f", compose_file, "down"]
    if volumes:
        args.append("-v")
    code, out, err = _run(["compose"] + args, timeout=60)
    return {"success": code == 0, "output": out, "error": err}

@router.get("/compose/ps")
async def compose_ps(compose_file: str = Query(...)):
    code, out, err = _run(["compose", "-f", compose_file, "ps", "--format", "json"])
    if code != 0:
        return []
    try:
        return json.loads(out)
    except Exception:
        return []


# ── System ─────────────────────────────────────────────────────────────────────
@router.post("/system/prune")
async def system_prune(volumes: bool = Query(False)):
    """Remove unused containers, networks, images."""
    args = ["system", "prune", "-f"]
    if volumes:
        args.append("--volumes")
    code, out, err = _run(args, timeout=120)
    return {"success": code == 0, "output": out}

@router.get("/system/df")
async def system_df():
    """Disk usage by Docker objects."""
    code, out, err = _run(["system", "df", "--format", "json"])
    if code != 0:
        return {"error": err}
    try:
        return json.loads(out)
    except Exception:
        return {"raw": out}


# ── Build ──────────────────────────────────────────────────────────────────────
class BuildImageBody(BaseModel):
    context: str  # path to build context
    dockerfile: Optional[str] = None
    tag: str
    build_args: Optional[Dict[str, str]] = None
    no_cache: bool = False

@router.post("/images/build")
async def build_image(body: BuildImageBody):
    args = ["build", "-t", body.tag]
    if body.dockerfile:
        args += ["-f", body.dockerfile]
    if body.no_cache:
        args.append("--no-cache")
    if body.build_args:
        for k, v in body.build_args.items():
            args += ["--build-arg", f"{k}={v}"]
    args.append(body.context)

    async def stream():
        proc = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                yield f"data: {json.dumps({'line': text})}\n\n"
        await proc.wait()
        yield f"data: {json.dumps({'done': True, 'success': proc.returncode == 0})}\n\n"

    from sse_starlette.sse import EventSourceResponse
    return EventSourceResponse(stream())
