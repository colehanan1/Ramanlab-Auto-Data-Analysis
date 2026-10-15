"""FastAPI app: dashboard, live state over SSE, mutes, Pi hand-off, media."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC = Path(__file__).parent / "static"


class MuteBody(BaseModel):
    muted: bool


class CompleteBody(BaseModel):
    output_dir: str
    push_to: str
    manifest: list[dict]


async def sse_frames(get_state, is_disconnected, sleep=asyncio.sleep, interval=1.0):
    while not await is_disconnected():
        yield f"data: {json.dumps(get_state(), default=str)}\n\n"
        await sleep(interval)


def _plain(name: str) -> bool:
    return bool(name) and "/" not in name and name not in (".", "..")


def create_app(hub, start_polling: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(hub.poll_forever()) if start_polling else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Rig Monitor", lifespan=lifespan)

    def rig_or_404(name: str):
        try:
            return hub.cfg.rig(name)
        except KeyError:
            raise HTTPException(404, f"unknown rig {name}")

    def session_rig(host: str, request: Request):
        rig = hub.resolve_rig(host, request.client.host if request.client else "")
        if rig is None:
            raise HTTPException(404, f"unknown rig {host}")
        return rig

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state():
        return hub.state()

    @app.get("/api/stream")
    async def stream(request: Request):
        return StreamingResponse(sse_frames(hub.state, request.is_disconnected),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.post("/api/mute/{rig}")
    def mute(rig: str, body: MuteBody):
        rig_or_404(rig)
        hub.store.set_muted(rig, body.muted)
        return {"rig": rig, "muted": body.muted}

    @app.post("/api/sessions/{host}/{batch_dir}/complete", status_code=202)
    def complete(host: str, batch_dir: str, body: CompleteBody, request: Request):
        rig = session_rig(host, request)
        try:
            hub.request_finalize(rig, batch_dir, body.output_dir, body.push_to, body.manifest)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"state": "pending"}

    @app.get("/api/sessions/{host}/{batch_dir}/status")
    def status(host: str, batch_dir: str, request: Request):
        rig = session_rig(host, request)
        s = hub.store.get_session(rig.name, batch_dir)
        if not s:
            return {"state": "unknown", "error": "", "dest": ""}
        return {"state": s["state"], "error": s["error"], "dest": s["dest"]}

    @app.post("/api/sessions/{host}/{batch_dir}/abort")
    def abort(host: str, batch_dir: str, request: Request):
        return {"state": hub.abort(session_rig(host, request), batch_dir)}

    @app.get("/media/{rig}/{batch_dir}/{name}")
    def media(rig: str, batch_dir: str, name: str):
        rig_or_404(rig)
        if not (_plain(batch_dir) and _plain(name) and name.endswith(".mp4")):
            raise HTTPException(404)
        s = hub.store.get_session(rig, batch_dir)
        try:
            base = (Path(s["dest"]) if s and s["state"] == "landed" and s["dest"]
                    else hub.transfer.staging_dir(rig, batch_dir))
        except ValueError:
            raise HTTPException(404)
        path = base / name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="video/mp4")

    @app.post("/api/mountcheck/{rig}", status_code=202)
    def mount_check(rig: str):
        rig_or_404(rig)
        if hub.mountcheck is None:
            raise HTTPException(409, "mount check is disabled in rigmon.yaml")
        if not hub.reachable.get(rig):
            raise HTTPException(409, "rig not reachable (is its script running?)")
        check_id = hub.start_mount_check(rig)
        if check_id is None:
            raise HTTPException(409, "a mount check is already running on this rig")
        return {"id": check_id}

    @app.get("/api/mountcheck/{rig}/latest")
    def mount_check_latest(rig: str):
        rig_or_404(rig)
        return hub.store.latest_mount_check(rig) or {}

    @app.get("/qc/{rig}/{batch_dir}/{name}")
    def qc_image(rig: str, batch_dir: str, name: str):
        rig_or_404(rig)
        if not (_plain(batch_dir) and _plain(name) and name.endswith(".jpg")):
            raise HTTPException(404)
        path = hub.cfg.qc_root / rig / batch_dir / name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/traces/{rig}/{batch_dir}/{name}")
    def trace_image(rig: str, batch_dir: str, name: str):
        rig_or_404(rig)
        if not (_plain(batch_dir) and _plain(name) and name.endswith(".png")):
            raise HTTPException(404)
        path = hub.cfg.qc_root / rig / batch_dir / "traces" / name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/png")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
