"""Akış router'ı — start/stop/config/status + MJPEG video + WS annotations/events.

İki-kanal tasarım: `GET /stream/video` ham/annotated MJPEG; `WS /stream/annotations`
kare başına bbox; `WS /stream/events` AuraEvent stream'i. Dashboard bbox toggle'ı
client-side (canvas) yapar — sunucuya gidiş-geliş yok.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from services.inference_api.models import StreamConfigPatch, StreamStartRequest

router = APIRouter(tags=["stream"])
log = logging.getLogger("aura.api.stream")
_BOUNDARY = "auraframe"


@router.post("/stream/start")
def stream_start(req: StreamStartRequest, request: Request):
    sm = request.app.state.stream
    sm.start(source=req.source, device=req.device, bbox_overlay=req.bbox_overlay)
    return {"status": "started", **sm.status()}


@router.post("/stream/stop")
def stream_stop(request: Request):
    request.app.state.stream.stop()
    return {"status": "stopped"}


@router.patch("/stream/config")
def stream_config(patch: StreamConfigPatch, request: Request):
    sm = request.app.state.stream
    if patch.bbox_overlay is not None:
        sm.bbox_overlay = patch.bbox_overlay
    if patch.conf_threshold is not None:
        sm.cfg.data.setdefault("models", {}).setdefault("detector", {})["conf"] = patch.conf_threshold
    return sm.status()


@router.get("/stream/status")
def stream_status(request: Request):
    return request.app.state.stream.status()


@router.get("/stream/video")
def stream_video(request: Request, bbox: bool = Query(False)):
    """MJPEG akışı. `?bbox=true` → server-side çizimli; `false` → ham (dashboard canvas çizer)."""
    sm = request.app.state.stream

    def generate():
        idle = 0
        while True:
            jpg = sm.latest_jpeg(bbox)
            if jpg:
                idle = 0
                yield (b"--" + _BOUNDARY.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            else:
                idle += 1
                if not sm.running and idle > 20:
                    break
            time.sleep(0.04)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )


@router.websocket("/stream/annotations")
async def ws_annotations(ws: WebSocket):
    sm = ws.app.state.stream
    await ws.accept()
    q = sm.subscribe_annotations()
    try:
        while True:
            await ws.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        sm.unsubscribe_annotations(q)


@router.websocket("/stream/events")
async def ws_events(ws: WebSocket):
    sm = ws.app.state.stream
    await ws.accept()
    q = sm.subscribe_events()
    try:
        while True:
            await ws.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        sm.unsubscribe_events(q)
