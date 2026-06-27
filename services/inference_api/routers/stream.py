"""Akış router'ı — start/stop/config/status + MJPEG video + WS annotations/events.

İki-kanal tasarım: `GET /stream/video` ham/annotated MJPEG; `WS /stream/annotations`
kare başına bbox; `WS /stream/events` RoadGuardEvent stream'i. Dashboard bbox toggle'ı
client-side (canvas) yapar — sunucuya gidiş-geliş yok.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from services.inference_api.models import StreamConfigPatch, StreamStartRequest
from services.inference_api.security import validate_source, verify_token, verify_token_read

router = APIRouter(tags=["stream"])
log = logging.getLogger("roadguard.api.stream")
_BOUNDARY = "roadguardframe"
# MJPEG multipart başlığı kare-içeriğinden bağımsız sabit → bir kez encode et,
# her karede (.encode() + literal birleştirme) yeniden kurma.
_FRAME_HEAD = b"--" + _BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\n\r\n"
_FRAME_TAIL = b"\r\n"


@router.post("/stream/start")
def stream_start(req: StreamStartRequest, request: Request, _=Depends(verify_token)):
    sm = request.app.state.stream
    # SEC-002: kaynagi cv2.VideoCapture'a gecmeden once dogrula (SSRF guard).
    source = validate_source(req.source)
    sm.start(source=source, device=req.device, bbox_overlay=req.bbox_overlay)
    return {"status": "started", **sm.status()}


@router.post("/stream/stop")
def stream_stop(request: Request, _=Depends(verify_token)):
    request.app.state.stream.stop()
    return {"status": "stopped"}


@router.patch("/stream/config")
def stream_config(patch: StreamConfigPatch, request: Request, _=Depends(verify_token)):
    sm = request.app.state.stream
    if patch.bbox_overlay is not None:
        sm.bbox_overlay = patch.bbox_overlay
    if patch.conf_threshold is not None:
        sm.cfg.data.setdefault("models", {}).setdefault("detector", {})[
            "conf"
        ] = patch.conf_threshold
    return sm.status()


@router.get("/stream/status")
def stream_status(request: Request):
    return request.app.state.stream.status()


@router.get("/stream/video")
def stream_video(request: Request, bbox: bool = Query(False), _=Depends(verify_token_read)):
    """MJPEG akışı. `?bbox=true` → server-side çizimli; `false` → ham (dashboard canvas çizer).

    PII koruma opt-in (ROADGUARD_API_PROTECT_READS): `<img>` başlık gönderemediğinden
    `?token=` query-param ile kimlik doğrular (bkz. verify_token_read).
    """
    sm = request.app.state.stream
    # DoS guard: eş-zamanlı MJPEG slotu ayır; sınır aşılırsa 429 (worker tüketme yok).
    if not sm.try_acquire_mjpeg():
        return JSONResponse({"detail": "eş-zamanlı video akışı üst sınırı aşıldı"}, status_code=429)

    def generate():
        idle = 0
        try:
            while True:
                jpg = sm.latest_jpeg(bbox)
                if jpg:
                    idle = 0
                    yield _FRAME_HEAD + jpg + _FRAME_TAIL
                else:
                    idle += 1
                    if not sm.running and idle > 20:
                        break
                time.sleep(0.04)
        finally:
            sm.release_mjpeg()  # bağlantı kapanınca (normal/erken kopma) slotu bırak

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )


@router.websocket("/stream/annotations")
async def ws_annotations(ws: WebSocket):
    sm = ws.app.state.stream
    await ws.accept()
    q = sm.subscribe_annotations()
    if q is None:  # DoS guard: abone üst sınırı aşıldı → 1013 (try again later)
        await ws.close(code=1013)
        return
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
    if q is None:  # DoS guard: abone üst sınırı aşıldı → 1013 (try again later)
        await ws.close(code=1013)
        return
    try:
        while True:
            await ws.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        sm.unsubscribe_events(q)
