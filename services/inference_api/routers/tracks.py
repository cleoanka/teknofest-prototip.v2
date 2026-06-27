"""Track router'ı — /tracks, /tracks/{id}, /tracks/{id}/history."""

from __future__ import annotations

from collections import deque

from fastapi import APIRouter, Depends, HTTPException, Request

from services.inference_api.security import verify_token_read

router = APIRouter(tags=["tracks"])


@router.get("/tracks")
def list_tracks(request: Request, _=Depends(verify_token_read)):
    sm = request.app.state.stream
    if not sm.pipeline:
        return {"tracks": [], "count": 0}
    tracks = [r.model_dump() for r in sm.pipeline.acc.active_tracks()]
    return {"tracks": tracks, "count": len(tracks)}


@router.get("/tracks/{track_id}")
def get_track(track_id: int, request: Request, _=Depends(verify_token_read)):
    sm = request.app.state.stream
    # Sözleşme (KASITLI): pipeline yokken tekil sorgu 404 döner (o track YOK).
    # list_tracks/track_history ise "koleksiyon boş" semantiğiyle 200+boş döner.
    # Aynı temel duruma iki farklı kontrat — REST'te kaynak-yok=404 / liste-boş=200
    # ayrımıyla tutarlı; tüketici tekil-track için 404'ü bekler.
    rec = sm.pipeline.acc.get(track_id) if sm.pipeline else None
    if rec is None:
        raise HTTPException(status_code=404, detail="track not found")
    return rec.model_dump()


@router.get("/tracks/{track_id}/history")
def track_history(track_id: int, request: Request, _=Depends(verify_token_read)):
    sm = request.app.state.stream
    if not sm.pipeline:
        return {"track_id": track_id, "history": []}
    # Yalnız son 200 örneği TUT (bounded deque), toplam eşleşmeyi ayrıca say:
    # tüm seriyi bellekte biriktirip dilimlemek yerine sabit-bellek tarama.
    # Sözleşme aynı: history = son 200, count = toplam eşleşme.
    series: deque = deque(maxlen=200)
    count = 0
    # CA-001: deque'i doğrudan iterlemek, worker thread eş-zamanlı emit_annotation
    # (append) yaparken "deque mutated during iteration" → 500 verir. Kilitli snapshot
    # üzerinde gez (tutarlı kopya; worker append'i yarışa girmez).
    for a in sm.pipeline.emitter.snapshot_annotations():
        for t in a.tracks:
            if t["track_id"] == track_id:
                series.append({"frame_id": a.frame_id, "ts": a.ts, **t})
                count += 1
    return {"track_id": track_id, "history": list(series), "count": count}
