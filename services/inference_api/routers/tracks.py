"""Track router'ı — /tracks, /tracks/{id}, /tracks/{id}/history."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["tracks"])


@router.get("/tracks")
def list_tracks(request: Request):
    sm = request.app.state.stream
    if not sm.pipeline:
        return {"tracks": [], "count": 0}
    tracks = [r.model_dump() for r in sm.pipeline.acc.active_tracks()]
    return {"tracks": tracks, "count": len(tracks)}


@router.get("/tracks/{track_id}")
def get_track(track_id: int, request: Request):
    sm = request.app.state.stream
    rec = sm.pipeline.acc.get(track_id) if sm.pipeline else None
    if rec is None:
        raise HTTPException(status_code=404, detail="track not found")
    return rec.model_dump()


@router.get("/tracks/{track_id}/history")
def track_history(track_id: int, request: Request):
    sm = request.app.state.stream
    if not sm.pipeline:
        return {"track_id": track_id, "history": []}
    series = []
    for a in sm.pipeline.emitter.annotations:
        for t in a.tracks:
            if t["track_id"] == track_id:
                series.append({"frame_id": a.frame_id, "ts": a.ts, **t})
    return {"track_id": track_id, "history": series[-200:], "count": len(series)}
