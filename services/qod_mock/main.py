"""CAMARA QoD (Quality-on-Demand) mock gateway — port 8081.

Gerçek CAMARA QoD sözleşmesini taklit eder: session aç/sorgula/sil. Final ortamında
yalnızca endpoint/credential Turkcell gateway'e çevrilir; sözleşme aynı kalır.

Çalıştır: uvicorn services.qod_mock.main:app --port 8081
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="AURA QoD Mock (CAMARA Quality-on-Demand)", version="2.0.0")

_sessions: dict[str, dict] = {}
_MAX_SESSIONS = 1000  # auth'suz mock: sınırsız POST /sessions ile bellek tükenmesini önle


def _expire_stale() -> None:
    """duration_seconds geçmiş oturumları lazy-temizle (CAMARA TTL sözleşme sadakati).

    Gerçek gateway oturumu süresi dolunca düşürür; mock'ta yalnız delete vardı →
    süresi dolan oturum sonsuza dek ACTIVE kalıp active_sessions sayacını şişiriyordu.
    """
    now = time.time()
    dead = [
        sid
        for sid, s in _sessions.items()
        if (s.get("duration_seconds") or 0) > 0 and now - s["created_at"] > s["duration_seconds"]
    ]
    for sid in dead:
        _sessions.pop(sid, None)


class SessionRequest(BaseModel):
    profile: str = Field(..., description="LOW_LATENCY | HIGH_THROUGHPUT")
    device_id: str
    duration_seconds: int | None = 60


class SessionResponse(BaseModel):
    session_id: str
    status: str
    granted_profile: str
    device_id: str
    created_at: float


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    count: int


@app.get("/health")
def health():
    return {"status": "ok", "service": "qod_mock", "active_sessions": len(_sessions)}


@app.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(req: SessionRequest):
    _expire_stale()  # önce süresi dolanları temizle
    if len(_sessions) >= _MAX_SESSIONS:
        raise HTTPException(status_code=429, detail="aktif oturum üst sınırı aşıldı")
    sid = uuid.uuid4().hex
    session = {
        "session_id": sid,
        "status": "ACTIVE",
        "granted_profile": req.profile,
        "device_id": req.device_id,
        "created_at": time.time(),
        "duration_seconds": req.duration_seconds,
    }
    _sessions[sid] = session
    return session


@app.get("/sessions", response_model=SessionListResponse)
def list_sessions():
    sessions = _sessions.values()
    return {"sessions": list(sessions), "count": len(sessions)}


@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@app.delete("/sessions/{session_id}", response_model=SessionResponse)
def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="session not found")
    s = _sessions.pop(session_id)
    s["status"] = "DELETED"
    return s
