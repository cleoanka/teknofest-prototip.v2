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
