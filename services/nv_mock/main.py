"""Number Verification (NV) mock — port 8082.

Sessiz doğrulama: SIM/şebeke bağını kontrol eder (SMS/OTP yok). Mock kuralı: geçerli
sim_token + Türk numarası → verified. Final ortamında endpoint operatör NV API'sine çevrilir.

Çalıştır: uvicorn services.nv_mock.main:app --port 8082
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AURA Number Verification Mock", version="2.0.0")


class VerifyRequest(BaseModel):
    phone_number: str
    sim_token: str | None = None


class VerifyResponse(BaseModel):
    verified: bool
    latency_ms: int
    phone_number: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "nv_mock"}


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    normalized = req.phone_number.replace(" ", "")
    verified = bool(req.sim_token) and normalized.startswith(("+90", "90", "0"))
    return VerifyResponse(verified=verified, latency_ms=40, phone_number=req.phone_number)
