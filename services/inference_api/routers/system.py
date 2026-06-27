"""Sistem router'ı — /health, /info."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from aura import __version__
from services.inference_api.security import verify_token_read

router = APIRouter(tags=["system"])


@router.get("/health")
def health(request: Request):
    sm = request.app.state.stream
    return {
        "status": "ok",
        "service": "inference_api",
        "version": __version__,
        "model_loaded": sm.pipeline is not None,
        "device": sm.cfg.get("runtime.device"),
        "ai_mode": sm.cfg.get("runtime.ai_mode"),
    }


@router.get("/info")
def info(request: Request, _=Depends(verify_token_read)):
    sm = request.app.state.stream
    cfg = sm.cfg
    return {
        "version": __version__,
        "config_summary": {
            "detector": cfg.get("models.detector.path"),
            "driver_state": cfg.get("models.driver_state.path"),
            "tracker": cfg.get("tracking.tracker"),
            "speed_mode": cfg.get("speed.mode"),
            "qod_backend": cfg.get("qod.backend"),
            "ai_mode": cfg.get("runtime.ai_mode"),
        },
        "status": sm.status(),
    }
