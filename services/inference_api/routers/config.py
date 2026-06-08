"""Config router'ı — /config (GET/PATCH). Çalışma zamanı ayarları."""

from __future__ import annotations

from fastapi import APIRouter, Request

from services.inference_api.models import ConfigPatch

router = APIRouter(tags=["config"])


@router.get("/config")
def get_config(request: Request):
    return request.app.state.stream.cfg.as_dict()


@router.patch("/config")
def patch_config(patch: ConfigPatch, request: Request):
    sm = request.app.state.stream
    data = sm.cfg.data
    if patch.conf_threshold is not None:
        data.setdefault("models", {}).setdefault("detector", {})["conf"] = patch.conf_threshold
    if patch.bbox_overlay is not None:
        sm.bbox_overlay = patch.bbox_overlay
        data.setdefault("dashboard", {})["default_bbox"] = patch.bbox_overlay
    if patch.qod_profile is not None:
        data.setdefault("qod", {}).setdefault("profiles", {})["quality"] = patch.qod_profile
    return {"status": "updated", "config": sm.cfg.as_dict()}
