"""Config router'ı — /config (GET/PATCH). Çalışma zamanı ayarları."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from services.inference_api.models import ConfigPatch
from services.inference_api.security import verify_token

router = APIRouter(tags=["config"])


@router.get("/config")
def get_config(request: Request):
    return request.app.state.stream.cfg.as_dict()


@router.patch("/config")
def patch_config(patch: ConfigPatch, request: Request, _=Depends(verify_token)):
    sm = request.app.state.stream
    data = sm.cfg.data
    if patch.conf_threshold is not None:
        data.setdefault("models", {}).setdefault("detector", {})["conf"] = patch.conf_threshold
    if patch.bbox_overlay is not None:
        sm.bbox_overlay = patch.bbox_overlay
        data.setdefault("dashboard", {})["default_bbox"] = patch.bbox_overlay
    if patch.qod_profile is not None:
        # Aktif QoD profilini seç. `qod.profiles` bilinen profil adlarının haritası
        # (ör. optimize/quality); seçim oraya değil `qod.active_profile`'a yazılır ve
        # geçersiz ad sessizce kabul edilmez (422), aksi halde aktif profil bozulurdu.
        qod = data.setdefault("qod", {})
        known = set((qod.get("profiles") or {}).keys())
        if known and patch.qod_profile not in known:
            raise HTTPException(
                status_code=422,
                detail=f"bilinmeyen qod_profile: {patch.qod_profile} (geçerli: {sorted(known)})",
            )
        qod["active_profile"] = patch.qod_profile
    return {"status": "updated", "config": sm.cfg.as_dict()}
