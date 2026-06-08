"""inference_api istek/yanıt DTO'ları (pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StreamStartRequest(BaseModel):
    source: str | None = Field(None, description="Video path, kamera index ('0') veya RTSP URL")
    device: str | None = Field(None, description="auto | cpu | cuda | mps")
    bbox_overlay: bool = True


class StreamConfigPatch(BaseModel):
    bbox_overlay: bool | None = None
    conf_threshold: float | None = Field(None, ge=0.0, le=1.0)


class CameraInfo(BaseModel):
    index: int
    name: str
    width: int
    height: int


class CamerasResponse(BaseModel):
    cameras: list[CameraInfo]
    rtsp_supported: bool = True


class ConfigPatch(BaseModel):
    conf_threshold: float | None = Field(None, ge=0.0, le=1.0)
    qod_profile: str | None = None
    bbox_overlay: bool | None = None


class EvalRunRequest(BaseModel):
    source: str | None = None
    ground_truth: str | None = None
    qod_comparison: bool = True
