"""Kamera/kaynak router'ı — /cameras.

OpenCV ile 0–N indekslerini dener; platforma göre isim çözer (macOS AVFoundation,
Windows DirectShow). iPhone Continuity Camera standart webcam olarak listelenir.
`ROADGUARD_CAMERA_PROBE=0` ile donanım taraması atlanır (CI/başsız ortam).
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess

import cv2
from fastapi import APIRouter

from services.inference_api.models import CameraInfo, CamerasResponse

router = APIRouter(tags=["cameras"])
log = logging.getLogger("roadguard.api.cameras")


def _macos_camera_names() -> list[str]:
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=3.0,
        ).stdout
        names = []
        for line in out.splitlines():
            s = line.strip()
            if s.endswith(":") and not s.startswith("Camera") and "Model" not in s and ":" == s[-1]:
                if s[:-1] and "Unique" not in s and "Manufacturer" not in s:
                    names.append(s[:-1])
        return names
    except Exception:  # noqa: BLE001
        return []


def _name_for(index: int, mac_names: list[str]) -> str:
    if platform.system() == "Darwin" and index < len(mac_names):
        return mac_names[index]
    return f"Camera {index}"


def enumerate_cameras(max_index: int = 6) -> list[CameraInfo]:
    if os.environ.get("ROADGUARD_CAMERA_PROBE", "1") == "0":
        return []
    mac_names = _macos_camera_names() if platform.system() == "Darwin" else []
    cams: list[CameraInfo] = []
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                cams.append(CameraInfo(index=i, name=_name_for(i, mac_names), width=w, height=h))
            cap.release()
        except Exception:  # noqa: BLE001
            pass
    return cams


@router.get("/cameras", response_model=CamerasResponse)
def list_cameras():
    return CamerasResponse(cameras=enumerate_cameras(), rtsp_supported=True)
