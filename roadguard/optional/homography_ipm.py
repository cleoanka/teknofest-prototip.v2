"""§8.3 Homography / IPM (Inverse Perspective Mapping).

Piksel koordinatlarını gerçek dünya metriklerine (metre) dönüştüren matris. Hız ve
yörünge hesabı için. Her kamera açısı uygun olmadığından toggle edilebilir.
`config.optional_modules.homography_ipm: true` + `speed.mode: ipm` ile etkin.

Kalibrasyon `speed.calibration_file` (ornek_kamera.yaml) ipm bölümünden okunur:
`src_points` (normalize ekran köşeleri) → `dst_points_m` (gerçek dünya metreleri).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger("roadguard.optional.homography_ipm")

_state: dict[int, dict] = {}
_ipm_cache: dict[str, IPM] = {}


class IPM:
    def __init__(self, src_points, dst_points_m):
        import cv2

        self.H = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points_m))

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        import cv2

        pt = np.array([[[x, y]]], dtype=np.float32)
        w = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(w[0]), float(w[1])


def _load_calib(cfg) -> dict | None:
    path = cfg.get("speed.calibration_file")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data.get("ipm")


def _get_ipm(calib: dict, key: str) -> IPM:
    ipm = _ipm_cache.get(key)
    if ipm is None:
        ipm = IPM(calib["src_points"], calib["dst_points_m"])
        _ipm_cache[key] = ipm
    return ipm


def ipm_speed(
    cfg, track_id: int, bbox, frame_idx: int, fps: float, frame_shape=None
) -> float | None:
    """IPM ile track hızı (km/h). Kalibrasyon yoksa None."""
    calib = _load_calib(cfg)
    if not calib or "src_points" not in calib or "dst_points_m" not in calib:
        return None
    ipm = _get_ipm(calib, str(cfg.get("speed.calibration_file")))
    H = frame_shape[0] if frame_shape else 1.0
    W = frame_shape[1] if frame_shape else 1.0
    nx = ((bbox.x1 + bbox.x2) / 2) / W  # alt-orta nokta, normalize
    ny = bbox.y2 / H
    wx, wy = ipm.to_world(nx, ny)
    st = _state.setdefault(track_id, {})
    prev = st.get("pos")
    st["pos"] = (wx, wy, frame_idx)
    if prev and frame_idx > prev[2]:
        dist = math.hypot(wx - prev[0], wy - prev[1])
        dt = (frame_idx - prev[2]) / max(fps, 1e-6)
        if dt > 0:
            return round(dist / dt * 3.6, 1)
    return None
