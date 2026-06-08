"""Kalibrasyon-bağımlı hız tahmini.

M2: arayüz + mode okuma (stub). M6: tripwire (iki sanal çizgi × gerçek mesafe),
ipm (homography, opsiyonel modül), disabled (hız iddiası yok → relative_velocity_flag).

Sistem kendi sınırlarını tanır: kalibrasyon yoksa hız uydurmaz, yalnızca anormal
göreli hız bayrağı üretir.
"""
from __future__ import annotations

from aura.schema import BBox, SpeedState


class SpeedEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.get("speed.mode", "disabled")
        self.calibration_file = cfg.get("speed.calibration_file")

    def update(self, track_id: int, bbox: BBox, frame_idx: int) -> SpeedState:
        """Track için hız durumu üret. M2: mode'a göre boş durum.

        disabled → value_kmh=None, relative_velocity_flag (M6'da hesaplanır).
        """
        # TODO(M6): tripwire frame-delta × mesafe; disabled için bbox-büyüme tabanlı
        # göreli hız anomali bayrağı.
        return SpeedState(mode=self.mode, value_kmh=None, relative_velocity_flag=False)
