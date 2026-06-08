"""Plaka okuma: sweet spot + voting buffer + OCR.

M2: arayüz + stub (her zaman 'pending'). M5: sweet-spot gating (araç frame
sweet_spot'una girene kadar OCR pasif), ardışık okuma voting buffer'ı, konsensüs →
kalıcı yaz + OCR kapat (erken çıkış), ret → QoD kalite tetiği + yeniden okuma.
Türk plaka regex post-validasyonu.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from aura.schema import BBox, PlateState

if TYPE_CHECKING:
    import numpy as np


class PlateReader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sweet_spot = cfg.get("plate.sweet_spot", {})
        self.buffer_size = int(cfg.get("plate.voting_buffer_size", 7))
        self.consensus_ratio = float(cfg.get("plate.consensus_ratio", 0.6))
        self.regex = cfg.get("plate.regex", r"^\d{2}[A-Z]{1,3}\d{2,4}$")
        self.min_pixel_height = int(cfg.get("plate.min_pixel_height", 16))
        self._state: dict[int, PlateState] = {}

    def update(self, track_id: int, plate_roi: "np.ndarray | None",
               vehicle_bbox: BBox, frame_shape: tuple[int, ...],
               frame: "np.ndarray | None" = None) -> PlateState:
        """Track için plaka durumunu güncelle. M2: pending döndürür."""
        # TODO(M5): sweet-spot gating + voting buffer + OCR + regex + QoD tetik.
        return self._state.setdefault(track_id, PlateState())
