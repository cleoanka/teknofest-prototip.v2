"""Plaka okuma: sweet spot + voting buffer + OCR + Türk plaka regex.

Akış (plan.md §6.5):
  araç sweet-spot'a girene kadar OCR pasif → girince ardışık okuma + voting buffer
  → konsensüs: kalıcı yaz + OCR kapat (erken çıkış) + PLATE_CONFIRMED
  → ret: PLATE_REJECTED + QoD kalite tetiği + yeniden okuma döngüsü
Yetersiz piksel (min_pixel_height altı) → QoD kalite tetiği.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from aura.optional.loader import get_optional
from aura.plate.ocr import build_ocr
from aura.plate.voting import VotingBuffer
from aura.schema import BBox, PlateState

if TYPE_CHECKING:
    import numpy as np


class PlateReader:
    def __init__(self, cfg, qod=None, ocr=None):
        self.cfg = cfg
        ss = cfg.get("plate.sweet_spot", {}) or {}
        self.sweet_spot = {
            "x1": float(ss.get("x1", 0.0)), "y1": float(ss.get("y1", 0.0)),
            "x2": float(ss.get("x2", 1.0)), "y2": float(ss.get("y2", 1.0)),
        }
        self.buffer_size = int(cfg.get("plate.voting_buffer_size", 7))
        self.consensus_ratio = float(cfg.get("plate.consensus_ratio", 0.6))
        self.regex = re.compile(cfg.get("plate.regex", r"^\d{2}[A-Z]{1,3}\d{2,4}$"))
        self.min_pixel_height = int(cfg.get("plate.min_pixel_height", 16))
        self.ocr = ocr if ocr is not None else build_ocr(cfg)
        self.qod = qod
        self.sr = get_optional(cfg, "super_resolution")  # §8.2 (lazy; kapalıysa None)
        self._state: dict[int, PlateState] = {}
        self._buffers: dict[int, VotingBuffer] = {}

    # --- sweet spot -------------------------------------------------------- #
    def in_sweet_spot(self, bbox: BBox, frame_shape: tuple[int, ...]) -> bool:
        h, w = frame_shape[0], frame_shape[1]
        cx, cy = bbox.center
        s = self.sweet_spot
        return (s["x1"] * w <= cx <= s["x2"] * w) and (s["y1"] * h <= cy <= s["y2"] * h)

    # --- ana giriş --------------------------------------------------------- #
    def update(self, track_id: int, plate_roi: "np.ndarray | None", vehicle_bbox: BBox,
               frame_shape: tuple[int, ...], frame: "np.ndarray | None" = None) -> PlateState:
        st = self._state.setdefault(track_id, PlateState())
        if st.status == "confirmed":
            return st                       # erken çıkış (OCR kapalı)
        if st.status == "rejected":
            st.status = "pending"           # yeniden okuma döngüsü

        # Sweet spot: araç netlik bölgesine girene kadar OCR pasif
        if not self.in_sweet_spot(vehicle_bbox, frame_shape):
            return st

        # §8.2 süper çözünürlük (etkinse) — küçük plakaları OCR öncesi büyüt
        if self.sr is not None and plate_roi is not None:
            plate_roi = self.sr.enhance(plate_roi)

        # Yetersiz piksel → kalite tetiği (OCR'a girmeden)
        if plate_roi is not None and plate_roi.shape[0] < self.min_pixel_height:
            if self.qod:
                self.qod.request_quality(track_id, reason="low_pixel")
            return st

        vehicle_crop = None
        if frame is not None:
            x1, y1 = max(0, int(vehicle_bbox.x1)), max(0, int(vehicle_bbox.y1))
            x2, y2 = int(vehicle_bbox.x2), int(vehicle_bbox.y2)
            vehicle_crop = frame[y1:y2, x1:x2]

        text, conf = self.ocr.read(plate_roi, vehicle_crop)
        buf = self._buffers.setdefault(
            track_id, VotingBuffer(self.buffer_size, self.consensus_ratio)
        )
        buf.add(text)
        st.votes = buf.counts()

        if buf.full:
            value, frac = buf.consensus()
            if value and self.regex.match(value):
                st.value = value
                st.confidence = round(frac, 2)
                st.status = "confirmed"
                st.ocr_disabled = True
            else:
                st.status = "rejected"
                st.confidence = round(frac, 2)
                if self.qod:
                    self.qod.request_quality(track_id, reason="consensus_fail")
                buf.clear()                  # yeniden okuma için tamponu boşalt
        return st

    def get(self, track_id: int) -> PlateState | None:
        return self._state.get(track_id)
