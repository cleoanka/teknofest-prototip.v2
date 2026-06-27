"""Dinamik ön-işleme katmanı.

Her filtre config'ten aç/kapa (`preprocessing.*`). M2: arayüz + pass-through
(config bayrakları okunur). Filtre implementasyonları (far maskeleme, motion-blur
deconvolution, yansıma süpürme, occlusion) sonraki iterasyonda doldurulur — arayüz
sabit kalır, downstream etkilenmez.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Preprocessor:
    def __init__(self, cfg):
        self.headlight = cfg.get("preprocessing.headlight_suppression", False)
        self.motion_blur = cfg.get("preprocessing.motion_blur_correction", False)
        self.reflection = cfg.get("preprocessing.reflection_suppression", False)
        self.occlusion = cfg.get("preprocessing.occlusion_handling", False)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Çevresel gürültüyü temizle. M2: pass-through."""
        # TODO(filters): headlight/motion-blur/reflection/occlusion uygula.
        return frame
