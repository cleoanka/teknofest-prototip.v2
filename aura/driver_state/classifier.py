"""Stage-2 sürücü durumu sınıflandırıcı (YOLO26l).

M2: arayüz + stub (tüm bayraklar False). M4: YOLO26l ile çoklu-etiket detection
(phone/smoking/no_seatbelt/fatigue). MediaPipe/landmark KESİNLİKLE kullanılmaz —
yorgunluk (kapalı göz/esneme/baş düşmesi) bir detection sınıfı olarak öğrenilir.

Girdi yalnızca sürücü kabini ROI'sidir (asla tam kare).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from aura.schema import DriverState

if TYPE_CHECKING:
    import numpy as np


class DriverStateClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.classes = list(cfg.get("models.driver_state.classes",
                                    ["phone", "smoking", "no_seatbelt", "fatigue"]))
        self.conf = float(cfg.get("models.driver_state.conf", 0.40))

    def infer(self, cabin_roi: "np.ndarray | None") -> DriverState:
        """Sürücü kabini ROI'sinden durum tespit et. M2: boş (hepsi False)."""
        # TODO(M4): YOLO26l detection → çoklu sınıf bayrakları + güven skorları.
        return DriverState()
