"""Stage-1 tespit + ROI kırpma arayüzü.

M2: arayüz + ROI geometri + StubDetector (boş çıktı; akış doğru).
M3: YOLO26Detector (gerçek) + MockDetector (numpy deterministik) + ByteTrack.

Tasarım kuralı: downstream'e asla tam kare gönderilmez; yalnızca ROI crop'lar.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aura.schema import BBox

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Detection:
    """Bir araç tespiti + (opsiyonel) takip ID'si + ROI crop'ları."""

    bbox: BBox
    track_id: int | None = None
    cabin_roi: "np.ndarray | None" = field(default=None, repr=False)
    plate_roi: "np.ndarray | None" = field(default=None, repr=False)


class Detector(ABC):
    """Tespit motoru soyut arayüzü (gerçek/mock implementasyonlar bunu uygular)."""

    @abstractmethod
    def detect(self, frame: "np.ndarray") -> list[Detection]:
        """Kareyi işle → araç tespitleri (track_id atanmış olabilir)."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - opsiyonel kaynak temizliği
        pass


class StubDetector(Detector):
    """M2 yer tutucusu: tespit üretmez. Akışı bozmadan pipeline'ı çalıştırır."""

    def detect(self, frame: "np.ndarray") -> list[Detection]:
        return []


def build_detector(cfg) -> Detector:
    """Config'e göre dedektör kur.

    M2: her zaman StubDetector. M3'te ai_mode (real|mock|auto) çözümlemesi eklenir.
    """
    return StubDetector()


# --------------------------------------------------------------------------- #
# ROI geometri (modelden bağımsız, saf hesap) — M3'te dedektör çıktısına uygulanır.
# --------------------------------------------------------------------------- #
def crop_rois(frame: "np.ndarray", bbox: BBox, cabin_ratio: float = 0.55
              ) -> tuple["np.ndarray | None", "np.ndarray | None"]:
    """Araç bbox'ından iki ROI üret: (sürücü kabini=üst, plaka bölgesi=alt).

    YOLO26l ve OCR yalnızca bu küçük crop'lar üzerinde çalışır (zero-waste prensibi).
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.x1))
    y1 = max(0, int(bbox.y1))
    x2 = min(w, int(bbox.x2))
    y2 = min(h, int(bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None, None
    split = y1 + int((y2 - y1) * cabin_ratio)
    cabin = frame[y1:split, x1:x2].copy()
    plate = frame[split:y2, x1:x2].copy()
    cabin = cabin if cabin.size else None
    plate = plate if plate.size else None
    return cabin, plate
