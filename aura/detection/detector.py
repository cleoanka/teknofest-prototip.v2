"""Stage-1 tespit + ROI kırpma arayüzü ve dedektör fabrikası.

- `YOLO26Detector` (gerçek): ultralytics YOLO + ByteTrack (lazy: aura/detection/yolo.py)
- `MockDetector` (deterministik numpy): parlak araç bloklarını eşikler, IoU-takip eder
  (lazy: aura/detection/mock.py) → model/ağırlık olmadan tüm hat uçtan-uca çalışır
- `StubDetector`: boş çıktı (test/iskelet)

`ai_mode` çözümlemesi: real | mock | auto (ultralytics+ağırlık varsa real, yoksa mock).
Tasarım kuralı: downstream'e asla tam kare gönderilmez; yalnızca ROI crop'lar.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from aura.config import is_synthetic_source
from aura.schema import BBox

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("aura.detection")


@dataclass
class Detection:
    """Bir araç tespiti + (opsiyonel) takip ID'si + ROI crop'ları."""

    bbox: BBox
    track_id: int | None = None
    cabin_roi: np.ndarray | None = field(default=None, repr=False)
    plate_roi: np.ndarray | None = field(default=None, repr=False)


@dataclass
class Person:
    """Bir kişi tespiti + ByteTrack takip ID'si (sürücü kilidi için kullanılır)."""

    bbox: BBox
    track_id: int | None = None


class Detector(ABC):
    """Tespit motoru soyut arayüzü (gerçek/mock implementasyonlar bunu uygular).

    Alt sınıflar her ``detect()`` çağrısından sonra o karede bulunan kişileri
    ``last_persons`` listesine yazar (sürücü kilidi bunları tüketir). Araç tespiti
    yapmayan/kişi üretmeyen implementasyonlar bunu boş bırakır.
    """

    #: Son karede tespit edilen kişiler (her detect() çağrısında güncellenir)
    last_persons: list[Person] = []

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Kareyi işle → araç tespitleri (track_id atanmış olabilir)."""
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 - opsiyonel hook (alt sınıflar override edebilir)
        """Opsiyonel kaynak temizliği (gerçek dedektörler override eder)."""


class StubDetector(Detector):
    """Tespit üretmez (iskelet/test)."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []


# --------------------------------------------------------------------------- #
# Fabrika + ai_mode çözümleme
# --------------------------------------------------------------------------- #
def _ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def resolve_ai_mode(cfg) -> str:
    """real | mock — config.runtime.ai_mode'a ('auto' dahil) göre."""
    mode = str(cfg.get("runtime.ai_mode", "auto")).lower()
    if mode == "real":
        return "real"
    if mode == "mock":
        return "mock"
    # auto: ultralytics kurulu VE detector ağırlığı mevcutsa real, aksi halde mock
    weight = Path(cfg.get("models.detector.path", "weights/yolo26s.pt"))
    if not weight.is_absolute():
        weight = Path(__file__).resolve().parents[2] / weight
    if not (_ultralytics_available() and weight.exists()):
        return "mock"
    # auto + ağırlık var: gömülü sentetik örnekte (renkli bloklar) COCO-YOLO araç
    # göremez → mock (zengin, çalışan demo). Gerçek footage/kamera → gerçek YOLO.
    if is_synthetic_source(cfg):
        return "mock"
    return "real"


def build_detector(cfg) -> Detector:
    """Config'e göre dedektör kur (ağır backend'ler lazy import edilir)."""
    mode = resolve_ai_mode(cfg)
    if mode == "real":
        from aura.detection.yolo import YOLO26Detector

        log.info("Detector: YOLO26 (gerçek) + %s", cfg.get("tracking.tracker", "bytetrack"))
        return YOLO26Detector(cfg)
    from aura.detection.mock import MockDetector

    log.info("Detector: deterministik MOCK (ağırlık yok / ai_mode=mock)")
    return MockDetector(cfg)


# --------------------------------------------------------------------------- #
# ROI geometri (modelden bağımsız, saf hesap)
# --------------------------------------------------------------------------- #
def crop_rois(
    frame: np.ndarray, bbox: BBox, cabin_ratio: float = 0.55
) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def crop_person_roi(
    frame: np.ndarray, bbox: BBox, pad_ratio: float = 0.15
) -> np.ndarray | None:
    """Kilitli sürücünün kutusundan ROI kes (kenarlardan `pad_ratio` kadar pay bırakır).

    Geometrik 'üst %55 kabin' tahmini yerine, sürücü olarak kilitlenmiş kişinin
    gerçek kutusundan kırpar; Stage-2 (YOLO26l driver_state) yalnızca bu crop'ta çalışır.
    """
    h, w = frame.shape[:2]
    pad_x = bbox.width * pad_ratio
    pad_y = bbox.height * pad_ratio
    x1 = max(0, int(bbox.x1 - pad_x))
    y1 = max(0, int(bbox.y1 - pad_y))
    x2 = min(w, int(bbox.x2 + pad_x))
    y2 = min(h, int(bbox.y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2].copy()
    return roi if roi.size else None
