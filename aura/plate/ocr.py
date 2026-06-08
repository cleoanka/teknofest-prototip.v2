"""OCR motorları — gerçek (EasyOCR) ve deterministik mock.

- `RealOCR`: EasyOCR ile plaka ROI'sinden metin okur (sentetik videodaki çizili
  plakaları gerçekten okuyabilir).
- `MockOCR`: EasyOCR/torch yokken araç renginden senaryo plakasını üretir
  (track başına kararlı → voting konsensüsü oluşur).
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

log = logging.getLogger("aura.plate.ocr")
_NON_ALNUM = re.compile(r"[^A-Z0-9]")


class OCREngine(ABC):
    @abstractmethod
    def read(self, plate_roi, vehicle_crop=None) -> tuple[str | None, float]:
        """Plaka ROI'sinden (metin|None, güven) döndür."""
        raise NotImplementedError


class RealOCR(OCREngine):
    def __init__(self, cfg):
        import easyocr

        from aura.device import cuda_is_usable

        langs = list(cfg.get("plate.ocr_lang", ["tr"]))
        # GPU varsa (ve doğrulanmış torch derlemesiyle çalışıyorsa) OCR'ı da
        # hızlandır; aksi halde CPU. cuda_is_usable() önbellekli probe kullanır.
        use_gpu = bool(cfg.get("plate.ocr_gpu", True)) and cuda_is_usable()
        self.reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
        log.info("EasyOCR yüklendi (langs=%s, gpu=%s)", langs, use_gpu)

    def read(self, plate_roi, vehicle_crop=None) -> tuple[str | None, float]:
        if plate_roi is None or getattr(plate_roi, "size", 0) == 0:
            return None, 0.0
        results = self.reader.readtext(plate_roi)
        if not results:
            return None, 0.0
        best = max(results, key=lambda r: r[2])
        text = _NON_ALNUM.sub("", best[1].upper())
        return (text or None), float(best[2])


class MockOCR(OCREngine):
    """Araç rengi (BGR) → senaryo plakası. Track başına kararlı."""

    _PLATES = [
        ((90, 200, 255), "34ABC123"),
        ((120, 255, 120), "06FY4571"),
        ((200, 150, 255), "35TR07"),
    ]

    def __init__(self, cfg):
        self.max_dist = 180.0

    def read(self, plate_roi, vehicle_crop=None) -> tuple[str | None, float]:
        if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
            return None, 0.0
        mean = vehicle_crop.reshape(-1, vehicle_crop.shape[-1])[:, :3].mean(axis=0)
        best_plate, best_d = None, 1e9
        for color, plate in self._PLATES:
            d = float(np.linalg.norm(mean - np.array(color, dtype=float)))
            if d < best_d:
                best_d, best_plate = d, plate
        if best_plate is None or best_d > self.max_dist:
            return None, 0.0
        return best_plate, round(max(0.6, 1.0 - best_d / 300.0), 2)


def _easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except Exception:
        return False


def build_ocr(cfg) -> OCREngine:
    mode = str(cfg.get("runtime.ai_mode", "auto")).lower()
    if mode != "mock" and _easyocr_available():
        return RealOCR(cfg)
    if mode == "real" and not _easyocr_available():
        log.warning("ai_mode=real ama EasyOCR yok → mock OCR'a düşülüyor")
    return MockOCR(cfg)
