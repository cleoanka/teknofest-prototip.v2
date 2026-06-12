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

from aura.config import is_synthetic_source

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
        # 4K araç crop'ları OCR'ı gereksiz yavaşlatır: uzun kenar bu değeri aşarsa
        # küçültülür (plaka okunaklılığı korunur, süre kat kat düşer).
        self.max_side = int(cfg.get("plate.ocr_max_side", 1280))
        # Küçük ROI'lerde (yükseklik < enhance_below) CLAHE+2x upscale varyantı denenir.
        self.enhance_below = int(cfg.get("plate.ocr_enhance_below_px", 64))
        log.info("EasyOCR yüklendi (langs=%s, gpu=%s)", langs, use_gpu)

    @staticmethod
    def _merge_line(results) -> tuple[str | None, float]:
        """EasyOCR segmentlerini satır bazında soldan sağa birleştir.

        Plaka çoğu karede '34' + 'TC' + '8532' gibi AYRI kutular halinde döner;
        yalnızca en güvenli tek kutuyu almak kesik okuma ('8532') üretir
        (v1 multi-block concat dersi). En güvenli kutunun satırındaki tüm
        kutular x'e göre sıralanıp birleştirilir.
        """
        if not results:
            return None, 0.0
        best = max(results, key=lambda r: r[2])
        bys = [p[1] for p in best[0]]
        b_cy = sum(bys) / len(bys)
        b_h = max(bys) - min(bys)
        line = []
        for box, txt, conf in results:
            ys = [p[1] for p in box]
            cy = sum(ys) / len(ys)
            if abs(cy - b_cy) <= max(b_h * 0.7, 8.0):
                line.append((min(p[0] for p in box), txt, conf))
        line.sort(key=lambda t: t[0])
        text = _NON_ALNUM.sub("", "".join(t[1] for t in line).upper())
        confs = [t[2] for t in line]
        return (text or None), float(sum(confs) / len(confs)) if confs else 0.0

    def _readtext(self, img) -> tuple[str | None, float]:
        return self._merge_line(self.reader.readtext(img))

    def read(self, plate_roi, vehicle_crop=None) -> tuple[str | None, float]:
        if plate_roi is None or getattr(plate_roi, "size", 0) == 0:
            return None, 0.0
        import cv2

        roi = plate_roi
        # Parlama/far testi (hidden_prototip dersi): aşırı parlak + düşük varyans
        # ROI ışık kaynağıdır, plaka değil → OCR'a hiç girmeden atla (FP + süre kazancı).
        mean = float(roi.mean())
        std = float(roi.std())
        if mean > 215.0 and std < 25.0:
            return None, 0.0
        h, w = roi.shape[:2]
        long_side = max(h, w)
        if long_side > self.max_side:
            scale = self.max_side / long_side
            roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        text, conf = self._readtext(roi)
        if text is None and roi.shape[0] < self.enhance_below:
            # Küçük/karanlık plaka: 2x büyüt + L kanalında CLAHE, bir kez daha dene.
            up = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
            lab_l, lab_a, lab_b = cv2.split(lab)
            lab_l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab_l)
            up = cv2.cvtColor(cv2.merge((lab_l, lab_a, lab_b)), cv2.COLOR_LAB2BGR)
            text, conf = self._readtext(up)
        return text, conf


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
    # auto + gömülü sentetik örnek → mock OCR (renk→plaka, hızlı ve deterministik;
    # detector/driver de bu kaynakta mock'a düştüğü için tüm hat tutarlı kalır)
    if mode == "auto" and is_synthetic_source(cfg):
        return MockOCR(cfg)
    if mode != "mock" and _easyocr_available():
        return RealOCR(cfg)
    if mode == "real" and not _easyocr_available():
        log.warning("ai_mode=real ama EasyOCR yok → mock OCR'a düşülüyor")
    return MockOCR(cfg)
