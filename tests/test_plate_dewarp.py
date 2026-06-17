"""WP-A1: dewarp + enhance + OCR motor seçimi (model GEREKTİRMEZ, saf cv2/numpy).

Karanlık/açılı otopark dersi: EasyOCR il-kodunu (3→0/2) tutarlı yanlış okuyor.
Çözüm OCR-öncesi perspektif düzeltme (dewarp) + kontrast iyileştirme (enhance)
+ opsiyonel PaddleOCR motoru. Bu testler gerçek model/ağırlık olmadan koşar.
"""

from __future__ import annotations

import cv2
import numpy as np

from aura.plate import ocr as ocr_mod
from aura.plate.dewarp import dewarp_plate
from aura.plate.enhance import enhance_plate
from aura.plate.ocr import PaddleOCRReader, RealOCR, build_ocr


# --- dewarp_plate ---------------------------------------------------------- #
def test_dewarp_none_and_empty_pass_through():
    assert dewarp_plate(None) is None
    empty = np.zeros((0, 0, 3), np.uint8)
    assert dewarp_plate(empty).size == 0


def test_dewarp_identity_when_no_quad():
    # Düz tek-renk görüntü (kenar/çerçeve YOK) → güvenli dörtgen bulunamaz →
    # görüntü AYNEN döner (belirsizlikte yanlış warp uygulanmaz, K-004).
    img = np.full((40, 120, 3), 60, np.uint8)
    out = dewarp_plate(img)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_dewarp_too_small_is_identity():
    tiny = np.full((4, 4, 3), 100, np.uint8)
    out = dewarp_plate(tiny)
    assert np.array_equal(out, tiny)


def test_dewarp_corrects_skewed_rectangle_aspect():
    # Açılı (trapez) görünen açık plaka çerçevesi çiz: dewarp onu fronto-paralel
    # düzleştirmeli ve çıktı TR plaka en-boy oranına (~4.7) YAKLAŞMALI. Girişin
    # trapez oranından belirgin biçimde DÜZELMİŞ olduğunu kontrol ederiz.
    img = np.full((160, 320, 3), 30, np.uint8)  # koyu zemin (karanlık otopark taklidi)
    # Perspektifle sıkışmış (sağ kenarı dar) açık dörtgen — plaka çerçevesi.
    quad = np.array([[40, 40], [280, 60], [260, 120], [60, 110]], dtype=np.int32)
    cv2.fillConvexPoly(img, quad, (235, 235, 235))
    cv2.polylines(img, [quad], True, (0, 0, 0), 2)
    out = dewarp_plate(img)
    # Düzeltme gerçekleşti → şekil değişti (kimlik değil) ve oran TR'ye yakınlaştı.
    assert out.shape != img.shape
    h, w = out.shape[:2]
    ratio = w / h
    assert 3.5 <= ratio <= 6.0  # 520/110 ≈ 4.73 hedefi etrafında
    assert out.dtype == np.uint8


def test_dewarp_output_is_uint8_image():
    img = np.full((100, 240, 3), 50, np.uint8)
    quad = np.array([[20, 20], [210, 30], [205, 80], [25, 75]], dtype=np.int32)
    cv2.fillConvexPoly(img, quad, (240, 240, 240))
    out = dewarp_plate(img)
    assert out.dtype == np.uint8 and out.ndim == 3


# --- enhance_plate --------------------------------------------------------- #
def test_enhance_none_and_empty_pass_through(cfg):
    assert enhance_plate(None, cfg) is None
    empty = np.zeros((0, 0, 3), np.uint8)
    assert enhance_plate(empty, cfg).size == 0


def test_enhance_preserves_shape_and_uint8(cfg):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 80, size=(48, 160, 3), dtype=np.uint8)  # karanlık plaka
    out = enhance_plate(img, cfg)
    assert out.shape == img.shape  # şekil korunur (ölçek büyütme YOK)
    assert out.dtype == np.uint8


def test_enhance_brightens_dark_input(cfg):
    # gamma>1 + CLAHE karanlık girişin ortalama parlaklığını artırmalı.
    dark = np.full((40, 120, 3), 25, np.uint8)
    out = enhance_plate(dark, cfg)
    assert float(out.mean()) >= float(dark.mean())


def test_enhance_supports_grayscale(cfg):
    gray = np.full((40, 120), 30, np.uint8)
    out = enhance_plate(gray, cfg)
    assert out.ndim == 2 and out.dtype == np.uint8 and out.shape == gray.shape


# --- build_ocr: motor seçimi + paddle fallback ----------------------------- #
class _StubRealOCR(RealOCR):
    """easyocr/torch yüklemeden RealOCR yerine geçen hafif sahte."""

    def __init__(self, cfg):  # noqa: D401 - gerçek __init__ atlanır
        self.kind = "real"


class _StubPaddle(PaddleOCRReader):
    def __init__(self, cfg):
        self.kind = "paddle"


def test_build_ocr_paddle_missing_falls_back_to_easyocr(monkeypatch, cfg):
    # plate.ocr_engine=paddleocr ama paddleocr YOK → LOGLU olarak EasyOCR (RealOCR).
    monkeypatch.setitem(cfg.data["plate"], "ocr_engine", "paddleocr")
    monkeypatch.setitem(cfg.data["runtime"], "ai_mode", "real")
    monkeypatch.setattr(ocr_mod, "_easyocr_available", lambda: True)
    monkeypatch.setattr(ocr_mod, "_paddleocr_available", lambda: False)
    monkeypatch.setattr(ocr_mod, "RealOCR", _StubRealOCR)
    engine = build_ocr(cfg)
    assert getattr(engine, "kind", None) == "real"


def test_build_ocr_paddle_present_selects_paddle(monkeypatch, cfg):
    # paddleocr kurulu + seçili → PaddleOCRReader sarmalanır.
    monkeypatch.setitem(cfg.data["plate"], "ocr_engine", "paddleocr")
    monkeypatch.setitem(cfg.data["runtime"], "ai_mode", "real")
    monkeypatch.setattr(ocr_mod, "_easyocr_available", lambda: True)
    monkeypatch.setattr(ocr_mod, "_paddleocr_available", lambda: True)
    monkeypatch.setattr(ocr_mod, "PaddleOCRReader", _StubPaddle)
    engine = build_ocr(cfg)
    assert getattr(engine, "kind", None) == "paddle"


def test_build_ocr_default_engine_is_easyocr(monkeypatch, cfg):
    # ocr_engine belirtilmese/easyocr ise MEVCUT yol: RealOCR (paddle'a hiç bakmaz).
    monkeypatch.setitem(cfg.data["runtime"], "ai_mode", "real")
    monkeypatch.setattr(ocr_mod, "_easyocr_available", lambda: True)
    monkeypatch.setattr(ocr_mod, "RealOCR", _StubRealOCR)
    engine = build_ocr(cfg)
    assert getattr(engine, "kind", None) == "real"


# --- PaddleOCR adaptör çıktı uyumu (motor olmadan, sahte engine) ----------- #
def test_paddle_adapter_normalizes_list_format():
    # Eski PaddleOCR liste formatı → EasyOCR (box, text, conf) üçlüsü.
    class _Eng:
        def ocr(self, img, **kw):
            return [[[[[0, 0], [10, 0], [10, 8], [0, 8]], ("34TC8532", 0.92)]]]

    adapter = ocr_mod._PaddleAdapter(_Eng())
    out = adapter.readtext(np.zeros((8, 40, 3), np.uint8))
    assert len(out) == 1
    box, text, conf = out[0]
    assert text == "34TC8532" and abs(conf - 0.92) < 1e-6 and len(box) == 4


def test_paddle_adapter_normalizes_dict_format():
    # Yeni PaddleX sözlük formatı → aynı (box, text, conf) sözleşmesi.
    class _Eng:
        def ocr(self, img, **kw):
            return [
                {
                    "rec_texts": ["34", "TC8532"],
                    "rec_scores": [0.8, 0.95],
                    "dt_polys": [
                        [[0, 0], [5, 0], [5, 8], [0, 8]],
                        [[6, 0], [40, 0], [40, 8], [6, 8]],
                    ],
                }
            ]

    adapter = ocr_mod._PaddleAdapter(_Eng())
    out = adapter.readtext(np.zeros((8, 40, 3), np.uint8))
    assert [t for _, t, _ in out] == ["34", "TC8532"]
    assert all(len(b) == 4 for b, _, _ in out)
