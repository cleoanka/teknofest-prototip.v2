"""OCR motor sarmalayıcısı: PaddleOCR sürüm-robust init + _PaddleAdapter (model gerektirmez)."""

from __future__ import annotations

import pytest

from roadguard.plate.ocr import PaddleOCRReader, _PaddleAdapter


# --- PaddleOCR sürüm-robust __init__ (W1 adversaryal düzeltmesi) ---------- #
class _ModernPaddle:
    """PaddleOCR 3.x: use_textline_orientation + device; eski bayrakları reddeder."""

    def __init__(self, **kw):
        if "use_angle_cls" in kw or "use_gpu" in kw or "show_log" in kw:
            raise TypeError("unexpected legacy kwarg")
        if "use_textline_orientation" not in kw:
            raise TypeError("use_textline_orientation zorunlu")
        self.kw = kw


class _LegacyPaddle:
    """PaddleOCR <=2.6: use_angle_cls + use_gpu; modern bayrakları reddeder."""

    def __init__(self, **kw):
        if "use_textline_orientation" in kw or "device" in kw:
            raise TypeError("unexpected modern kwarg")
        if "use_angle_cls" not in kw:
            raise TypeError("use_angle_cls zorunlu")
        self.kw = kw


class _LangOnlyPaddle:
    """Hiçbir orientation/cihaz bayrağını kabul etmeyen sürüm (son çare: yalnız lang)."""

    def __init__(self, **kw):
        if set(kw) - {"lang"}:
            raise TypeError("yalnız lang kabul edilir")
        self.kw = kw


def test_build_engine_modern_signature_keeps_orientation_and_device():
    eng = PaddleOCRReader._build_engine(_ModernPaddle, use_gpu=True)
    assert isinstance(eng, _ModernPaddle)
    assert eng.kw["use_textline_orientation"] is True  # oryantasyon AÇIK
    assert eng.kw["device"] == "gpu"  # GPU geçti
    # CPU yolu da device geçirir
    eng_cpu = PaddleOCRReader._build_engine(_ModernPaddle, use_gpu=False)
    assert eng_cpu.kw["device"] == "cpu"


def test_build_engine_falls_back_to_legacy_signature():
    eng = PaddleOCRReader._build_engine(_LegacyPaddle, use_gpu=True)
    assert isinstance(eng, _LegacyPaddle)
    assert eng.kw["use_angle_cls"] is True  # açı sınıflandırması AÇIK (sessizce kaybolmaz)
    assert eng.kw["use_gpu"] is True  # GPU geçti
    eng_cpu = PaddleOCRReader._build_engine(_LegacyPaddle, use_gpu=False)
    assert eng_cpu.kw["use_gpu"] is False


def test_build_engine_last_resort_lang_only():
    # Hiçbir orientation/device bayrağı kabul edilmiyorsa yine de kurulum başarılı olur.
    eng = PaddleOCRReader._build_engine(_LangOnlyPaddle, use_gpu=True)
    assert isinstance(eng, _LangOnlyPaddle)
    assert eng.kw == {"lang": "en"}


# --- _PaddleAdapter çıktı normalizasyonu (eski liste / yeni sözlük) ------- #
class _DictEngine:
    def ocr(self, img, **kw):
        return [
            {
                "rec_texts": ["34", "TC", "8532"],
                "rec_scores": [0.9, 0.8, 0.95],
                "dt_polys": [
                    [[0, 0], [10, 0], [10, 8], [0, 8]],
                    [[12, 0], [22, 0], [22, 8], [12, 8]],
                    [[24, 0], [44, 0], [44, 8], [24, 8]],
                ],
            }
        ]


class _ListEngine:
    def ocr(self, img, **kw):
        return [
            [
                [[[0, 0], [10, 0], [10, 8], [0, 8]], ("34TC8532", 0.91)],
            ]
        ]


@pytest.mark.parametrize(
    "engine,expected",
    [
        (_DictEngine(), [("34", 0.9), ("TC", 0.8), ("8532", 0.95)]),
        (_ListEngine(), [("34TC8532", 0.91)]),
    ],
)
def test_paddle_adapter_normalizes_both_formats(engine, expected):
    out = _PaddleAdapter(engine).readtext(object())
    assert len(out) == len(expected)
    for (box, txt, conf), (etxt, econf) in zip(out, expected, strict=True):
        assert txt == etxt
        assert abs(conf - econf) < 1e-6
        assert isinstance(box, list) and len(box) >= 1
