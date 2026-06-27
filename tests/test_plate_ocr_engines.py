"""OCR motor mantığı: RealOCR.read (parlama filtresi, max_side, enhance_below ikinci
deneme), _merge_line çok-satır birleştirme, _FastPlateAdapter, MockOCR, build_ocr mock
yolları. Model GEREKTİRMEZ — easyocr/torch yüklenmez; reader sahte readtext ile enjekte.
"""

from __future__ import annotations

import numpy as np
import pytest

from roadguard.plate import ocr as ocr_mod
from roadguard.plate.ocr import MockOCR, RealOCR, _FastPlateAdapter, build_ocr


# --- RealOCR.read: easyocr yüklemeden sahte readtext ile -------------------- #
class _FakeReader:
    """easyocr.Reader.readtext sözleşmesini taklit eden sahte motor.

    `script` bir çağrı-listesidir: her çağrıda sıradaki readtext sonucu döner; böylece
    'ilk geçiş None → enhance_below ikinci deneme' dalı tetiklenebilir.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def readtext(self, img):
        self.calls.append(np.asarray(img).shape)
        return self.script[min(len(self.calls) - 1, len(self.script) - 1)]


def _make_real(reader, max_side=1280, enhance_below=64):
    """RealOCR'ı __init__'siz kur (easyocr yüklenmez); alanları elle set et."""
    r = RealOCR.__new__(RealOCR)
    r.reader = reader
    r.max_side = max_side
    r.enhance_below = enhance_below
    return r


def test_realocr_read_none_on_empty():
    r = _make_real(_FakeReader([[]]))
    assert r.read(None) == (None, 0.0)
    assert r.read(np.zeros((0, 0, 3), np.uint8)) == (None, 0.0)


def test_realocr_glare_filter_skips_bright_low_variance():
    # mean>215 & std<25 → ışık kaynağı (plaka değil) → OCR'a hiç girmeden None.
    r = _make_real(_FakeReader([[]]))
    glare = np.full((40, 120, 3), 240, np.uint8)  # çok parlak, düz (std~0)
    assert r.read(glare) == (None, 0.0)
    assert r.reader.calls == []  # readtext HİÇ çağrılmadı


def test_realocr_reads_single_box():
    box = [[0, 0], [40, 0], [40, 10], [0, 10]]
    r = _make_real(_FakeReader([[(box, "34TC8532", 0.9)]]))
    roi = np.full((30, 90, 3), 120, np.uint8)
    text, conf = r.read(roi)
    assert text == "34TC8532" and abs(conf - 0.9) < 1e-6


def test_realocr_resizes_when_over_max_side():
    box = [[0, 0], [40, 0], [40, 10], [0, 10]]
    r = _make_real(_FakeReader([[(box, "34TC8532", 0.9)]]), max_side=100)
    roi = np.full((50, 300, 3), 120, np.uint8)  # uzun kenar 300 > 100
    r.read(roi)
    # readtext'e giren görüntünün uzun kenarı max_side'a indirilmiş olmalı
    h, w = r.reader.calls[0][:2]
    assert max(h, w) <= 100


def test_realocr_enhance_below_second_attempt():
    # İlk geçiş None + roi yüksekliği < enhance_below → CLAHE+2x ile ikinci deneme.
    box = [[0, 0], [40, 0], [40, 10], [0, 10]]
    r = _make_real(
        _FakeReader([[], [(box, "34TC8532", 0.8)]]),  # 1.: boş, 2.: başarılı
        enhance_below=64,
    )
    roi = np.full((30, 90, 3), 80, np.uint8)  # yükseklik 30 < 64
    text, conf = r.read(roi)
    assert text == "34TC8532"
    assert len(r.reader.calls) == 2  # ikinci deneme yapıldı


def test_realocr_no_second_attempt_when_tall():
    # roi yüksekliği >= enhance_below → ilk None final, ikinci deneme YOK.
    r = _make_real(_FakeReader([[]]), enhance_below=20)
    roi = np.full((40, 120, 3), 80, np.uint8)  # 40 >= 20
    text, conf = r.read(roi)
    assert text is None
    assert len(r.reader.calls) == 1


# --- _merge_line çok-satır/çok-kutu birleştirme --------------------------- #
def test_merge_line_concats_same_line_left_to_right():
    # Aynı satırdaki üç kutu (34 / TC / 8532) x'e göre sıralanıp birleştirilir.
    results = [
        ([[24, 0], [44, 0], [44, 8], [24, 8]], "8532", 0.95),
        ([[0, 0], [10, 0], [10, 8], [0, 8]], "34", 0.9),
        ([[12, 0], [22, 0], [22, 8], [12, 8]], "TC", 0.8),
    ]
    text, conf = RealOCR._merge_line(results)
    assert text == "34TC8532"  # soldan sağa
    assert abs(conf - (0.95 + 0.9 + 0.8) / 3) < 1e-6


def test_merge_line_excludes_other_row():
    # Best kutunun satırındakiler birleşir; farklı y'deki kutu (alt satır) dışlanır.
    results = [
        ([[0, 0], [20, 0], [20, 8], [0, 8]], "34TC", 0.95),  # best (üst satır)
        ([[22, 0], [42, 0], [42, 8], [22, 8]], "8532", 0.9),  # aynı satır
        ([[0, 80], [20, 80], [20, 88], [0, 88]], "XX", 0.5),  # çok aşağıda → dışla
    ]
    text, _ = RealOCR._merge_line(results)
    assert "XX" not in text
    assert text == "34TC8532"


def test_merge_line_empty_returns_none():
    assert RealOCR._merge_line([]) == (None, 0.0)


def test_merge_line_strips_non_alnum_and_uppercases():
    results = [([[0, 0], [40, 0], [40, 8], [0, 8]], "34-tc 8532!", 0.9)]
    text, _ = RealOCR._merge_line(results)
    assert text == "34TC8532"


# --- _FastPlateAdapter ----------------------------------------------------- #
class _Pred:
    def __init__(self, plate, char_probs=None):
        self.plate = plate
        self.char_probs = char_probs


class _FakeRecognizer:
    def __init__(self, preds, raise_exc=False):
        self.preds = preds
        self.raise_exc = raise_exc
        self.last_gray_ndim = None

    def run(self, img, return_confidence=False):
        if self.raise_exc:
            raise RuntimeError("motor patladı")
        self.last_gray_ndim = getattr(img, "ndim", None)
        return self.preds


def test_fastplate_adapter_single_box_with_conf():
    rec = _FakeRecognizer([_Pred("34TC8532", char_probs=[0.9, 0.8, 0.95, 0.9])])
    out = _FastPlateAdapter(rec).readtext(np.full((20, 80, 3), 100, np.uint8))
    assert len(out) == 1
    box, text, conf = out[0]
    assert text == "34TC8532"
    assert 0.0 < conf <= 1.0  # char_probs ortalaması
    assert len(box) == 4
    assert rec.last_gray_ndim == 2  # 3-kanal BGR griye çevrildi


def test_fastplate_adapter_empty_on_no_preds():
    assert _FastPlateAdapter(_FakeRecognizer([])).readtext(np.zeros((8, 8, 3), np.uint8)) == []


def test_fastplate_adapter_empty_on_no_plate():
    rec = _FakeRecognizer([_Pred(None)])
    assert _FastPlateAdapter(rec).readtext(np.zeros((8, 8, 3), np.uint8)) == []


def test_fastplate_adapter_empty_on_engine_error():
    rec = _FakeRecognizer([], raise_exc=True)
    assert _FastPlateAdapter(rec).readtext(np.zeros((8, 8, 3), np.uint8)) == []


def test_fastplate_adapter_empty_on_none_image():
    assert _FastPlateAdapter(_FakeRecognizer([_Pred("X")])).readtext(None) == []


def test_fastplate_adapter_zero_conf_when_no_probs():
    rec = _FakeRecognizer([_Pred("34TC8532", char_probs=None)])
    out = _FastPlateAdapter(rec).readtext(np.full((20, 80, 3), 100, np.uint8))
    assert out[0][2] == 0.0


# --- MockOCR --------------------------------------------------------------- #
def test_mock_ocr_matches_nearby_color(cfg):
    m = MockOCR(cfg)
    # ilk senaryo rengi (90,200,255) → '34ABC123'
    crop = np.full((10, 10, 3), 0, np.uint8)
    crop[:] = (90, 200, 255)
    text, conf = m.read(None, crop)
    assert text == "34ABC123"
    assert conf >= 0.6


def test_mock_ocr_none_on_far_color(cfg):
    m = MockOCR(cfg)
    m.max_dist = 5.0  # çok sıkı → hiçbir senaryo rengine yeterince yakın değil
    crop = np.full((10, 10, 3), 0, np.uint8)
    crop[:] = (0, 0, 0)
    assert m.read(None, crop) == (None, 0.0)


def test_mock_ocr_none_on_empty_crop(cfg):
    m = MockOCR(cfg)
    assert m.read(None, None) == (None, 0.0)
    assert m.read(None, np.zeros((0, 0, 3), np.uint8)) == (None, 0.0)


def test_mock_ocr_max_dist_from_config(cfg):
    # cfg'ten okunabilir; anahtar yoksa varsayılan 180.0 (eski hardcoded davranış).
    m = MockOCR(cfg)
    assert m.max_dist == 180.0


def test_mock_ocr_max_dist_none_cfg_defaults_180():
    m = MockOCR(None)
    assert m.max_dist == 180.0


# --- build_ocr mock yolları ----------------------------------------------- #
def test_build_ocr_explicit_mock_mode(monkeypatch, cfg):
    monkeypatch.setitem(cfg.data["runtime"], "ai_mode", "mock")
    assert isinstance(build_ocr(cfg), MockOCR)


def test_build_ocr_auto_synthetic_source_uses_mock(monkeypatch, cfg):
    monkeypatch.setitem(cfg.data["runtime"], "ai_mode", "auto")
    monkeypatch.setattr(ocr_mod, "is_synthetic_source", lambda c: True)
    assert isinstance(build_ocr(cfg), MockOCR)


def test_any_real_ocr_available_counts_fastplate(monkeypatch, cfg):
    """Düzeltme: yapılandırılan motor (fastplate) varsa EasyOCR yok bile gerçek-OCR
    mevcut sayılır → sessizce MockOCR'a (sahte plaka) DÜŞMEZ."""
    monkeypatch.setitem(cfg.data.setdefault("plate", {}), "ocr_engine", "fastplate")
    monkeypatch.setattr(ocr_mod, "_easyocr_available", lambda: False)
    monkeypatch.setattr(ocr_mod, "_fastplate_available", lambda: True)
    assert ocr_mod._any_real_ocr_available(cfg) is True


def test_build_ocr_real_mode_no_real_ocr_raises(monkeypatch, cfg):
    """K-004: ai_mode=real iken HİÇBİR gerçek OCR yoksa açık hata (mock sahte plaka üretmesin)."""
    monkeypatch.setitem(cfg.data["runtime"], "ai_mode", "real")
    monkeypatch.setattr(ocr_mod, "_easyocr_available", lambda: False)
    monkeypatch.setattr(ocr_mod, "_fastplate_available", lambda: False)
    monkeypatch.setattr(ocr_mod, "_paddleocr_available", lambda: False)
    with pytest.raises(RuntimeError):
        build_ocr(cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
