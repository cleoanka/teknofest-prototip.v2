"""Gri-bölge erken-okuma yolu (reader._early_read) kapsamı — model GEREKTİRMEZ.

Doğrular:
  - lp_h < gray_zone_min_px → EK yol DA kapalı (çok küçük), oy YOK.
  - Gri-bölge [gray_zone_min_px, lp_vote_min_px): tek-motor misread (mutabakatsız,
    high_conf altı) oya GİRMEZ — güvenlik korunur.
  - Gri-bölge: ikinci motor MUTABAKATI varsa + format-geçerli → oya GİRER (weight_cap'li).
  - Gri-bölge: mutabakat motoru yok ama conf >= high_conf → oya GİRER.
  - SR/upscale + füzyon crop'u BÜYÜTÜR (OCR'a giren görüntü ham kırpıktan büyük).
  - lp_h >= lp_vote_min_px PLAIN okuma davranışı DEĞİŞMEZ (gri-bölge yalnız altta).

LP dedektörü sahte modelle enjekte edilir (gerçek YOLO yüklenmez); OCR sahte.
"""

from __future__ import annotations

import numpy as np

from aura.plate.reader import PlateReader
from aura.schema import BBox

FRAME_SHAPE = (360, 640, 3)


class FakeOCR:
    """Sabit (metin, güven) döndürür; OCR'a giren görüntüleri kaydeder (büyütme kanıtı)."""

    def __init__(self, value=("34TC8532", 0.95)):
        self.value = value
        self.calls: list = []

    def read(self, plate_roi, vehicle_crop=None):
        self.calls.append(None if plate_roi is None else np.asarray(plate_roi).copy())
        return self.value


class SeqOCR:
    """Çağrı sırasına göre farklı sonuç döndürür (mutabakat senaryoları)."""

    def __init__(self, seq):
        self.seq = list(seq)
        self.i = 0
        self.calls: list = []

    def read(self, plate_roi, vehicle_crop=None):
        self.calls.append(None if plate_roi is None else np.asarray(plate_roi).copy())
        v = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return v


class FakeQoD:
    def __init__(self):
        self.calls = []
        self.released = []

    def request_quality(self, track_id, reason):
        self.calls.append((track_id, reason))

    def release_quality(self, track_id):
        self.released.append(track_id)


class _Scalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _Listable:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return list(self._vals)


class _FakeBox:
    def __init__(self, xyxy, conf):
        self._xyxy = xyxy
        self.conf = _Scalar(conf)

    @property
    def xyxy(self):
        return [_Listable(self._xyxy)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeLP:
    def __init__(self, box_xyxy, conf=0.9):
        self.box_xyxy = box_xyxy
        self.conf = conf

    def predict(self, img, **kw):
        return [_FakeResult([_FakeBox(self.box_xyxy, self.conf)])]


def _center_bbox():
    return BBox(x1=300, y1=210, x2=340, y2=260, conf=0.9, cls="car")


def _reader(ocr, qod=None, **er):
    from aura.config import load_config

    r = PlateReader(load_config(), qod=qod, ocr=ocr)
    # Enjekte-OCR LP/early-read'i kapatır (ocr is None şartı); mekanizmayı test için aç.
    r._lp_enabled = True
    r._lp_failed = False
    r._lp_device = "cpu"
    r._er_enabled = True
    r.lp_vote_min_px = 45
    r._er_gray_min_px = er.get("gray_zone_min_px", 28)
    r._er_high_conf = er.get("high_conf", 0.90)
    r._er_require_agreement = er.get("require_engine_agreement", True)
    r._er_fuse_frames = er.get("fuse_frames", 5)
    r._er_weight_cap = er.get("weight_cap", 0.6)
    # min_weight'i düşür ki tek gri-bölge oyunun havuza GİRDİĞİni tek update'te görelim
    # (güvenlik kapısı = format+mutabakat; min_weight ayrı dürüstlük zırhı, ayrıca test edilir).
    return r


def _gray_lp(reader, lp_h):
    """lp_h yüksekliğinde gri-bölge LP kutusu kur (gray_zone_min_px <= lp_h < vote_min)."""
    reader._lp_model = FakeLP(box_xyxy=(0, 0, 60, lp_h))


def _roi():
    return np.full((120, 200, 3), 90, np.uint8)


# --- gray_zone_min_px ALTI: EK yol DA kapalı ----------------------------- #
def test_below_gray_zone_no_vote():
    ocr = FakeOCR(("34TC8532", 0.99))
    r = _reader(ocr)
    _gray_lp(r, 20)  # 20 < gray_zone_min_px(28) → EK yol kapalı
    r.update(1, _roi(), _center_bbox(), FRAME_SHAPE)
    assert 1 not in r._pools or not r._pools[1].raw_reads  # oy yazılmadı
    # OCR de hiç çağrılmamalı (gri-bölge yolu girmedi)
    assert ocr.calls == []


# --- gri-bölge: mutabakatsız tek-motor misread oya GİRMEZ ----------------- #
def test_gray_zone_single_engine_no_agreement_no_vote():
    # require_agreement=True ama ikinci motor kurulamaz (mock/enjekte) ve conf<high_conf →
    # 14TC857 tek-motor misread'i oya GİREMEZ (güvenlik).
    ocr = FakeOCR(("14TC857", 0.80))  # format-geçerli AMA conf<high_conf, mutabakat yok
    r = _reader(ocr, require_engine_agreement=True, high_conf=0.90)
    r._er_agree_built = True
    r._er_agree_ocr = None  # ikinci motor yok → conf-eşiğine düşer, 0.80<0.90 → red
    _gray_lp(r, 30)  # gri-bölge
    r.update(1, _roi(), _center_bbox(), FRAME_SHAPE)
    assert 1 not in r._pools or not r._pools[1].raw_reads


# --- gri-bölge: ikinci motor MUTABAKATI → oya GİRER ----------------------- #
def test_gray_zone_with_agreement_votes():
    ocr = FakeOCR(("34TC8532", 0.80))  # düşük conf ama mutabakat olacak
    r = _reader(ocr, require_engine_agreement=True, high_conf=0.99)
    r._er_agree_built = True
    r._er_agree_ocr = FakeOCR(("34TC8532", 0.85))  # AYNI format-geçerli plaka → mutabakat
    _gray_lp(r, 35)
    r.update(1, _roi(), _center_bbox(), FRAME_SHAPE)
    assert 1 in r._pools and r._pools[1].raw_reads
    text, w = r._pools[1].raw_reads[-1]
    assert text == "34TC8532"
    # weight_cap uygulanmış: eff = conf(0.80) * (size_w * weight_cap). size_w = 35/40=0.875.
    expected = 0.80 * (min(1.0, 35 / 40.0) * r._er_weight_cap)
    assert abs(w - expected) < 1e-6


def test_gray_zone_agreement_mismatch_no_vote():
    # İki motor FARKLI plaka okur → mutabakat YOK + conf<high_conf → oya girmez.
    ocr = FakeOCR(("34TC8532", 0.80))
    r = _reader(ocr, require_engine_agreement=True, high_conf=0.99)
    r._er_agree_built = True
    r._er_agree_ocr = FakeOCR(("34TD8532", 0.85))  # FARKLI → mutabakat yok
    _gray_lp(r, 35)
    r.update(1, _roi(), _center_bbox(), FRAME_SHAPE)
    assert 1 not in r._pools or not r._pools[1].raw_reads


# --- gri-bölge: mutabakat motoru yok ama conf >= high_conf → oya GİRER ---- #
def test_gray_zone_high_conf_path_votes():
    ocr = FakeOCR(("34TC8532", 0.97))  # conf >= high_conf(0.90)
    r = _reader(ocr, require_engine_agreement=True, high_conf=0.90)
    r._er_agree_built = True
    r._er_agree_ocr = None  # ikinci motor yok → high_conf eşiğine düşer
    _gray_lp(r, 40)
    r.update(1, _roi(), _center_bbox(), FRAME_SHAPE)
    assert 1 in r._pools and r._pools[1].raw_reads
    assert r._pools[1].raw_reads[-1][0] == "34TC8532"


# --- format-geçersiz gri-bölge okuma oya GİRMEZ (mutabakatlı olsa bile) --- #
def test_gray_zone_format_invalid_no_vote():
    ocr = FakeOCR(("XYZ", 0.99))  # format-geçersiz
    r = _reader(ocr, require_engine_agreement=True, high_conf=0.50)
    r._er_agree_built = True
    r._er_agree_ocr = FakeOCR(("XYZ", 0.99))
    _gray_lp(r, 35)
    r.update(1, _roi(), _center_bbox(), FRAME_SHAPE)
    assert 1 not in r._pools or not r._pools[1].raw_reads


# --- SR/upscale + füzyon crop'u BÜYÜTÜR ---------------------------------- #
def test_early_read_upscales_crop_before_ocr():
    # OCR'a giren görüntü ham gri-bölge kırpığından (lp_h=30) belirgin BÜYÜK olmalı.
    ocr = FakeOCR(("34TC8532", 0.97))
    r = _reader(ocr, require_engine_agreement=False, high_conf=0.90, fuse_frames=1)
    r._er_sr_scale = 3.0
    _gray_lp(r, 30)
    r.update(1, _roi(), _center_bbox(), FRAME_SHAPE)
    assert ocr.calls, "OCR çağrılmalı (gri-bölge yolu)"
    fed = ocr.calls[-1]
    # ham kırpık lp_h=30 (pad + <48 → _lp_crop 2x → ~60+); SR 3x daha büyütür → >> 60
    assert fed.shape[0] >= 90


def test_er_fuse_combines_multiple_crops():
    # _er_fuse: 2+ kırpık → ortak yüksekliğe hizalanmış median kompozit döner.
    ocr = FakeOCR(("34TC8532", 0.97))
    r = _reader(ocr)
    from collections import deque

    dq = deque(maxlen=5)
    dq.append(np.full((60, 120, 3), 50, np.uint8))
    dq.append(np.full((50, 100, 3), 200, np.uint8))
    fused = r._er_fuse(dq)
    assert fused is not None
    assert fused.shape[0] == 50 and fused.shape[1] == 100  # ortak (min) yükseklik/genişlik
    assert fused.dtype == np.uint8
    # median(50,200)=125 civarı
    assert 100 <= int(fused.mean()) <= 150


def test_er_fuse_single_crop_returns_none():
    r = _reader(FakeOCR())
    from collections import deque

    dq = deque(maxlen=5)
    dq.append(np.full((60, 120, 3), 50, np.uint8))
    assert r._er_fuse(dq) is None  # tek kare → füzyon yok


def test_er_upscale_identity_on_empty():
    r = _reader(FakeOCR())
    assert r._er_upscale(None) is None
    empty = np.zeros((0, 0, 3), np.uint8)
    assert r._er_upscale(empty) is empty


# --- PLAIN davranış (lp_h >= vote_min) DEĞİŞMEZ -------------------------- #
def test_plain_above_vote_min_unaffected():
    # lp_h >= lp_vote_min_px (45): NORMAL yol; early_read'e GİRMEZ, oy tam ağırlıkla yazılır.
    ocr = FakeOCR(("34TC8532", 1.0))
    r = _reader(ocr)
    r._er_agree_built = True
    r._er_agree_ocr = None
    _gray_lp(r, 50)  # 50 >= 45 → normal yol
    r.update(1, _roi(), _center_bbox(), FRAME_SHAPE)
    text, w = r._pools[1].raw_reads[-1]
    assert text == "34TC8532"
    # size_w = clamp(50/40)=1.0, weight_cap UYGULANMAZ (normal yol) → eff = 1.0*1.0
    assert abs(w - 1.0) < 1e-6
