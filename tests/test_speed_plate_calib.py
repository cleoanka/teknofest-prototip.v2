"""Plaka→hız oto-kalibrasyon bağlantısı (LP dedektörü kutusu → observe_plate).

İki kanıt:
  1) SpeedEstimator.update(plate_bbox=...) → ScaleField'e ağırlık 1.0 plaka örneği
     besler (araç-genişliği 0.25'e baskın); plaka olmadan yalnız araç örneği gelir.
  2) PlateReader, LP dedektörünün plate_roi-yerel kutusunu FRAME koordinatına doğru
     çevirir (offset = araç bbox + crop yüksekliğinden; cabin_ratio gerekmez).
"""

from __future__ import annotations

import numpy as np

from aura.config import load_config
from aura.plate.reader import PlateReader
from aura.schema import BBox

FRAME = (720, 1280, 3)
_ASPECT = 520.0 / 120.0  # TR Tip-1 plaka en/boy ≈ 4.33


def _metric_cfg():
    cfg = load_config()
    cfg.data["speed"]["mode"] = "metric"
    return cfg


def _car(x_left: float, y2: float = 400.0, width: float = 180.0, height: float = 130.0) -> BBox:
    return BBox(x1=x_left, y1=y2 - height, x2=x_left + width, y2=y2, conf=0.9, cls="car")


def _plate(ppm: float, y2: float = 400.0, x_left: float = 120.0) -> BBox:
    """Cepheden (aspect 4.33) plaka: genişlik = ppm × 0.520 m → bilinen ppm üretir."""
    w = ppm * 0.520
    h = w / _ASPECT
    return BBox(x1=x_left, y1=y2 - h, x2=x_left + w, y2=y2, conf=1.0, cls="plate")


# --- 1) estimator bağlantısı -------------------------------------------------- #
def test_plate_bbox_adds_high_weight_sample():
    """plate_bbox verilince her karede plaka + araç örneği eklenir (2×); verilmezse 1×."""
    est = _build()
    est_plate = _build()
    plate = _plate(ppm=100.0)
    for i in range(8):
        est.update(1, _car(100.0 + 50.0 * i), i, FRAME)  # plakasız
        est_plate.update(1, _car(100.0 + 50.0 * i), i, FRAME, plate_bbox=plate)
    assert est._metric.scale.n_samples == 8  # yalnız araç
    assert est_plate._metric.scale.n_samples == 16  # araç + plaka


def test_plate_ppm_dominates_vehicle_estimate():
    """Plaka (ağırlık 1.0) ve araç (0.25) farklı ppm verince, füzyon plakaya yakın çıkar."""
    est = _build()
    # araç 180px / 1.80m = 100 ppm; plaka 130 ppm — ikisi de y2=400'de (temiz ağırlıklı ort.)
    plate = _plate(ppm=130.0, y2=400.0)
    for i in range(8):
        est.update(1, _car(100.0 + 50.0 * i, y2=400.0), i, FRAME, plate_bbox=plate)
    sf = est._metric.scale
    assert sf.is_ready
    # ağırlıklı ortalama: (8·0.25·100 + 8·1.0·130)/(8·0.25 + 8·1.0) = 124 → plakaya yakın
    assert abs(sf.ppm_at(400.0) - 124.0) < 1.0


def test_no_plate_unaffected():
    """plate_bbox=None (varsayılan) eski davranışı bozmaz — kalibrasyon yine oturur."""
    est = _build()
    val = None
    for i in range(16):
        val = est.update(1, _car(100.0 + 50.0 * i), i, FRAME).value_kmh
    assert val is not None  # araç-genişliği yedeğiyle km/h üretildi


def _build():
    from aura.speed.estimator import SpeedEstimator as _SE

    est = _SE(_metric_cfg())
    est.fps = 25.0
    return est


# --- 2) reader koordinat eşlemesi -------------------------------------------- #
def test_reader_maps_lp_box_to_frame_coords(monkeypatch):
    """LP kutusu plate_roi-yerel; reader frame koordinatına çevirip last_plate_bbox'a yazar."""
    cfg = load_config()  # super_resolution: false → self.sr is None
    reader = PlateReader(cfg)  # ocr=None → LP yolu etkin
    reader.ocr = _FakeOCR()  # ağır OCR motorunu baypas et

    # plate_roi = frame[split:y2, x1:x2] doğrudan dilim; LP yerel kutusu:
    local_box = (10, 5, 140, 35)  # 130px geniş × 30px → aspect 4.33, lp_h=30 (gate'leri geçer)
    monkeypatch.setattr(
        reader, "_lp_crop", lambda roi: (np.zeros((30, 130, 3), np.uint8), 30, local_box)
    )

    plate_roi = np.zeros((40, 200, 3), np.uint8)  # yükseklik 40 (≥ min_pixel_height)
    vehicle = BBox(x1=500, y1=300, x2=800, y2=500, conf=0.9, cls="car")  # sweet-spot içi
    reader.update(1, plate_roi, vehicle, FRAME)

    bb = reader.last_plate_bbox
    assert bb is not None
    # offset_x = max(0,500)=500 ; offset_y = min(720,500) - 40 = 460
    assert (bb.x1, bb.y1, bb.x2, bb.y2) == (510.0, 465.0, 640.0, 495.0)
    assert bb.cls == "plate"


def test_reader_resets_bbox_each_call(monkeypatch):
    """Plaka bulunamayan karede last_plate_bbox None'a döner (önceki track'ten sızmaz)."""
    cfg = load_config()
    reader = PlateReader(cfg)
    reader.ocr = _FakeOCR()
    monkeypatch.setattr(reader, "_lp_crop", lambda roi: (roi, None, None))  # LP tespit yok
    plate_roi = np.zeros((40, 200, 3), np.uint8)
    vehicle = BBox(x1=500, y1=300, x2=800, y2=500, conf=0.9, cls="car")  # sweet-spot içi
    reader.update(1, plate_roi, vehicle, FRAME)
    assert reader.last_plate_bbox is None


class _FakeOCR:
    def read(self, plate_roi, vehicle_crop=None):
        return "34ABC123", 0.9
