"""Mock dedektör + IoU takipçi + ROI geometri (model gerektirmez, CI-uyumlu)."""

from __future__ import annotations

import cv2
import numpy as np

from aura.detection.detector import (
    StubDetector,
    build_detector,
    cap_roi_to_area,
    crop_person_roi,
    crop_rois,
)
from aura.detection.mock import MockDetector, SimpleIoUTracker, _iou
from aura.schema import BBox


def _frame_with_blocks() -> np.ndarray:
    f = np.full((360, 640, 3), 40, np.uint8)  # koyu asfalt
    cv2.rectangle(f, (100, 100), (180, 180), (120, 255, 120), -1)
    cv2.rectangle(f, (400, 150), (500, 260), (90, 200, 255), -1)
    return f


def test_mock_detector_finds_blocks(cfg):
    dets = MockDetector(cfg).detect(_frame_with_blocks())
    assert len(dets) == 2
    assert all(d.track_id is not None for d in dets)
    assert all(d.cabin_roi is not None and d.plate_roi is not None for d in dets)


def test_mock_tracker_stable_ids(cfg):
    det = MockDetector(cfg)
    f = _frame_with_blocks()
    a = det.detect(f)
    b = det.detect(f)
    assert sorted(d.track_id for d in a) == sorted(d.track_id for d in b)


def test_iou_basic():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_crop_rois_split():
    f = np.zeros((100, 100, 3), np.uint8)
    cabin, plate = crop_rois(f, BBox(x1=10, y1=10, x2=50, y2=50))
    assert cabin is not None and plate is not None
    assert cabin.shape[0] + plate.shape[0] == 40  # 55/45 yükseklik bölünmesi


# --- SORUN 3: devasa sürücü ROI sınırı (cap_roi_to_area, saf geometri) ---------
def test_cap_roi_oversized_is_cropped_to_ratio():
    # Devasa ROI (kabin fallback): kare alanının %50'si → %10'a kırpılmalı.
    f = np.zeros((400, 400, 3), np.uint8)  # kare alanı 160000
    box = (0, 0, 400, 200)  # alan 80000 = %50 (eşik %10'u aşar)
    capped = cap_roi_to_area(f, box, max_area_ratio=0.10, corner=(1.0, 1.0))
    assert capped is not None
    nx1, ny1, nx2, ny2 = capped
    new_area = (nx2 - nx1) * (ny2 - ny1)
    assert new_area <= 0.10 * 400 * 400 + 1  # %10 alana indi
    assert new_area > 0


def test_cap_roi_small_unchanged_returns_none():
    # Dar ROI (kişi-kutusu kesik) eşiğin altında → None (kırpma YOK, davranış değişmez).
    f = np.zeros((400, 400, 3), np.uint8)
    box = (0, 0, 100, 100)  # alan 10000 = %6.25 < %10
    assert cap_roi_to_area(f, box, max_area_ratio=0.10) is None


def test_cap_roi_disabled_returns_none():
    f = np.zeros((400, 400, 3), np.uint8)
    box = (0, 0, 400, 400)  # tüm kare
    assert cap_roi_to_area(f, box, max_area_ratio=0.0) is None  # 0 = kapalı


def test_cap_roi_anchors_to_driver_corner():
    # Sağ-alt köşe hedefi: kırpılmış kutu ROI'nin sağ-alt köşesine yaslanmalı.
    f = np.zeros((400, 400, 3), np.uint8)
    box = (0, 0, 400, 200)  # alan %50
    nx1, ny1, nx2, ny2 = cap_roi_to_area(f, box, 0.10, corner=(1.0, 1.0))
    assert nx2 == 400  # sağ kenara yaslı
    assert ny2 == 200  # alt kenara yaslı (ROI'nin alt sınırı)


def test_cap_roi_preserves_aspect_ratio():
    # En-boy oranı korunur (kare-kök ölçek).
    f = np.zeros((400, 400, 3), np.uint8)
    box = (0, 0, 400, 200)  # 2:1
    nx1, ny1, nx2, ny2 = cap_roi_to_area(f, box, 0.10)
    rw, rh = nx2 - nx1, ny2 - ny1
    assert abs((rw / rh) - 2.0) < 0.15  # ~2:1 korunur


def test_build_detector_falls_back_to_mock(cfg):
    # ai_mode=auto + ağırlık yok → MockDetector (ağırlık mevcut olsa da deterministik:
    # var olmayan yola işaret et ki auto-fallback dalı sınansın)
    cfg.data["runtime"]["ai_mode"] = "auto"
    cfg.data["models"]["detector"]["path"] = "weights/__nonexistent__.pt"
    assert isinstance(build_detector(cfg), MockDetector)


# --- BUG: paylaşılan mutable class-level last_* (regresyon) ---------------------
def test_detector_last_lists_are_instance_level():
    # İki StubDetector örneği AYNI listeyi paylaşmamalı (klasik mutable-default tuzağı).
    a, b = StubDetector(), StubDetector()
    assert a.last_persons is not b.last_persons
    assert a.last_signs is not b.last_signs
    assert a.last_aux is not b.last_aux
    a.last_persons.append("x")
    assert b.last_persons == []  # b etkilenmemeli


def test_mock_detector_sets_last_aux(cfg):
    # MockDetector taban siniftaki paylaşılan last_aux'a düşmemeli; kendi listesi olmalı.
    m1, m2 = MockDetector(cfg), MockDetector(cfg)
    assert m1.last_aux == [] and m1.last_aux is not m2.last_aux


# --- crop_rois dejenere kutu + sıfır-boyut crop -------------------------------
def test_crop_rois_degenerate_box_returns_none():
    f = np.zeros((100, 100, 3), np.uint8)
    # x2<=x1 → (None, None)
    assert crop_rois(f, BBox(x1=50, y1=10, x2=50, y2=40)) == (None, None)
    # kare dışı tamamen → kırpılınca x2<=x1
    assert crop_rois(f, BBox(x1=-20, y1=-20, x2=-5, y2=-5)) == (None, None)


def test_crop_rois_clamps_to_frame():
    f = np.zeros((100, 100, 3), np.uint8)
    cabin, plate = crop_rois(f, BBox(x1=-10, y1=-10, x2=200, y2=200))
    assert cabin is not None and plate is not None
    assert cabin.shape[1] == 100  # genişlik kare sınırına kırpıldı


# --- crop_person_roi ----------------------------------------------------------
def test_crop_person_roi_pads_and_clamps():
    f = np.zeros((100, 100, 3), np.uint8)
    roi = crop_person_roi(f, BBox(x1=40, y1=40, x2=60, y2=60), pad_ratio=0.5)
    assert roi is not None
    # 20px kutuya %50 pad = her yandan 10px → 40px genişlik
    assert roi.shape[0] == 40 and roi.shape[1] == 40


def test_crop_person_roi_degenerate_returns_none():
    f = np.zeros((100, 100, 3), np.uint8)
    assert crop_person_roi(f, BBox(x1=-50, y1=-50, x2=-40, y2=-40), pad_ratio=0.0) is None


# --- cap_roi_to_area kenar durumları ------------------------------------------
def test_cap_roi_zero_frame_area_returns_none():
    f = np.zeros((0, 0, 3), np.uint8)
    assert cap_roi_to_area(f, (0, 0, 10, 10), 0.1) is None


def test_cap_roi_degenerate_box_returns_none():
    f = np.zeros((400, 400, 3), np.uint8)
    # kutu kare dışına kırpılınca rw<=0
    assert cap_roi_to_area(f, (500, 500, 600, 600), 0.1) is None


def test_cap_roi_top_left_corner_anchor():
    # corner=(0,0) → sol-üst köşeye yaslı (DriverLock sol-sürücü sözleşmesi).
    f = np.zeros((400, 400, 3), np.uint8)
    nx1, ny1, nx2, ny2 = cap_roi_to_area(f, (0, 0, 400, 200), 0.10, corner=(0.0, 0.0))
    assert nx1 == 0 and ny1 == 0  # sol-üste yaslı


# --- SimpleIoUTracker: max_age dolma → track silme ----------------------------
def test_tracker_drops_track_after_max_age():
    tr = SimpleIoUTracker(iou_thr=0.3, max_age=2)
    ((tid, _),) = tr.update([(0, 0, 10, 10)])
    assert tid == 1 and len(tr.tracks) == 1
    # araç kareden çıkar → eşleşme yok, age artar; max_age aşılınca silinir
    for _ in range(3):
        tr.update([])
    assert 1 not in tr.tracks  # track silindi


def test_tracker_new_id_when_no_match():
    tr = SimpleIoUTracker(iou_thr=0.5, max_age=10)
    ((a, _),) = tr.update([(0, 0, 10, 10)])
    ((b, _),) = tr.update([(100, 100, 110, 110)])  # örtüşmeyen kutu → yeni ID
    assert a == 1 and b == 2


# --- mock sentetik sürücü / tabela dalları ------------------------------------
def test_mock_synthetic_person_emitted(cfg):
    cfg.data["driver_lock"]["mock_synthetic_person"] = True
    det = MockDetector(cfg)
    dets = det.detect(_frame_with_blocks())
    assert len(det.last_persons) == len(dets) == 2
    # sentetik sürücü ID'si araç ID'sine bağlı (100000 + vehicle_tid)
    veh_ids = {d.track_id for d in dets}
    assert all((p.track_id - 100000) in veh_ids for p in det.last_persons)


def test_mock_synthetic_sign_emitted(cfg):
    cfg.data["sign"]["mock_synthetic"] = True
    cfg.data["sign"]["mock_speed_limit"] = 70
    det = MockDetector(cfg)
    det.detect(_frame_with_blocks())
    assert len(det.last_signs) == 1
    assert det.last_signs[0].cls == "speed_limit_70"
