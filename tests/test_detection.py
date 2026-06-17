"""Mock dedektör + IoU takipçi + ROI geometri (model gerektirmez, CI-uyumlu)."""

from __future__ import annotations

import cv2
import numpy as np

from aura.detection.detector import build_detector, cap_roi_to_area, crop_rois
from aura.detection.mock import MockDetector, _iou
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
