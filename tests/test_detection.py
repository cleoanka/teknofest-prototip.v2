"""Mock dedektör + IoU takipçi + ROI geometri (model gerektirmez, CI-uyumlu)."""

from __future__ import annotations

import cv2
import numpy as np

from aura.detection.detector import build_detector, crop_rois
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


def test_build_detector_falls_back_to_mock(cfg):
    # ai_mode=auto + ağırlık yok → MockDetector
    assert isinstance(build_detector(cfg), MockDetector)
