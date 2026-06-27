"""detection — Stage-1 YOLO26s tespit + ByteTrack + ROI crop."""

from roadguard.detection.detector import (
    Detection,
    Detector,
    StubDetector,
    build_detector,
    crop_rois,
)

__all__ = ["Detection", "Detector", "StubDetector", "build_detector", "crop_rois"]
