"""Pipeline orkestratörü.

Akış (plan.md §6.9):
  preprocessing → detection+track → ROI → stability ⊗ (driver_state ∥ plate)
                → speed → accumulator → events + annotations

İki-kanal çıktı: `AnnotationFrame` (kare başına bbox, dashboard canvas için) ve
`AuraEvent` (durum değişimleri). Pipeline upstream/downstream'i bilmez.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

from aura.accumulator.accumulator import Accumulator
from aura.detection.detector import build_detector, crop_rois
from aura.driver_state.classifier import DriverStateClassifier
from aura.events.emitter import EventEmitter
from aura.plate.reader import PlateReader
from aura.preprocessing.preprocess import Preprocessor
from aura.schema import AnnotationFrame, AuraEvent, TrackRecord
from aura.speed.estimator import SpeedEstimator
from aura.stability.state_machine import StabilityTracker

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("aura.pipeline")

_DRIVER_FIELDS = ("phone", "smoking", "no_seatbelt", "fatigue")


def record_to_annotation(rec: TrackRecord) -> dict:
    """TrackRecord → dashboard canvas için annotation sözlüğü."""
    return {
        "track_id": rec.track_id,
        "bbox": [rec.bbox.x1, rec.bbox.y1, rec.bbox.x2, rec.bbox.y2],
        "cls": rec.vehicle_class,
        "conf": rec.bbox.conf,
        "plate": rec.plate.value,
        "plate_status": rec.plate.status,
        "driver": rec.driver.active_flags(),
        "speed_kmh": rec.speed.value_kmh,
        "relative_velocity_flag": rec.speed.relative_velocity_flag,
        "risk_flags": rec.risk_flags,
        "qod_active": rec.qod_active,
    }


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pre = Preprocessor(cfg)
        self.detector = build_detector(cfg)
        self.stability = StabilityTracker(cfg)
        self.driver = DriverStateClassifier(cfg)
        self.plate = PlateReader(cfg)
        self.speed = SpeedEstimator(cfg)
        self.acc = Accumulator(cfg)
        self.emitter = EventEmitter()
        self.frame_idx = 0

    # --- tek kare ---------------------------------------------------------- #
    def process_frame(self, frame: "np.ndarray", frame_idx: int | None = None
                      ) -> tuple[AnnotationFrame, list[AuraEvent]]:
        idx = self.frame_idx if frame_idx is None else frame_idx
        frame = self.pre.process(frame)
        detections = self.detector.detect(frame)

        events: list[AuraEvent] = []
        track_dicts: list[dict] = []

        for det in detections:
            tid = det.track_id if det.track_id is not None else -1
            cabin, plate_roi = crop_rois(frame, det.bbox)

            # Stage-2 sürücü durumu → 16/8 kararlılık süzgeci (alan-bazında)
            driver = self.driver.infer(cabin)
            for f in _DRIVER_FIELDS:
                stable = self.stability.update(
                    f"{tid}:driver.{f}", getattr(driver, f), driver.confidence.get(f, 1.0)
                )
                setattr(driver, f, bool(stable))

            plate = self.plate.update(tid, plate_roi, det.bbox, frame.shape, frame=frame)
            speed = self.speed.update(tid, det.bbox, idx)

            rec, ev = self.acc.update_track(
                tid, frame_idx=idx, bbox=det.bbox, vehicle_class=det.bbox.cls,
                plate=plate, driver=driver, speed=speed,
            )
            events.extend(ev)
            track_dicts.append(record_to_annotation(rec))

        anno = AnnotationFrame(frame_id=idx, tracks=track_dicts)
        for e in events:
            self.emitter.emit_event(e)
        self.emitter.emit_annotation(anno)
        self.frame_idx = idx + 1
        return anno, events

    # --- video / kamera ---------------------------------------------------- #
    def frames(self, source, max_frames: int | None = None
               ) -> Iterator[tuple["np.ndarray", AnnotationFrame, list[AuraEvent]]]:
        """Kaynağı aç ve (frame, annotation, events) üret. Kaynak: path | index | URL."""
        import cv2

        src = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Kaynak açılamadı: {source}")
        i = 0
        try:
            while True:
                if max_frames is not None and i >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                anno, events = self.process_frame(frame, i)
                yield frame, anno, events
                i += 1
        finally:
            cap.release()

    def run_video(self, source, max_frames: int | None = None) -> list[AuraEvent]:
        """Tüm kaynağı işle, üretilen tüm event'leri döndür (offline/eval kullanımı)."""
        all_events: list[AuraEvent] = []
        for _frame, _anno, events in self.frames(source, max_frames):
            all_events.extend(events)
        return all_events

    def close(self) -> None:
        self.detector.close()
