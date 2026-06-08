"""Pipeline orkestratörü.

Akış (plan.md §6.9):
  preprocessing → detection+track → ROI → stability ⊗ (driver_state ∥ plate)
                → speed → accumulator → events + annotations

İki-kanal çıktı: `AnnotationFrame` (kare başına bbox, dashboard canvas için) ve
`AuraEvent` (durum değişimleri). Pipeline upstream/downstream'i bilmez.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from aura.accumulator.accumulator import Accumulator
from aura.detection.detector import build_detector, crop_rois
from aura.driver_state.classifier import build_driver_classifier
from aura.events.emitter import EventEmitter
from aura.optional.loader import get_optional
from aura.plate.reader import PlateReader
from aura.preprocessing.preprocess import Preprocessor
from aura.qod.client import QoDController
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
        self.driver = build_driver_classifier(cfg)
        self.qod = QoDController(cfg)
        self.plate = PlateReader(cfg, qod=self.qod)
        self.speed = SpeedEstimator(cfg)
        self.high_speed = float(cfg.get("risk.high_speed_kmh", 90))
        self.acc = Accumulator(cfg)
        self.emitter = EventEmitter()
        self.frame_idx = 0
        self.fps = 30.0
        # §8 opsiyonel: kapalıysa None döner, import bile yapılmaz (lazy)
        self.zwp = get_optional(cfg, "zero_waste_payload")

    # --- tek kare ---------------------------------------------------------- #
    def process_frame(
        self, frame: np.ndarray, frame_idx: int | None = None
    ) -> tuple[AnnotationFrame, list[AuraEvent]]:
        idx = self.frame_idx if frame_idx is None else frame_idx
        self.qod.set_now(idx / max(self.fps, 1e-6))
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
            speed = self.speed.update(tid, det.bbox, idx, frame.shape)
            # göreli hız bayrağını da 16/8 süzgecinden geçir (eşik civarı salınımı önle)
            speed.relative_velocity_flag = bool(
                self.stability.update(f"{tid}:speed.rel", speed.relative_velocity_flag)
            )
            if speed.relative_velocity_flag or (
                speed.value_kmh is not None and speed.value_kmh >= self.high_speed
            ):
                self.qod.request_optimize(tid, "speed_anomaly")

            qod_active, qod_profile = self.qod.state(tid)
            rec, ev = self.acc.update_track(
                tid,
                frame_idx=idx,
                bbox=det.bbox,
                vehicle_class=det.bbox.cls,
                plate=plate,
                driver=driver,
                speed=speed,
                qod_active=qod_active,
                qod_profile=qod_profile,
            )
            events.extend(ev)
            adict = record_to_annotation(rec)
            if self.zwp is not None:  # §8.1 sıfır-atık payload
                adict["zwp"] = self.zwp.build_payload(adict, plate_roi)
            track_dicts.append(adict)

        self.qod.tick()
        events.extend(self.qod.drain_events())

        anno = AnnotationFrame(frame_id=idx, tracks=track_dicts)
        for e in events:
            self.emitter.emit_event(e)
        self.emitter.emit_annotation(anno)
        self.frame_idx = idx + 1
        return anno, events

    # --- video / kamera ---------------------------------------------------- #
    def frames(
        self, source, max_frames: int | None = None
    ) -> Iterator[tuple[np.ndarray, AnnotationFrame, list[AuraEvent]]]:
        """Kaynağı aç ve (frame, annotation, events) üret. Kaynak: path | index | URL."""
        import cv2

        src = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Kaynak açılamadı: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        self.speed.fps = self.fps
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
