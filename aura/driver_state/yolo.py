"""Gerçek Stage-2 sürücü-durum sınıflandırıcı — ultralytics YOLO26l.

Cabin ROI üzerinde çoklu-etiket detection (phone/smoking/no_seatbelt/fatigue).
Aynı anda birden çok sınıf aktif olabilir (classification değil, detection).
MediaPipe/landmark kullanılmaz.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aura.device import resolve_device
from aura.driver_state.classifier import DriverClassifier
from aura.schema import DriverState

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("aura.driver_state.yolo")


class YOLO26lDriverClassifier(DriverClassifier):
    def __init__(self, cfg):
        from ultralytics import YOLO

        self.path = cfg.get("models.driver_state.path", "weights/yolo26l.pt")
        self.model = YOLO(self.path)
        self.conf = float(cfg.get("models.driver_state.conf", 0.40))
        self.imgsz = int(cfg.get("models.driver_state.imgsz", 320))
        self.classes = list(
            cfg.get("models.driver_state.classes", ["phone", "smoking", "no_seatbelt", "fatigue"])
        )
        self.device = resolve_device(cfg.get("runtime.device", "auto"))
        log.info("YOLO26l yüklendi: %s (imgsz=%d, device=%s)", self.path, self.imgsz, self.device)

    def infer(self, cabin_roi: np.ndarray | None) -> DriverState:
        ds = DriverState()
        if cabin_roi is None or cabin_roi.size == 0:
            return ds
        results = self.model.predict(
            cabin_roi, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )
        if not results:
            return ds
        r = results[0]
        names = getattr(r, "names", None) or self.model.names
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return ds
        for b in boxes:
            cls_idx = int(b.cls.item())
            name = (
                names[cls_idx]
                if isinstance(names, (list, tuple))
                else names.get(cls_idx, str(cls_idx))
            )
            if name in self.classes and hasattr(ds, name):
                setattr(ds, name, True)
                ds.confidence[name] = max(ds.confidence.get(name, 0.0), float(b.conf.item()))
        return ds
