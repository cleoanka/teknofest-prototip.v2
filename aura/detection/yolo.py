"""Gerçek Stage-1 dedektör — ultralytics YOLO26 + ByteTrack.

`ai_mode=real` (veya `auto` + ağırlık mevcut) iken kullanılır. ByteTrack tracking
mode ultralytics'e dahildir (`tracker="bytetrack.yaml"`). Yalnızca config'teki
araç sınıfları geçirilir; her tespit için ROI crop'lar üretilir.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aura.detection.detector import Detection, Detector, Person, Sign, crop_rois
from aura.device import resolve_device
from aura.schema import BBox

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("aura.detection.yolo")


class YOLO26Detector(Detector):
    def __init__(self, cfg):
        from ultralytics import YOLO

        self.path = cfg.get("models.detector.path", "weights/yolo26s.pt")
        self.model = YOLO(self.path)
        self.conf = float(cfg.get("models.detector.conf", 0.35))
        self.iou = float(cfg.get("models.detector.iou", 0.45))
        self.imgsz = int(cfg.get("models.detector.imgsz", 640))
        self.tracker = str(cfg.get("tracking.tracker", "bytetrack"))
        vc = cfg.get("models.detector.vehicle_classes", [])
        self.vehicle_classes = set(vc) if vc else set()
        # Sürücü kilidi için kişi sınıfları (aynı ByteTrack geçişinde toplanır)
        pc = cfg.get("driver_lock.person_classes", ["person"])
        self.person_classes = set(pc) if pc else set()
        self.last_persons: list[Person] = []
        # Trafik tabelası sınıfları (araç/kişi DIŞI; SignTracker tüketir). value_map
        # anahtarları da tabela sayılır → config'te classes eksik kalsa bile yakalanır.
        self.sign_enabled = bool(cfg.get("sign.enabled", True))
        sc = cfg.get("sign.classes", []) or []
        vmap = cfg.get("sign.value_map", {}) or {}
        self.sign_classes = set(sc) | {str(k) for k in vmap}
        self.last_signs: list[Sign] = []
        self.device = resolve_device(cfg.get("runtime.device", "auto"))
        log.info(
            "YOLO26 yüklendi: %s (imgsz=%d, tracker=%s, device=%s)",
            self.path,
            self.imgsz,
            self.tracker,
            self.device,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=f"{self.tracker}.yaml",
            device=self.device,
            verbose=False,
        )
        dets: list[Detection] = []
        self.last_persons = []
        self.last_signs = []
        if not results:
            return dets
        r = results[0]
        names = getattr(r, "names", None) or self.model.names
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return dets
        for b in boxes:
            cls_idx = int(b.cls.item())
            cls_name = (
                names[cls_idx]
                if isinstance(names, (list, tuple))
                else names.get(cls_idx, str(cls_idx))
            )
            is_person = cls_name in self.person_classes
            is_sign = self.sign_enabled and cls_name in self.sign_classes
            # vehicle_classes boşsa "süzgeç yok" demektir → kişi/tabela dışı her şey araç.
            is_vehicle = (
                cls_name in self.vehicle_classes
                if self.vehicle_classes
                else not (is_person or is_sign)
            )
            if not (is_vehicle or is_person or is_sign):
                continue
            xyxy = b.xyxy[0].tolist()
            tid = int(b.id.item()) if getattr(b, "id", None) is not None else None
            bbox = BBox(
                x1=xyxy[0],
                y1=xyxy[1],
                x2=xyxy[2],
                y2=xyxy[3],
                conf=float(b.conf.item()),
                cls=cls_name,
            )
            # Tabela → SignTracker'a (sahne-seviyesi; ID-merkezli akışa girmez)
            if is_sign:
                self.last_signs.append(Sign(bbox=bbox, cls=cls_name, track_id=tid))
                continue
            # Kişi sınıfı → sürücü kilidine; (araç sınıfıyla çakışmaz, COCO'da ayrık)
            if is_person:
                self.last_persons.append(Person(bbox=bbox, track_id=tid))
                continue
            d = Detection(bbox=bbox, track_id=tid)
            d.cabin_roi, d.plate_roi = crop_rois(frame, bbox)
            dets.append(d)
        return dets
