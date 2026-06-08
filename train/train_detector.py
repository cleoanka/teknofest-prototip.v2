"""Stage-1 araç tespit modeli eğitimi — YOLO26s fine-tune.

Araç sınıfları (car, truck, bus, minibus). ultralytics `model.train()`.
Çıktı `weights/custom_detector.pt` → config'te `models.detector.path` swap.
"""

from __future__ import annotations

import logging

from train.utils import export_best, resolve_device

log = logging.getLogger("aura.train.detector")


def train_detector(args) -> int:
    from ultralytics import YOLO  # lazy: --help torch gerektirmesin

    log.info("YOLO26s fine-tune: data=%s epochs=%d imgsz=%d", args.data, args.epochs, args.imgsz)
    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=resolve_device(args.device),
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    export_best(results, "custom_detector.pt")
    return 0
