"""Stage-2 sürücü durumu modeli eğitimi — YOLO26l fine-tune.

Sınıflar: phone, smoking, no_seatbelt, fatigue. 320px imgsz (cabin ROI küçük).
Yorgunluk bir detection sınıfıdır — MediaPipe/landmark KULLANILMAZ.
Çıktı `weights/custom_driver.pt` → config'te `models.driver_state.path` swap.
"""

from __future__ import annotations

import logging

from train.utils import export_best, resolve_device

log = logging.getLogger("aura.train.driver_state")


def train_driver_state(args) -> int:
    from ultralytics import YOLO  # lazy

    log.info("YOLO26l fine-tune: data=%s epochs=%d imgsz=%d", args.data, args.epochs, args.imgsz)
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
    export_best(results, "custom_driver.pt")
    return 0
