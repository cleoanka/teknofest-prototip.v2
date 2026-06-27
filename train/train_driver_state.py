"""Stage-2 sürücü durumu modeli eğitimi — YOLO26 fine-tune.

Sınıflar: phone, smoking, no_seatbelt, fatigue (data.yaml'a göre). Cabin/sürücü ROI
küçük → imgsz 320 önerilir. Yorgunluk bir DETECTION sınıfıdır (kapalı göz/esneme/baş
düşmesi); MediaPipe/landmark KULLANILMAZ. Çıktı ``weights/custom_driver.pt`` (veya --out);
config'te ``models.driver_state.path`` + ``backend: yolo`` ile devreye alınır. docs/egitim.md.
"""

from __future__ import annotations

import logging

from train.utils import run_finetune

log = logging.getLogger("roadguard.train.driver_state")


def train_driver_state(args) -> int:
    return run_finetune(args, "custom_driver.pt", "YOLO26 sürücü-durum")
