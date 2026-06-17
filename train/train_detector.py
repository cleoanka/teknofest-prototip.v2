"""Stage-1 araç tespit modeli eğitimi — YOLO26 fine-tune.

Araç + plaka + araç-içi sınıflar (data.yaml'a göre). ultralytics ``model.train`` →
``model.val`` → metrik + best export. Çıktı ``weights/custom_detector.pt`` (veya --out);
config'te ``models.detector.path`` bununla değiştirilir. Detay: docs/egitim.md.
"""

from __future__ import annotations

import logging

from train.utils import run_finetune

log = logging.getLogger("aura.train.detector")


def train_detector(args) -> int:
    return run_finetune(args, "custom_detector.pt", "YOLO26 dedektör")
