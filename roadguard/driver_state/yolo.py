"""Gerçek Stage-2 sürücü-durum sınıflandırıcı — ultralytics YOLO26l.

Cabin ROI üzerinde çoklu-etiket detection (phone/smoking/no_seatbelt/fatigue).
Aynı anda birden çok sınıf aktif olabilir (classification değil, detection).
MediaPipe/landmark kullanılmaz.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roadguard.config import resolve_repo_path
from roadguard.device import resolve_device
from roadguard.driver_state.classifier import DriverClassifier
from roadguard.schema import DriverState
from roadguard.taxonomy import canonical

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("roadguard.driver_state.yolo")


class YOLO26lDriverClassifier(DriverClassifier):
    def __init__(self, cfg):
        from ultralytics import YOLO

        self.path = str(
            resolve_repo_path(cfg.get("models.driver_state.path", "weights/yolo26l.pt"))
        )
        self.model = YOLO(self.path)
        self.conf = float(cfg.get("models.driver_state.conf", 0.40))
        self.imgsz = int(cfg.get("models.driver_state.imgsz", 320))
        self.classes = list(
            cfg.get("models.driver_state.classes", ["phone", "smoking", "seatbelt", "fatigue"])
        )
        self.device = resolve_device(cfg.get("runtime.device", "auto"))
        log.info("YOLO26l yüklendi: %s (imgsz=%d, device=%s)", self.path, self.imgsz, self.device)
        # Sınıf-kapsama doğrulaması (Codex): backend=yolo iken STOK COCO yolo26l verilirse
        # smoking/seatbelt/fatigue ÜRETİLEMEZ (yalnız phone ↔ COCO 'cell phone'). Sistem
        # eksik davranışı sessizce algılamasın diye startup'ta uyarır.
        try:
            model_canon = {canonical(str(n)) for n in self.model.names.values()}
            missing = [c for c in self.classes if c not in model_canon]
            if missing:
                producible = sorted(set(self.classes) & model_canon)
                log.warning(
                    "driver_state modeli (%s) şu davranış sınıflarını ÜRETEMEZ: %s — yalnız %s "
                    "üretilir. Tüm sınıflar için fine-tune ağırlık ya da pose backend gerekir.",
                    self.path,
                    missing,
                    producible or "(yok)",
                )
        except Exception:  # noqa: BLE001 — doğrulama best-effort, akışı bozmamalı
            pass

    def infer(self, cabin_roi: np.ndarray | None, track_id: int | None = None) -> DriverState:
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
            # Kanonik ad eşlemesi: stok COCO 'cell phone' → 'phone' vb. — model
            # değişse de config/şema sözleşmesi aynı kalır (roadguard/taxonomy.py).
            name = canonical(name)
            if name in self.classes and hasattr(ds, name):
                setattr(ds, name, True)
                ds.confidence[name] = max(ds.confidence.get(name, 0.0), float(b.conf.item()))
        return ds
