"""Deterministik sürücü-durum mock'ı.

Cabin ROI'nin baskın rengini sentetik senaryo sürücü davranışına eşler. Ağırlık
olmadan anlamlı DRIVER_STATE event'leri + risk tetikleri üretir (gerçek YOLO26l
yerine demo/CI fallback'i).
"""

from __future__ import annotations

import numpy as np

from roadguard.driver_state.classifier import DriverClassifier
from roadguard.schema import DriverState

# Sentetik araç renkleri (BGR) → HAM sürücü-durum bayrakları (seatbelt = kemer VAR).
# no_seatbelt İHLALİ burada üretilmez; engine kemerin yokluğundan türetir:
#   araç 1: telefon ama KEMERLİ        → engine no_seatbelt üretmez
#   araç 2: sigara, KEMER YOK           → engine no_seatbelt türetir
#   araç 3: yorgun ama KEMERLİ          → engine no_seatbelt üretmez
_REFS: list[tuple[tuple[int, int, int], set[str]]] = [
    ((90, 200, 255), {"phone", "seatbelt"}),  # araç 1 (sarı/turuncu)
    ((120, 255, 120), {"smoking"}),  # araç 2 (yeşil) — kemer yok
    ((200, 150, 255), {"fatigue", "seatbelt"}),  # araç 3 (pembe)
]


class MockDriverClassifier(DriverClassifier):
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_dist = 160.0  # bundan uzaksa "durum yok" (arka plan)

    def infer(self, cabin_roi: np.ndarray | None, track_id: int | None = None) -> DriverState:
        # track_id ABC sözleşmesi gereği kabul edilir; mock durumsuz → yok sayılır.
        # (Liskov: tüm backend'ler aynı imzayı taşır; engine._infer TypeError
        # maskelemesine GÜVENMEZ — gerçek bir TypeError artık yutulmaz.)
        del track_id
        if cabin_roi is None or cabin_roi.size == 0:
            return DriverState()
        mean = cabin_roi.reshape(-1, cabin_roi.shape[-1])[:, :3].mean(axis=0)
        best_flags: set[str] | None = None
        best_d = 1e9
        for color, flags in _REFS:
            d = float(np.linalg.norm(mean - np.array(color, dtype=float)))
            if d < best_d:
                best_d, best_flags = d, flags
        if best_flags is None or best_d > self.max_dist:
            return DriverState()
        conf = round(max(0.5, 1.0 - best_d / 300.0), 2)
        ds = DriverState()
        for f in best_flags:
            setattr(ds, f, True)
            ds.confidence[f] = conf
        return ds
