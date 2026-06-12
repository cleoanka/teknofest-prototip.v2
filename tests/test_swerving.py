"""Swerving (dikkatsiz sürüş / yalpalama) tespiti — yanal yörünge analizi.

Algoritma fps-bağımsız (pencere saniye) ve ölçek-bağımsızdır (genlik araç
genişliği biriminde): cx serisinden doğrusal trend çıkarılır, kalan sinyalde
histerezisli salınım sayılır. Testler 100px genişlikli sentetik araçla koşar
(SpeedEstimator varsayılan fps=30 → 3.0s pencere = 90 kare).
"""

from __future__ import annotations

import math

from aura.schema import BBox
from aura.speed.estimator import SpeedEstimator


def _bbox(cx: float, w: float = 100.0) -> BBox:
    return BBox(x1=cx - w / 2, y1=400, x2=cx + w / 2, y2=500, conf=0.9, cls="car")


def _estimator(cfg) -> SpeedEstimator:
    cfg.data.setdefault("speed", {})["mode"] = "disabled"
    return SpeedEstimator(cfg)


FRAME = (1080, 1920, 3)


def test_straight_driving_no_swerving(cfg):
    est = _estimator(cfg)
    states = [est.update(1, _bbox(500 + i * 3), i, FRAME) for i in range(90)]
    assert not any(s.swerving for s in states)


def test_weaving_sets_swerving(cfg):
    est = _estimator(cfg)
    # Yaklaşma kayması (i*2) ÜZERİNE 40px genlikli, 30-kare periyotlu yalpalama
    states = []
    for i in range(90):
        cx = 500 + i * 2 + 40.0 * math.sin(2 * math.pi * i / 30)
        states.append(est.update(2, _bbox(cx), i, FRAME))
    assert any(s.swerving for s in states)


def test_single_lane_change_is_not_swerving(cfg):
    est = _estimator(cfg)
    # Tek yönlü şerit değişimi: trend'e gider, salınım değildir → bayrak YOK
    path = [500.0] * 30 + [500.0 + min((i + 1) * 6.0, 180.0) for i in range(60)]
    states = [est.update(3, _bbox(cx), i, FRAME) for i, cx in enumerate(path)]
    assert not any(s.swerving for s in states)


def test_small_jitter_no_swerving(cfg):
    est = _estimator(cfg)
    # ±3px bbox titremesi (gürültü) — amp_ratio×genişlik (20px) altında kalır
    states = [est.update(4, _bbox(500 + (3.0 if i % 2 else -3.0)), i, FRAME) for i in range(90)]
    assert not any(s.swerving for s in states)
