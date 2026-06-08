"""Hız: disabled relative_velocity_flag (hızlı/yavaş) + tripwire (model gerektirmez)."""

from __future__ import annotations

from aura.config import load_config
from aura.schema import BBox
from aura.speed.estimator import SpeedEstimator

FRAME = (360, 640, 3)


def _bbox_at(cy_norm: float) -> BBox:
    y = cy_norm * 360
    return BBox(x1=300, y1=y - 20, x2=340, y2=y + 20, conf=0.9, cls="car")


def test_disabled_no_value(cfg):
    cfg.data["speed"]["mode"] = "disabled"
    s = SpeedEstimator(cfg).update(1, _bbox_at(0.3), 0, FRAME)
    assert s.mode == "disabled" and s.value_kmh is None


def test_relative_flag_fast(cfg):
    est = SpeedEstimator(cfg)
    est.fps = 30
    s = None
    for i in range(8):
        s = est.update(1, _bbox_at(0.10 + 0.05 * i), i, FRAME)  # 0.05/kare ≫ 0.012
    assert s.relative_velocity_flag is True


def test_relative_flag_slow(cfg):
    est = SpeedEstimator(cfg)
    est.fps = 30
    s = None
    for i in range(8):
        s = est.update(1, _bbox_at(0.10 + 0.005 * i), i, FRAME)  # 0.005/kare < 0.012
    assert s.relative_velocity_flag is False


def test_tripwire_produces_speed_and_latches():
    cfg = load_config()
    cfg.data["speed"]["mode"] = "tripwire"
    est = SpeedEstimator(cfg)
    est.fps = 30
    val = None
    for i in range(21):
        cy = 0.35 + 0.02 * i  # line_a(0.40) ve line_b(0.70) geçişi
        s = est.update(1, _bbox_at(min(cy, 0.99)), i, FRAME)
        val = s.value_kmh
        assert s.mode == "tripwire"
    assert val is not None and val > 0  # hız hesaplandı
    # latched: sonraki kare aynı değeri döndürür
    again = est.update(1, _bbox_at(0.99), 30, FRAME).value_kmh
    assert again == val
