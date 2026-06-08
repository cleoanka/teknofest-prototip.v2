"""§8 opsiyonel modüller — lazy import (kapalıyken import yok) + işlevsellik."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from aura.config import load_config
from aura.optional.loader import _reset_cache, get_optional, is_enabled
from aura.pipeline import Pipeline
from aura.schema import BBox

SAMPLE = "data/samples/ornek.mp4"
_MODULES = ("zero_waste_payload", "super_resolution", "homography_ipm")


def _ensure_sample():
    if not Path(SAMPLE).exists():
        from aura.synthetic import generate

        generate(Path("data/samples"), 90, 30, 640, 360)


def _purge():
    _reset_cache()
    for m in _MODULES:
        sys.modules.pop(f"aura.optional.{m}", None)


def test_disabled_modules_not_imported(cfg):
    _ensure_sample()
    _purge()
    assert not is_enabled(cfg, "zero_waste_payload")
    Pipeline(cfg).run_video(SAMPLE, max_frames=20)
    for m in ("zero_waste_payload", "super_resolution"):
        assert f"aura.optional.{m}" not in sys.modules  # kapalıyken import YOK


def test_get_optional_none_when_off(cfg):
    _purge()
    assert get_optional(cfg, "super_resolution") is None
    assert "aura.optional.super_resolution" not in sys.modules


def test_zero_waste_enabled_imports_and_attaches(cfg):
    _ensure_sample()
    _purge()
    cfg.data["optional_modules"]["zero_waste_payload"] = True
    cfg.data.setdefault("runtime", {})["ai_mode"] = "mock"  # sentetik karede deterministik tespit
    pipe = Pipeline(cfg)
    pipe.run_video(SAMPLE, max_frames=40)
    assert "aura.optional.zero_waste_payload" in sys.modules
    anno = pipe.emitter.latest_annotation()
    assert anno and anno.tracks and "zwp" in anno.tracks[0]
    assert "structured" in anno.tracks[0]["zwp"]


def test_super_resolution_enhances(cfg):
    _purge()
    cfg.data["optional_modules"]["super_resolution"] = True
    sr = get_optional(cfg, "super_resolution")
    assert sr is not None
    out = sr.enhance(np.zeros((10, 20, 3), np.uint8), scale=2)
    assert out.shape[0] == 20 and out.shape[1] == 40


def test_homography_ipm_speed():
    cfg = load_config()
    cfg.data["optional_modules"]["homography_ipm"] = True
    cfg.data["speed"]["calibration_file"] = "config/calibration/ornek_kamera.yaml"
    from aura.optional.homography_ipm import _state, ipm_speed

    _state.clear()
    fs = (360, 640, 3)
    v1 = ipm_speed(cfg, 1, BBox(x1=300, y1=200, x2=340, y2=250), 0, 30, fs)
    v2 = ipm_speed(cfg, 1, BBox(x1=300, y1=230, x2=340, y2=300), 1, 30, fs)
    assert v1 is None  # ilk kare → referans yok
    assert v2 is not None and v2 >= 0  # IPM ile hız hesaplandı
