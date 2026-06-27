"""roadguard/preprocessing/preprocess.py — config bayrak okuma + pass-through (M2).

Filtre implementasyonları sonraki iterasyonda gelecek; bu test arayüzün sabit
kaldığını ve flag'lerin config'ten doğru okunduğunu çitler (downstream etkilenmez).
"""

from __future__ import annotations

import numpy as np

from roadguard.config import Config
from roadguard.preprocessing.preprocess import Preprocessor


def test_flags_default_false():
    pre = Preprocessor(Config({}))
    assert pre.headlight is False
    assert pre.motion_blur is False
    assert pre.reflection is False
    assert pre.occlusion is False


def test_flags_read_from_config():
    cfg = Config(
        {
            "preprocessing": {
                "headlight_suppression": True,
                "motion_blur_correction": True,
                "reflection_suppression": False,
                "occlusion_handling": True,
            }
        }
    )
    pre = Preprocessor(cfg)
    assert pre.headlight is True
    assert pre.motion_blur is True
    assert pre.reflection is False
    assert pre.occlusion is True


def test_process_is_passthrough_returns_same_array():
    pre = Preprocessor(Config({}))
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    out = pre.process(frame)
    assert out is frame  # M2: değiştirmeden geri verir (kimlik)


def test_process_passthrough_with_flags_enabled():
    # Flag'ler açık olsa da M2'de implementasyon yok → yine pass-through
    cfg = Config({"preprocessing": {"headlight_suppression": True}})
    pre = Preprocessor(cfg)
    frame = np.ones((2, 2, 3), dtype=np.uint8) * 7
    out = pre.process(frame)
    assert np.array_equal(out, frame)
