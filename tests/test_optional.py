"""§8 opsiyonel modüller — lazy import (kapalıyken import yok) + işlevsellik."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from roadguard.config import load_config
from roadguard.optional.loader import _reset_cache, get_optional, is_enabled
from roadguard.pipeline import Pipeline
from roadguard.schema import BBox

SAMPLE = "data/samples/ornek.mp4"
_MODULES = ("zero_waste_payload", "super_resolution", "homography_ipm")


def _ensure_sample():
    if not Path(SAMPLE).exists():
        from roadguard.synthetic import generate

        generate(Path("data/samples"), 90, 30, 640, 360)


def _purge():
    _reset_cache()
    for m in _MODULES:
        sys.modules.pop(f"roadguard.optional.{m}", None)


def test_disabled_modules_not_imported(cfg):
    _ensure_sample()
    _purge()
    assert not is_enabled(cfg, "zero_waste_payload")
    Pipeline(cfg).run_video(SAMPLE, max_frames=20)
    for m in ("zero_waste_payload", "super_resolution"):
        assert f"roadguard.optional.{m}" not in sys.modules  # kapalıyken import YOK


def test_get_optional_none_when_off(cfg):
    _purge()
    assert get_optional(cfg, "super_resolution") is None
    assert "roadguard.optional.super_resolution" not in sys.modules


def test_zero_waste_enabled_imports_and_attaches(cfg):
    _ensure_sample()
    _purge()
    cfg.data["optional_modules"]["zero_waste_payload"] = True
    cfg.data.setdefault("runtime", {})["ai_mode"] = "mock"  # sentetik karede deterministik tespit
    pipe = Pipeline(cfg)
    pipe.run_video(SAMPLE, max_frames=40)
    assert "roadguard.optional.zero_waste_payload" in sys.modules
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


# --------------------------------------------------------------------------- #
# zero_waste_payload.build_payload: ROI yok / imencode başarısız / ROI var
# --------------------------------------------------------------------------- #
def test_build_payload_structured_only_no_roi():
    from roadguard.optional.zero_waste_payload import build_payload

    track = {"track_id": 7, "cls": "car", "plate": "34TC8532", "speed_kmh": 50}
    p = build_payload(track, plate_roi=None)
    assert p["track_id"] == 7
    assert p["structured"]["cls"] == "car"
    assert p["structured"]["plate"] == "34TC8532"
    assert "plate_roi_jpeg_b64" not in p  # ROI yok → JPEG eklenmez


def test_build_payload_empty_roi_skipped():
    from roadguard.optional.zero_waste_payload import build_payload

    empty = np.zeros((0, 0, 3), np.uint8)  # size==0
    p = build_payload({"track_id": 1}, plate_roi=empty)
    assert "plate_roi_jpeg_b64" not in p


def test_build_payload_attaches_roi_jpeg():
    from roadguard.optional.zero_waste_payload import build_payload

    roi = np.full((12, 30, 3), 200, np.uint8)
    p = build_payload({"track_id": 2}, plate_roi=roi)
    assert "plate_roi_jpeg_b64" in p
    assert p["roi_bytes"] > 0
    assert isinstance(p["plate_roi_jpeg_b64"], str)


def test_build_payload_imencode_fail_no_roi_key(monkeypatch):
    # cv2.imencode ok=False döndürürse (kodlama başarısız) JPEG eklenmez
    import roadguard.optional.zero_waste_payload as zwp

    monkeypatch.setattr(zwp.cv2, "imencode", lambda *a, **k: (False, None))
    roi = np.full((12, 30, 3), 200, np.uint8)
    p = zwp.build_payload({"track_id": 3}, plate_roi=roi)
    assert "plate_roi_jpeg_b64" not in p
    assert "roi_bytes" not in p


# --------------------------------------------------------------------------- #
# super_resolution.enhance: None/boş erken-dönüş + _warned bir-kez
# --------------------------------------------------------------------------- #
def test_super_resolution_none_returns_none():
    import roadguard.optional.super_resolution as sr

    assert sr.enhance(None) is None


def test_super_resolution_empty_roi_passthrough():
    import roadguard.optional.super_resolution as sr

    empty = np.zeros((0, 0, 3), np.uint8)
    out = sr.enhance(empty)
    assert out is empty  # size==0 → olduğu gibi döner


def test_super_resolution_warns_only_once(caplog):
    import roadguard.optional.super_resolution as sr

    sr._warned = False  # durumu sıfırla
    roi = np.full((5, 8, 3), 100, np.uint8)
    with caplog.at_level("INFO", logger="roadguard.optional.super_resolution"):
        sr.enhance(roi, scale=2)
        sr.enhance(roi, scale=2)
    infos = [r for r in caplog.records if "bicubic" in r.getMessage()]
    assert len(infos) == 1  # uyarı yalnız bir kez loglanır


def test_homography_ipm_speed():
    cfg = load_config()
    cfg.data["optional_modules"]["homography_ipm"] = True
    cfg.data["speed"]["calibration_file"] = "config/calibration/ornek_kamera.yaml"
    from roadguard.optional.homography_ipm import _state, ipm_speed

    _state.clear()
    fs = (360, 640, 3)
    v1 = ipm_speed(cfg, 1, BBox(x1=300, y1=200, x2=340, y2=250), 0, 30, fs)
    v2 = ipm_speed(cfg, 1, BBox(x1=300, y1=230, x2=340, y2=300), 1, 30, fs)
    assert v1 is None  # ilk kare → referans yok
    assert v2 is not None and v2 >= 0  # IPM ile hız hesaplandı
