"""Integration testleri — gerçek model ağırlığı gerektirir; CI'da skip edilir.

`pytest -m "not integration"` (CI) bunları atlar. Yerelde ağırlık + ultralytics
varsa `pytest -m integration` ile çalıştırın.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_real_yolo_detector_loads(cfg):
    pytest.importorskip("ultralytics")
    weight = Path(cfg.get("models.detector.path", "weights/yolo26s.pt"))
    if not weight.exists():
        pytest.skip("detector ağırlığı yok (bootstrap indirmedi)")
    cfg.data["runtime"]["ai_mode"] = "real"
    from roadguard.detection.detector import build_detector
    from roadguard.detection.yolo import YOLO26Detector

    det = build_detector(cfg)
    assert isinstance(det, YOLO26Detector)


@pytest.mark.integration
def test_real_pipeline_end_to_end(cfg):
    pytest.importorskip("ultralytics")
    if not Path(cfg.get("models.detector.path", "weights/yolo26s.pt")).exists():
        pytest.skip("ağırlık yok")
    cfg.data["runtime"]["ai_mode"] = "real"
    src = "data/samples/ornek.mp4"
    if not Path(src).exists():
        from roadguard.synthetic import generate

        generate(Path("data/samples"), 30, 30, 640, 360)
    from roadguard.pipeline import Pipeline

    events = Pipeline(cfg).run_video(src, max_frames=30)
    assert isinstance(events, list)
