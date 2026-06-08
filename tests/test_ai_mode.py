"""`ai_mode=auto` kaynak-bilinçli çözümleme.

Gömülü sentetik örnek (renkli bloklar) yalnızca mock dedektörle anlamlı tespit
üretir; COCO-eğitimli gerçek YOLO bu blokları araç olarak görmez (0 bbox). Bu
yüzden `auto`, ağırlık mevcut olsa bile sentetik kaynakta mock'a düşer; gerçek
footage/kamera/URL kaynağında ise gerçek YOLO kullanır.
"""

from __future__ import annotations

from aura.config import SAMPLE_VIDEO, is_synthetic_source
from aura.detection.detector import resolve_ai_mode
from aura.driver_state.classifier import resolve_driver_mode


def test_is_synthetic_source_bundled_sample(cfg):
    cfg.data["runtime"]["source"] = str(SAMPLE_VIDEO)
    assert is_synthetic_source(cfg) is True


def test_is_synthetic_source_camera_index(cfg):
    cfg.data["runtime"]["source"] = "0"
    assert is_synthetic_source(cfg) is False


def test_is_synthetic_source_rtsp_url(cfg):
    cfg.data["runtime"]["source"] = "rtsp://cam/live"
    assert is_synthetic_source(cfg) is False


def test_auto_uses_mock_on_synthetic_even_with_weights(cfg, monkeypatch, tmp_path):
    """Ağırlık + ultralytics var ama kaynak sentetik → mock (yoksa 0 bbox olurdu)."""
    import aura.detection.detector as det
    import aura.driver_state.classifier as drv

    weight = tmp_path / "w.pt"
    weight.write_bytes(b"x")  # "ağırlık mevcut" simülasyonu
    monkeypatch.setattr(det, "_ultralytics_available", lambda: True)
    monkeypatch.setattr(drv, "_ultralytics_available", lambda: True)
    cfg.data["models"]["detector"]["path"] = str(weight)
    cfg.data["models"]["driver_state"]["path"] = str(weight)
    cfg.data["runtime"]["ai_mode"] = "auto"
    cfg.data["runtime"]["source"] = str(SAMPLE_VIDEO)
    assert resolve_ai_mode(cfg) == "mock"
    assert resolve_driver_mode(cfg) == "mock"


def test_auto_uses_real_on_real_source_with_weights(cfg, monkeypatch, tmp_path):
    """Gerçek kaynak (URL/kamera) + ağırlık → gerçek YOLO."""
    import aura.detection.detector as det

    weight = tmp_path / "w.pt"
    weight.write_bytes(b"x")
    monkeypatch.setattr(det, "_ultralytics_available", lambda: True)
    cfg.data["models"]["detector"]["path"] = str(weight)
    cfg.data["runtime"]["ai_mode"] = "auto"
    cfg.data["runtime"]["source"] = "rtsp://cam/live"
    assert resolve_ai_mode(cfg) == "real"


def test_auto_falls_back_to_mock_without_weights(cfg, monkeypatch):
    import aura.detection.detector as det

    monkeypatch.setattr(det, "_ultralytics_available", lambda: True)
    cfg.data["models"]["detector"]["path"] = "weights/__nonexistent__.pt"
    cfg.data["runtime"]["ai_mode"] = "auto"
    cfg.data["runtime"]["source"] = "rtsp://cam/live"
    assert resolve_ai_mode(cfg) == "mock"


def test_explicit_modes_are_honored(cfg):
    cfg.data["runtime"]["source"] = str(SAMPLE_VIDEO)  # sentetik olsa bile
    cfg.data["runtime"]["ai_mode"] = "real"
    assert resolve_ai_mode(cfg) == "real"
    cfg.data["runtime"]["ai_mode"] = "mock"
    assert resolve_ai_mode(cfg) == "mock"
