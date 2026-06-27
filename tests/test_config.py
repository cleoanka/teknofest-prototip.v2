"""Config profil katmanı + derin-merge testleri (WS-A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from roadguard.config import (
    Config,
    _apply_env_overrides,
    _deep_merge,
    available_profiles,
    is_synthetic_source,
    load_config,
    resolve_profile_path,
    resolve_source,
)


def test_deep_merge_recursive_and_replace():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2, 3], "c": 9}
    overlay = {"a": {"y": 20, "z": 30}, "b": [9]}
    out = _deep_merge(base, overlay)
    assert out["a"] == {"x": 1, "y": 20, "z": 30}  # iç içe sözlük birleşir
    assert out["b"] == [9]  # liste TAMAMEN değişir (kısmi merge yok)
    assert out["c"] == 9  # dokunulmayan korunur
    # mutasyonsuz: kaynaklar değişmemeli
    assert base["a"] == {"x": 1, "y": 2}


def test_default_detector_is_yolo26():
    cfg = load_config()
    assert cfg.profile is None
    assert "yolo26" in cfg.get("models.detector.path")  # YOLO26 mandate (varsayılan)


def test_profiles_exist():
    profs = available_profiles()
    for expected in ("server", "laptop", "v4-finetune"):
        assert expected in profs


@pytest.mark.parametrize(
    "profile,needle",
    [
        ("server", "yolo26l"),
        ("laptop", "yolo26s"),
        ("v4-finetune", "yolguvenligi_types_v4"),
    ],
)
def test_profile_overrides_detector(profile, needle):
    cfg = load_config(profile=profile)
    assert cfg.profile == profile
    assert needle in cfg.get("models.detector.path")
    # derin-merge: profilde olmayan taban anahtarları korunmalı
    assert cfg.get("plate.regex") is not None
    assert cfg.get("qod.approach.enabled") is True


def test_unknown_profile_is_ignored_gracefully():
    cfg = load_config(profile="does-not-exist")
    assert cfg.profile is None  # uygulanmadı ama hata da atmadı
    assert "yolo26" in cfg.get("models.detector.path")  # taban davranış korunur


def test_env_profile(monkeypatch):
    monkeypatch.setenv("ROADGUARD_PROFILE", "laptop")
    cfg = load_config()
    assert cfg.profile == "laptop"
    assert "yolo26s" in cfg.get("models.detector.path")


def test_explicit_profile_beats_env(monkeypatch):
    monkeypatch.setenv("ROADGUARD_PROFILE", "laptop")
    cfg = load_config(profile="server")
    assert cfg.profile == "server"


def test_resolve_profile_path_bare_name():
    p = resolve_profile_path("server")
    assert p.name == "server.yaml"
    assert p.parent.name == "profiles"


# --------------------------------------------------------------------------- #
# resolve_source: kamera/URL geçişi, dosya çözümü, fallback dalı
# --------------------------------------------------------------------------- #
def test_resolve_source_camera_index_passthrough():
    cfg = Config({"runtime": {"source": "0"}})
    assert resolve_source(cfg) == "0"


def test_resolve_source_int_passthrough():
    cfg = Config({"runtime": {"source": 0}})
    assert resolve_source(cfg) == 0


def test_resolve_source_stream_url_passthrough():
    cfg = Config({"runtime": {"source": "rtsp://cam/stream"}})
    assert resolve_source(cfg) == "rtsp://cam/stream"


def test_resolve_source_fallback_to_sample_when_missing():
    # Var olmayan dosya → paketteki örnek videoya düşer (sessiz ölü-akış yerine demo)
    cfg = Config({"runtime": {"source": "yok/olmayan_video.mp4"}})
    resolved = resolve_source(cfg)
    assert Path(resolved).name == "ornek.mp4"


def test_resolve_source_existing_file_absolute(tmp_path):
    vid = tmp_path / "real.mp4"
    vid.write_bytes(b"\x00")
    cfg = Config({"runtime": {"source": str(vid)}})
    resolved = resolve_source(cfg)
    assert Path(resolved) == vid


def test_is_synthetic_source_true_for_sample():
    cfg = load_config()  # varsayılan source = ornek.mp4
    cfg.data.setdefault("runtime", {})["source"] = "data/samples/ornek.mp4"
    assert is_synthetic_source(cfg) is True


def test_is_synthetic_source_false_for_camera():
    cfg = Config({"runtime": {"source": "0"}})
    assert is_synthetic_source(cfg) is False


# --------------------------------------------------------------------------- #
# _apply_env_overrides: ai_mode/device + port int-parse (isdigit dalı)
# --------------------------------------------------------------------------- #
def test_apply_env_overrides_ai_mode_and_device(monkeypatch):
    monkeypatch.setenv("AI_MODE", "mock")
    monkeypatch.setenv("ROADGUARD_DEVICE", "cpu")
    data = _apply_env_overrides({})
    assert data["runtime"]["ai_mode"] == "mock"
    assert data["runtime"]["device"] == "cpu"


def test_apply_env_overrides_port_int_parse(monkeypatch):
    monkeypatch.setenv("ROADGUARD_INFERENCE_PORT", "8123")
    data = _apply_env_overrides({})
    assert data["services"]["inference_api"] == 8123
    assert isinstance(data["services"]["inference_api"], int)


def test_apply_env_overrides_non_numeric_port_ignored(monkeypatch):
    # rakam-değil değer (isdigit False) → atlanır, kazara str yazılmaz
    monkeypatch.setenv("ROADGUARD_QOD_MOCK_PORT", "abc")
    data = _apply_env_overrides({})
    assert "qod_mock" not in data.get("services", {})


def test_apply_env_overrides_no_env_no_change(monkeypatch):
    monkeypatch.delenv("AI_MODE", raising=False)
    monkeypatch.delenv("ROADGUARD_DEVICE", raising=False)
    for k in ("ROADGUARD_INFERENCE_PORT", "ROADGUARD_QOD_MOCK_PORT", "ROADGUARD_NV_MOCK_PORT"):
        monkeypatch.delenv(k, raising=False)
    data = _apply_env_overrides({"services": {}})
    assert data == {"services": {}}
