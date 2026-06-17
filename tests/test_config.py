"""Config profil katmanı + derin-merge testleri (WS-A)."""

from __future__ import annotations

import pytest

from aura.config import (
    _deep_merge,
    available_profiles,
    load_config,
    resolve_profile_path,
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
    monkeypatch.setenv("AURA_PROFILE", "laptop")
    cfg = load_config()
    assert cfg.profile == "laptop"
    assert "yolo26s" in cfg.get("models.detector.path")


def test_explicit_profile_beats_env(monkeypatch):
    monkeypatch.setenv("AURA_PROFILE", "laptop")
    cfg = load_config(profile="server")
    assert cfg.profile == "server"


def test_resolve_profile_path_bare_name():
    p = resolve_profile_path("server")
    assert p.name == "server.yaml"
    assert p.parent.name == "profiles"
