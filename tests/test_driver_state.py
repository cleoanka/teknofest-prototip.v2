"""Sürücü-durum mock sınıflandırıcı (renk→durum eşlemesi, CI-uyumlu)."""

from __future__ import annotations

import numpy as np

from aura.driver_state import build_driver_classifier
from aura.driver_state.mock import MockDriverClassifier


def _cabin(color) -> np.ndarray:
    f = np.zeros((40, 60, 3), np.uint8)
    f[:] = color
    return f


def test_mock_maps_scenario_colors(cfg):
    clf = MockDriverClassifier(cfg)
    a1 = clf.infer(_cabin((90, 200, 255)))  # araç 1: telefon + KEMERLİ
    assert a1.phone is True and a1.seatbelt is True
    ds = clf.infer(_cabin((120, 255, 120)))  # araç 2: sigara, KEMER YOK
    # no_seatbelt artık HAM bayrak değil (Katman B'de türetilir) → mock onu üretmez
    assert ds.smoking is True and ds.seatbelt is False and ds.no_seatbelt is False
    assert clf.infer(_cabin((200, 150, 255))).fatigue is True  # araç 3: yorgun + kemerli


def test_mock_background_no_state(cfg):
    ds = MockDriverClassifier(cfg).infer(_cabin((40, 40, 40)))  # asfalt
    assert ds.active_flags() == []


def test_mock_multilabel_confidence(cfg):
    ds = MockDriverClassifier(cfg).infer(_cabin((90, 200, 255)))  # araç 1: phone + seatbelt
    assert set(ds.confidence) == {"phone", "seatbelt"}
    assert all(0.5 <= c <= 1.0 for c in ds.confidence.values())


def test_build_falls_back_to_mock_and_handles_none(cfg):
    # auto + ağırlık yok → mock (yolo26l mevcut olsa da var olmayan yola işaret et)
    cfg.data["runtime"]["ai_mode"] = "auto"
    cfg.data["models"]["driver_state"]["path"] = "weights/__nonexistent__.pt"
    clf = build_driver_classifier(cfg)
    assert isinstance(clf, MockDriverClassifier)
    assert clf.infer(None).active_flags() == []
