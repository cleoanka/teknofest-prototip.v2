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


def test_mock_accepts_track_id_kwarg(cfg):
    """Liskov: mock infer artık track_id kabul eder (ABC sözleşmesiyle uyumlu).

    Önceden mock track_id ALMIYORDU ve engine._infer TypeError'ı yakalayıp ikinci
    kez çağırarak maskeliyordu. Artık imza tutarlı → maskeleme gerekmez.
    """
    clf = MockDriverClassifier(cfg)
    a1 = clf.infer(_cabin((90, 200, 255)), track_id=7)  # araç 1: telefon + kemer
    assert a1.phone is True and a1.seatbelt is True
    # track_id durumsuz mock'ta sonucu DEĞİŞTİRMEZ (yok sayılır)
    a2 = clf.infer(_cabin((90, 200, 255)), track_id=None)
    assert a2.phone is True and a2.seatbelt is True


def test_mock_no_state_beyond_max_dist(cfg):
    """Renk en yakın referanstan max_dist'ten uzaksa → durum YOK (arka plan)."""
    clf = MockDriverClassifier(cfg)
    clf.max_dist = 160.0
    # araç-1 ref (90,200,255)'e uzak bir renk seç (gri-yeşil): tüm ref'lerden >160
    far = clf.infer(_cabin((10, 10, 10)))  # neredeyse siyah
    assert far.active_flags() == []
    assert far.confidence == {}


def test_mock_within_max_dist_yields_state(cfg):
    """max_dist içindeki renk → en yakın referansın bayrakları üretilir."""
    clf = MockDriverClassifier(cfg)
    # araç-1 ref'ine YAKIN (hafif kaydırılmış) renk: eşik içinde kalır
    ds = clf.infer(_cabin((95, 195, 250)))
    assert ds.phone is True and ds.seatbelt is True
