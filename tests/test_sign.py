"""Trafik tabelası (sign) feature — SignTracker aktif limit + accumulator hız ihlali.

Mock modda, model ağırlığı GEREKTİRMEZ (saf hesap). Zincir:
  tabela tespiti → SignTracker aktif limit → accumulator speed.over_limit
  → SPEED_LIMIT_VIOLATION event + risk_flags.
"""

from __future__ import annotations

import numpy as np

from aura.accumulator.accumulator import Accumulator
from aura.config import Config
from aura.detection.detector import Sign
from aura.detection.mock import MockDetector
from aura.scene.sign_tracker import SCENE_TRACK_ID, SignTracker
from aura.schema import BBox, SceneContext, SpeedState


def _sign(cls: str, conf: float = 0.9) -> Sign:
    return Sign(bbox=BBox(x1=10, y1=10, x2=40, y2=40, conf=conf, cls=cls), cls=cls)


def _vehicle() -> BBox:
    return BBox(x1=0, y1=0, x2=50, y2=50, conf=0.9, cls="car")


# --- SignTracker ----------------------------------------------------------- #
def test_speed_limit_sign_sets_active_limit(cfg):
    st = SignTracker(cfg)
    scene, events = st.update([_sign("speed_limit_50")], frame_idx=0)
    assert scene.active_speed_limit_kmh == 50
    assert scene.speed_limit_source_cls == "speed_limit_50"
    assert len(events) == 1
    assert events[0].type == "SPEED_LIMIT_DETECTED"
    assert events[0].track_id == SCENE_TRACK_ID
    assert events[0].payload["speed_limit_kmh"] == 50


def test_generic_sign_has_no_limit(cfg):
    st = SignTracker(cfg)
    scene, events = st.update([_sign("sign")], frame_idx=0)
    assert scene.active_speed_limit_kmh is None
    assert scene.sign_count == 1
    assert events == []


def test_low_confidence_sign_ignored(cfg):
    st = SignTracker(cfg)  # min_conf 0.40
    scene, events = st.update([_sign("speed_limit_50", conf=0.2)], frame_idx=0)
    assert scene.active_speed_limit_kmh is None
    assert events == []


def test_limit_persists_then_decays(cfg):
    st = SignTracker(cfg)
    st.update([_sign("speed_limit_70")], frame_idx=0)
    # Tabela artık görünmüyor ama persistence sınırında hâlâ geçerli
    scene, _ = st.update([], frame_idx=st.persistence)
    assert scene.active_speed_limit_kmh == 70
    # Sınır aşılınca sessizce düşer
    scene2, events2 = st.update([], frame_idx=st.persistence + 1)
    assert scene2.active_speed_limit_kmh is None
    assert events2 == []


def test_limit_change_emits_new_event_only_on_change(cfg):
    st = SignTracker(cfg)
    _, e1 = st.update([_sign("speed_limit_50")], frame_idx=0)
    _, e2 = st.update([_sign("speed_limit_50")], frame_idx=1)  # aynı limit → event yok
    _, e3 = st.update([_sign("speed_limit_30")], frame_idx=2)  # değişti → yeni event
    assert len(e1) == 1 and e2 == [] and len(e3) == 1
    assert e3[0].payload["speed_limit_kmh"] == 30


def test_highest_confidence_sign_wins(cfg):
    st = SignTracker(cfg)
    scene, _ = st.update(
        [_sign("speed_limit_30", conf=0.5), _sign("speed_limit_90", conf=0.95)], frame_idx=0
    )
    assert scene.active_speed_limit_kmh == 90


# --- Accumulator hız-limiti ihlali ----------------------------------------- #
def test_accumulator_fires_violation_over_limit(cfg):
    acc = Accumulator(cfg)
    acc.set_scene(SceneContext(active_speed_limit_kmh=50))
    rec, events = acc.update_track(
        1,
        frame_idx=0,
        bbox=_vehicle(),
        vehicle_class="car",
        speed=SpeedState(value_kmh=80.0, mode="metric", is_calibrated=True),
    )
    viol = [e for e in events if e.type == "SPEED_LIMIT_VIOLATION"]
    assert len(viol) == 1
    assert viol[0].payload["limit_kmh"] == 50
    assert viol[0].payload["speed_kmh"] == 80.0
    assert viol[0].payload["over_by_kmh"] == 30.0
    assert "speed_limit_violation" in rec.risk_flags
    # Genel RISK_ALERT yerine zengin event çıkmalı (çift event olmamalı)
    assert not any(
        e.type == "RISK_ALERT" and e.payload.get("rule") == "speed_limit_violation" for e in events
    )


def test_no_violation_under_limit(cfg):
    acc = Accumulator(cfg)
    acc.set_scene(SceneContext(active_speed_limit_kmh=90))
    rec, events = acc.update_track(
        2, frame_idx=0, bbox=_vehicle(), vehicle_class="car", speed=SpeedState(value_kmh=70.0)
    )
    assert not any(e.type == "SPEED_LIMIT_VIOLATION" for e in events)
    assert "speed_limit_violation" not in rec.risk_flags


def test_no_active_limit_no_violation_even_if_fast(cfg):
    acc = Accumulator(cfg)  # set_scene çağrılmadı → active_speed_limit None
    rec, events = acc.update_track(
        3, frame_idx=0, bbox=_vehicle(), vehicle_class="car", speed=SpeedState(value_kmh=200.0)
    )
    assert not any(e.type == "SPEED_LIMIT_VIOLATION" for e in events)
    assert "speed_limit_violation" not in rec.risk_flags


def test_violation_fires_once_not_every_frame(cfg):
    acc = Accumulator(cfg)
    acc.set_scene(SceneContext(active_speed_limit_kmh=50))
    fast = SpeedState(value_kmh=80.0, mode="metric", is_calibrated=True)
    _, ev0 = acc.update_track(9, frame_idx=0, bbox=_vehicle(), vehicle_class="car", speed=fast)
    _, ev1 = acc.update_track(9, frame_idx=1, bbox=_vehicle(), vehicle_class="car", speed=fast)
    assert any(e.type == "SPEED_LIMIT_VIOLATION" for e in ev0)
    # Aynı bayrak zaten aktif → ikinci karede tekrar event üretilmez
    assert not any(e.type == "SPEED_LIMIT_VIOLATION" for e in ev1)


# --- Mock dedektör → last_signs wiring (ağırlıksız) ------------------------ #
def test_mock_detector_emits_synthetic_sign():
    cfg = Config(
        {
            "models": {"detector": {"vehicle_classes": ["car"]}},
            "sign": {
                "enabled": True,
                "mock_synthetic": True,
                "mock_speed_limit": 70,
                "value_map": {"speed_limit_70": 70},
            },
        }
    )
    det = MockDetector(cfg)
    det.detect(np.zeros((120, 200, 3), dtype=np.uint8))
    assert len(det.last_signs) == 1
    assert det.last_signs[0].cls == "speed_limit_70"
