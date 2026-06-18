"""SignTracker kapsam-kapatan birim testleri (test_gaps).

Mock-mod, model gerektirmez. Kenar durumlar: limit→farklı-limit görünür kalırken
(decay'e ulaşmaz), generic+limit karışımı, conf-tie determinizmi, devre-dışı.
"""

from __future__ import annotations

from aura.detection.detector import Sign
from aura.scene.sign_tracker import SignTracker
from aura.schema import BBox


def _sign(cls: str, conf: float = 0.9) -> Sign:
    return Sign(bbox=BBox(x1=10, y1=10, x2=40, y2=40, conf=conf, cls=cls), cls=cls)


def test_disabled_tracker_no_events(cfg):
    cfg.data["sign"]["enabled"] = False
    st = SignTracker(cfg)
    scene, events = st.update([_sign("speed_limit_50")], frame_idx=0)
    assert events == []
    assert scene.active_speed_limit_kmh is None
    assert scene.sign_count == 1


def test_limit_changes_while_visible_every_frame_no_decay(cfg):
    """Tabela her kare görünür ama limit değişir → last_seen tazelenir, decay'e ulaşmaz."""
    st = SignTracker(cfg)
    # 50 limiti, sonra persistence'tan çok sonra 30 görünür AMA görünür olduğu için decay yok
    _, e0 = st.update([_sign("speed_limit_50")], frame_idx=0)
    far = st.persistence + 500
    scene, e1 = st.update([_sign("speed_limit_30")], frame_idx=far)
    assert scene.active_speed_limit_kmh == 30  # decay değil, değişim
    assert len(e0) == 1 and len(e1) == 1
    assert e1[0].payload["speed_limit_kmh"] == 30
    assert st._last_seen == far  # her görülmede tazelendi


def test_generic_and_limit_mix_picks_limit(cfg):
    st = SignTracker(cfg)
    scene, events = st.update([_sign("sign", conf=0.99), _sign("speed_limit_50")], frame_idx=0)
    assert scene.active_speed_limit_kmh == 50  # generic atlanır, limit seçilir
    assert len(events) == 1


def test_conf_tie_first_seen_wins(cfg):
    """Eşit conf → strict '>' nedeniyle ilk görülen kazanır (deterministik)."""
    st = SignTracker(cfg)
    scene, _ = st.update(
        [_sign("speed_limit_30", conf=0.8), _sign("speed_limit_90", conf=0.8)], frame_idx=0
    )
    assert scene.active_speed_limit_kmh == 30  # eşitlikte ilk (30) kazanır


def test_limit_of_helper(cfg):
    st = SignTracker(cfg)
    assert st.limit_of("sign") is None  # generic → limit yok
    # value_map'te bir hız-limiti sınıfı olmalı
    some_cls = next(iter(st.value_map))
    assert st.limit_of(some_cls) == st.value_map[some_cls]


def test_active_limit_property_tracks_state(cfg):
    st = SignTracker(cfg)
    assert st.active_limit is None
    st.update([_sign("speed_limit_70")], frame_idx=0)
    assert st.active_limit == 70
