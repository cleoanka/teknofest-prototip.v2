"""QoD histerezisi: tetik/bırak, çift-tetik önleme, cooldown (model gerektirmez)."""
from __future__ import annotations

from aura.qod import QoDController


def test_trigger_emits_and_activates(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.request_quality(1, "consensus_fail")
    ev = q.drain_events()
    assert len(ev) == 1 and ev[0].type == "QOD_TRIGGER"
    assert ev[0].payload["profile"] == "HIGH_THROUGHPUT"
    assert q.state(1)[0] is True


def test_no_double_trigger_while_active(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.request_quality(1, "r")
    q.request_quality(1, "r")
    assert sum(e.type == "QOD_TRIGGER" for e in q.drain_events()) == 1


def test_release_after_min_active(cfg):
    q = QoDController(cfg)  # min_active=3s
    q.set_now(0.0)
    q.request_quality(1, "r")
    q.drain_events()
    q.set_now(1.0)
    q.tick()
    assert q.state(1)[0] is True            # 1s < 3s → hâlâ aktif
    q.set_now(3.0)
    q.tick()
    ev = q.drain_events()
    assert any(e.type == "QOD_RELEASE" for e in ev) and q.state(1)[0] is False


def test_cooldown_blocks_retrigger(cfg):
    q = QoDController(cfg)  # cooldown=5s
    q.set_now(0.0)
    q.request_quality(1, "r")
    q.drain_events()
    q.set_now(3.0)
    q.tick()
    q.drain_events()                         # 3.0'da bırakıldı
    q.set_now(5.0)
    q.request_quality(1, "r")                # 5-3=2 < 5 → engellenir
    assert q.state(1)[0] is False and not q.drain_events()
    q.set_now(8.5)
    q.request_quality(1, "r")                # 8.5-3=5.5 ≥ 5 → izin verilir
    assert q.state(1)[0] is True


def test_optimize_profile(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.request_optimize(2, "speed_anomaly")
    ev = q.drain_events()
    assert ev[0].payload["profile"] == "LOW_LATENCY"
