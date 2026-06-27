"""QoD histerezisi: tetik/bırak, çift-tetik önleme, cooldown (model gerektirmez)."""

from __future__ import annotations

from roadguard.qod import QoDController


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
    assert q.state(1)[0] is True  # 1s < 3s → hâlâ aktif
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
    q.drain_events()  # 3.0'da bırakıldı
    q.set_now(5.0)
    q.request_quality(1, "r")  # 5-3=2 < 5 → engellenir
    assert q.state(1)[0] is False and not q.drain_events()
    q.set_now(8.5)
    q.request_quality(1, "r")  # 8.5-3=5.5 ≥ 5 → izin verilir
    assert q.state(1)[0] is True


def test_optimize_profile(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.request_optimize(2, "speed_anomaly")
    ev = q.drain_events()
    assert ev[0].payload["profile"] == "LOW_LATENCY"


# --------------------------------------------------------------------------- #
# release() — manuel bırakma + cooldown kaydı
# --------------------------------------------------------------------------- #
def test_manual_release_emits_and_records_cooldown(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.request_optimize(1, "swerving")
    q.drain_events()
    q.release(1)
    ev = q.drain_events()
    assert any(e.type == "QOD_RELEASE" for e in ev)
    assert q.state(1)[0] is False
    # cooldown kaydedildi: hemen yeniden tetikleme engellenir.
    q.request_optimize(1, "swerving")
    assert q.state(1)[0] is False and not q.drain_events()


def test_release_unknown_track_noop(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.release(999)  # oturum yok → sessizce hiçbir şey yapma
    assert not q.drain_events()


# --------------------------------------------------------------------------- #
# release_quality() — yalnız KALİTE oturumunu bırak, optimize'a DOKUNMA
# --------------------------------------------------------------------------- #
def test_release_quality_releases_quality_session(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.request_quality(1, "consensus_ok")
    q.drain_events()
    q.release_quality(1)
    assert q.state(1)[0] is False
    assert any(e.type == "QOD_RELEASE" for e in q.drain_events())


def test_release_quality_does_not_touch_optimize_session(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.request_optimize(1, "swerving")  # kind=optimize
    q.drain_events()
    q.release_quality(1)  # guard: kind != quality → dokunma
    assert q.state(1)[0] is True  # optimize oturumu yaşıyor
    assert not q.drain_events()


# --------------------------------------------------------------------------- #
# Tek-oturum/track sınırı: ikinci request (farklı kind) sessizce yok sayılır
# --------------------------------------------------------------------------- #
def test_second_request_ignored_while_active(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.request_optimize(1, "swerving")
    q.drain_events()
    # Aynı track'e farklı kind ile ikinci talep → tek-oturum varsayımı, yok sayılır.
    q.request_quality(1, "consensus_fail")
    assert not any(e.type == "QOD_TRIGGER" for e in q.drain_events())
    # Oturum HÂLÂ optimize (kind değişmedi).
    assert q.active_sessions()[1]["kind"] == "optimize"


# --------------------------------------------------------------------------- #
# Bug fix: ongoing trigger min_active sayacını tazeler → kritik an sürerken
# tick() oturumu DÜŞÜRMEZ (LOW_LATENCY yaşamaya devam eder).
# --------------------------------------------------------------------------- #
def test_ongoing_request_refreshes_min_active_keeps_session(cfg):
    q = QoDController(cfg)  # min_active=3
    q.set_now(0.0)
    q.request_optimize(1, "swerving")
    q.drain_events()
    # 2s sonra anomali sürüyor → tekrar request (since tazelenir), tick → düşmez.
    q.set_now(2.0)
    q.request_optimize(1, "swerving")
    q.tick()
    assert q.state(1)[0] is True
    # 4s: ilk tetikten 4s geçti AMA son tetikten yalnız 2s → hâlâ aktif (tazeleme çalıştı).
    q.set_now(4.0)
    q.tick()
    assert q.state(1)[0] is True
    # Anomali bitti (artık request yok): son tetik 2.0 + min_active 3 = 5.0'da düşer.
    q.set_now(5.0)
    q.tick()
    assert q.state(1)[0] is False


def test_ongoing_refresh_emits_no_extra_trigger(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.request_optimize(1, "swerving")
    q.set_now(1.0)
    q.request_optimize(1, "swerving")  # tazeleme — yeni event YOK
    assert sum(e.type == "QOD_TRIGGER" for e in q.drain_events()) == 1


# --------------------------------------------------------------------------- #
# Frame-saat ts: emit edilen QoD event'leri _now'u ts olarak taşır
# --------------------------------------------------------------------------- #
def test_qod_events_use_frame_clock_ts(cfg):
    q = QoDController(cfg)
    q.set_now(7.0)
    q.request_optimize(1, "swerving")
    trig = q.drain_events()
    assert trig and trig[0].ts == 7.0
    q.set_now(11.0)
    q.tick()  # 11-7=4 >= 3 → release
    rel = q.drain_events()
    assert rel and rel[0].ts == 11.0


def test_tick_empty_sessions_noop(cfg):
    q = QoDController(cfg)
    q.set_now(5.0)
    q.tick()  # oturum yok → erken çıkış, event yok
    assert not q.drain_events()


def test_active_sessions_is_copy(cfg):
    q = QoDController(cfg)
    q.set_now(0.0)
    q.request_optimize(1, "r")
    snap = q.active_sessions()
    snap.clear()  # dışarıdaki kopyayı boşalt → iç durum etkilenmemeli
    assert q.state(1)[0] is True
