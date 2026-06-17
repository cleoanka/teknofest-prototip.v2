"""Stage-2 Katman B — ID-merkezli motor (DriverStateEngine + TrackVoter) testleri.

Model GEREKTİRMEZ: ham tahminleri taklit eden bir stub ile yalnızca ID-merkezli
oylama + no_seatbelt TÜRETME mantığını doğrular (inşaat aşaması — AI henüz yok).

Kemer tasarımı: model 'seatbelt' (kemer VAR) tespit eder; no_seatbelt ihlali engine'de
kemerin yokluğundan türetilir.
"""

from __future__ import annotations

from aura.driver_state.engine import DriverStateEngine
from aura.driver_state.voting import TrackVoter
from aura.schema import DriverState


class _StubModel:
    """Sıraya konmuş ham DriverState'leri sırayla döndüren sahte Katman-A modeli."""

    def __init__(self, sequence: list[DriverState]):
        self._seq = sequence
        self.i = 0

    def infer(self, _roi):
        ds = self._seq[min(self.i, len(self._seq) - 1)]
        self.i += 1
        return ds


def _ds(**flags) -> DriverState:
    ds = DriverState()
    for k, v in flags.items():
        setattr(ds, k, True)
        ds.confidence[k] = v if isinstance(v, float) else 1.0
    return ds


# --- TrackVoter ham oy mantığı --------------------------------------------- #


def test_voter_needs_min_votes():
    """Eşiğin altındaki oy → pasif; eşiği geçince → aktif."""
    v = TrackVoter(window=4, min_votes=3)
    for _ in range(2):
        v.update(_ds(phone=0.9), 0)
    assert v.stable_raw().phone is False  # 2 oy < 3
    v.update(_ds(phone=0.9), 0)
    assert v.stable_raw().phone is True  # 3 oy >= 3


def test_voter_filters_single_frame_noise():
    """16/8: tek karelik patlama (1 oy) kararlı duruma dönüşmez."""
    v = TrackVoter(window=16, min_votes=8)
    v.update(_ds(phone=0.99), 0)
    for _ in range(15):
        v.update(DriverState(), 0)
    assert v.stable_raw().active_flags() == []


def test_voter_reports_mean_confidence():
    v = TrackVoter(window=4, min_votes=2)
    v.update(_ds(smoking=0.6), 0)
    v.update(_ds(smoking=0.8), 0)
    ds = v.stable_raw()
    assert ds.smoking is True
    assert abs(ds.confidence["smoking"] - 0.7) < 1e-6


def test_voter_tracks_seen_and_votes():
    v = TrackVoter(window=3, min_votes=2)
    v.update(_ds(seatbelt=0.9), 0)
    v.update(DriverState(), 0)
    assert v.seen == 2
    assert v.votes("seatbelt") == 1


# --- DriverStateEngine: ID-merkezli davranış + no_seatbelt türetme ---------- #


def test_engine_is_id_centric(cfg):
    """Farklı track_id'ler birbirinin tamponunu KİRLETMEZ."""
    eng = DriverStateEngine(cfg)
    eng.model = _StubModel([_ds(phone=0.9, seatbelt=0.9)])  # phone + kemerli
    for fi in range(eng.min_votes):
        out1 = eng.process(track_id=1, cabin_roi=object(), frame_idx=fi)
    assert out1.phone is True
    assert 2 not in eng.voters  # ID 2 için tampon hiç oluşmadı


def test_engine_derives_no_seatbelt_when_belt_absent(cfg):
    """Toggle AÇIKKEN: kemer hiç görülmezse, yeterli gözlemden sonra no_seatbelt türetilir."""
    eng = DriverStateEngine(cfg)
    eng.window, eng.min_votes = 16, 8
    eng.derive_no_seatbelt = True  # türetmeyi aç (varsayılan kapalı)
    eng.model = _StubModel([_ds(smoking=0.9)])  # sigara, kemer YOK
    outs = [eng.process(1, object(), fi) for fi in range(8)]
    assert outs[6].no_seatbelt is False  # 7 gözlem < min_votes → henüz türetme yok
    assert outs[7].no_seatbelt is True  # 8. gözlemde türetildi
    assert outs[7].smoking is True
    assert "no_seatbelt" in outs[7].confidence


def test_engine_no_seatbelt_disabled_by_default(cfg):
    """Varsayılan KAPALI: kemer hiç görülmese bile no_seatbelt türetilmez."""
    eng = DriverStateEngine(cfg)
    eng.window, eng.min_votes = 16, 8
    assert eng.derive_no_seatbelt is False  # config varsayılanı
    eng.model = _StubModel([_ds(smoking=0.9)])  # kemer YOK ama toggle kapalı
    out = None
    for fi in range(20):
        out = eng.process(1, object(), fi)
    assert out.no_seatbelt is False
    assert out.smoking is True


def test_engine_no_seatbelt_requires_driver_present(cfg):
    """Kapı AÇIK (varsayılan): sürücü kabinde görülmezse kemer yok olsa bile türetilmez."""
    eng = DriverStateEngine(cfg)
    eng.window, eng.min_votes = 16, 8
    eng.derive_no_seatbelt = True  # türetmeyi aç
    assert eng.require_present is True  # config varsayılanı (no_seatbelt.require_driver_present)
    eng.model = _StubModel([_ds(smoking=0.9)])  # kemer YOK ama sürücü de YOK (present=False)
    out = None
    for fi in range(20):
        out = eng.process(1, object(), fi, driver_present=False)
    assert out.no_seatbelt is False  # sürücü-varlığı kapısı ihlali engelledi
    assert out.smoking is True  # ham bayraklar yine de oylanır


def test_engine_present_gate_can_be_disabled(cfg):
    """Kapı KAPATILIRSA: sürücü görülmese bile eski davranış → no_seatbelt türetilir."""
    eng = DriverStateEngine(cfg)
    eng.window, eng.min_votes = 16, 8
    eng.derive_no_seatbelt = True
    eng.require_present = False  # kapıyı kapat
    eng.model = _StubModel([_ds(smoking=0.9)])  # kemer YOK
    out = None
    for fi in range(8):
        out = eng.process(1, object(), fi, driver_present=False)
    assert out.no_seatbelt is True  # kapı kapalı → varlıktan bağımsız türetilir


def test_engine_no_violation_when_belt_present(cfg):
    """Toggle açık olsa bile kemer kararlı görülürse no_seatbelt türetilmez."""
    eng = DriverStateEngine(cfg)
    eng.window, eng.min_votes = 16, 8
    eng.derive_no_seatbelt = True
    eng.model = _StubModel([_ds(phone=0.9, seatbelt=0.95)])  # telefon ama KEMERLİ
    out = None
    for fi in range(12):
        out = eng.process(1, object(), fi)
    assert out.seatbelt is True
    assert out.no_seatbelt is False
    assert out.phone is True


def test_engine_prune_drops_stale_tracks(cfg):
    eng = DriverStateEngine(cfg)
    eng.max_age = 30
    eng.model = _StubModel([DriverState()])
    eng.process(1, object(), frame_idx=0)
    assert 1 in eng.voters
    eng.prune(frame_idx=100)  # 100 - 0 > 30 → düşmeli
    assert 1 not in eng.voters
