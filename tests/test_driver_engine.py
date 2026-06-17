"""Stage-2 Katman B — ID-merkezli motor (DriverStateEngine + TrackVoter) testleri.

Bu testler MODEL GEREKTİRMEZ: ham tahminleri taklit eden basit bir stub model ile
yalnızca ID-merkezli oylama + aux füzyonu mantığını doğrular.
(Mustafa'nın feature/stage2-driver-state testleri; v2.3'te zengin Katman A + aux
füzyonu ile entegre edildi — stub artık track_id kabul eder, aux_flags testi eklendi.)
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

    def infer(self, _roi, track_id=None):  # track_id: gerçek backend imzasıyla uyum
        ds = self._seq[min(self.i, len(self._seq) - 1)]
        self.i += 1
        return ds


def _ds(**flags) -> DriverState:
    ds = DriverState()
    for k, v in flags.items():
        setattr(ds, k, True)
        ds.confidence[k] = v if isinstance(v, float) else 1.0
    return ds


# --- TrackVoter birim mantığı ---------------------------------------------- #


def test_voter_needs_min_votes():
    """Eşiğin altındaki oy → pasif; eşiği geçince → aktif."""
    v = TrackVoter(window=4, min_votes=3)
    for _ in range(2):
        v.update(_ds(phone=0.9), 0)
    assert v.stable().phone is False  # 2 oy < 3
    v.update(_ds(phone=0.9), 0)
    assert v.stable().phone is True  # 3 oy >= 3


def test_voter_filters_single_frame_noise():
    """16/8: tek karelik patlama (1 oy) kararlı duruma dönüşmez."""
    v = TrackVoter(window=16, min_votes=8)
    v.update(_ds(phone=0.99), 0)
    for _ in range(15):
        v.update(DriverState(), 0)
    assert v.stable().active_flags() == []


def test_voter_reports_mean_confidence():
    v = TrackVoter(window=4, min_votes=2)
    v.update(_ds(smoking=0.6), 0)
    v.update(_ds(smoking=0.8), 0)
    ds = v.stable()
    assert ds.smoking is True
    assert abs(ds.confidence["smoking"] - 0.7) < 1e-6


def test_voter_window_slides_out_old_votes():
    """Pencere dolunca eski True oylar düşer → bayrak tekrar pasifleşebilir."""
    v = TrackVoter(window=3, min_votes=2)
    v.update(_ds(phone=0.9), 0)
    v.update(_ds(phone=0.9), 0)
    assert v.stable().phone is True
    for _ in range(3):  # pencereyi negatiflerle doldur
        v.update(DriverState(), 0)
    assert v.stable().phone is False


# --- DriverStateEngine: ID-merkezli davranış -------------------------------- #


def test_engine_is_id_centric(cfg):
    """Farklı track_id'ler birbirinin tamponunu KİRLETMEZ."""
    eng = DriverStateEngine(cfg)
    eng.model = _StubModel([_ds(phone=0.9)])  # her çağrıda phone=True döndür
    out1 = None
    for fi in range(eng.min_votes):
        out1 = eng.process(track_id=1, cabin_roi=object(), frame_idx=fi)
    assert out1.phone is True
    assert 2 not in eng.voters  # ID 2 için tampon hiç oluşmadı


def test_engine_accumulates_over_frames(cfg):
    """Aynı ID kareler boyunca eşiğe ulaşınca bayrak aktifleşir (önce pasif)."""
    eng = DriverStateEngine(cfg)
    eng.window, eng.min_votes = 16, 8
    eng.model = _StubModel([_ds(no_seatbelt=0.9)])
    results = [eng.process(1, object(), fi).no_seatbelt for fi in range(8)]
    assert results[:7] == [False] * 7  # eşik altında pasif
    assert results[7] is True  # 8. oyla aktif


def test_engine_prune_drops_stale_tracks(cfg):
    eng = DriverStateEngine(cfg)
    eng.max_age = 30
    eng.model = _StubModel([DriverState()])
    eng.process(1, object(), frame_idx=0)
    assert 1 in eng.voters
    eng.prune(frame_idx=100)  # 100 - 0 > 30 → düşmeli
    assert 1 not in eng.voters


def test_engine_aux_flags_fused_before_voting(cfg):
    """Stage-1 nesne kanıtı (aux_flags) ham tahmine OR'lanır ve oylamadan geçer."""
    eng = DriverStateEngine(cfg)
    eng.window, eng.min_votes = 8, 4
    eng.model = _StubModel([DriverState()])  # model hiç bayrak üretmiyor
    out = None
    for fi in range(4):  # yalnız aux kanıtıyla 4 oy → aktif
        out = eng.process(1, object(), fi, aux_flags={"phone": 0.55})
    assert out.phone is True
    assert out.confidence["phone"] > 0.0


def test_engine_reads_voting_config_from_models_driver_state(cfg):
    """Config models.driver_state.voting.* gerçekten okunuyor (varsayılan 16/8/30)."""
    eng = DriverStateEngine(cfg)
    assert eng.window == cfg.get("models.driver_state.voting.window", 16)
    assert eng.min_votes == cfg.get("models.driver_state.voting.min_votes", 8)
    assert eng.max_age == cfg.get("models.driver_state.voting.max_age", 30)
