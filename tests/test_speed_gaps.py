"""Kapsam-kapatan birim testleri: speed (estimator + calibration), swerving, sign.

INCELEME BULGULARI'ndaki test_gaps + bug'lar için kenar-durum / hata-yolu / dal kapsamı.
Hepsi MOCK-mod (model gerektirmez): saf hesap + sentetik bbox. K-004: mevcut davranış
korunur; bu testler eklenen prune/dal davranışını ve daha önce test edilmeyen yolları
 kanıtlar.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from roadguard.config import load_config
from roadguard.schema import BBox
from roadguard.speed.calibration import (
    KalmanSpeed1D,
    MetricSpeedEstimator,
    ScaleField,
    SpeedTrack,
    plate_ppm,
)
from roadguard.speed.estimator import SpeedEstimator

FRAME = (720, 1280, 3)


def _car(x_left: float, y2: float = 400.0, width: float = 180.0, height: float = 130.0) -> BBox:
    return BBox(x1=x_left, y1=y2 - height, x2=x_left + width, y2=y2, conf=0.9, cls="car")


def _tw_bbox(cy_norm: float, h: int = 720) -> BBox:
    """Merkez-y'si tam `cy_norm` (normalize) olan bbox (tripwire cy_norm = center_y/h)."""
    cy = cy_norm * h
    return BBox(x1=300, y1=cy - 20, x2=380, y2=cy + 20, conf=0.9, cls="car")


def _metric_cfg():
    cfg = load_config()
    cfg.data["speed"]["mode"] = "metric"
    return cfg


def _settings(**over):
    base = dict(
        plate_width_m=0.520,
        plate_aspect_tolerance=0.35,
        vehicle_width_m={"car": 1.80, "truck": 2.50},
        vehicle_ppm_weight=0.25,
        calib_min_samples=6,
        speed_window_frames=6,
        speed_max_accel_mps2=8.0,
        speed_kalman_q=3.0,
        speed_kalman_r=8.0,
        speed_ema_alpha=0.0,
        speed_metric_max_kmh=200.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ====================================================================== #
# 1) SpeedEstimator.prune — bellek sızıntısı (bug fix) + tripwire bayat-durum
# ====================================================================== #
def test_prune_evicts_stale_track_state(cfg):
    cfg.data["speed"]["mode"] = "metric"
    cfg.data["speed"]["max_age"] = 5
    est = SpeedEstimator(cfg)
    est.fps = 25.0
    for i in range(16):
        est.update(1, _car(100.0 + 50.0 * i), i, FRAME)
    # track 1 için durum birikti
    assert 1 in est._hist and 1 in est._tracks and 1 in est._last_seen
    assert 1 in est._metric._kalman  # metric Kalman durumu da kuruldu
    # Çok ileri bir kareye prune → max_age aşıldı, hepsi düşmeli
    est.prune(frame_idx=100)
    assert 1 not in est._hist
    assert 1 not in est._lat_hist
    assert 1 not in est._tracks
    assert 1 not in est._last_speed
    assert 1 not in est._last_seen
    assert 1 not in est._metric._kalman  # forward to metric.prune


def test_prune_keeps_recent_track(cfg):
    cfg.data["speed"]["mode"] = "disabled"
    cfg.data["speed"]["max_age"] = 30
    est = SpeedEstimator(cfg)
    est.update(7, _car(100.0), frame_idx=100, frame_shape=FRAME)
    est.prune(frame_idx=110)  # 110-100=10 <= 30 → kalır
    assert 7 in est._hist and 7 in est._last_seen


def test_prune_resets_tripwire_stale_state(cfg):
    """line_a geçti ama line_b'den önce kayboldu → s['a'] kalıcı; prune temizlemeli."""
    cfg.data["speed"]["mode"] = "tripwire"
    cfg.data["speed"]["max_age"] = 5
    est = SpeedEstimator(cfg)
    est.fps = 30.0
    # line_a(0.40) geç ama line_b(0.70) geçme
    est.update(3, _tw_bbox(0.30), 0, FRAME)
    est.update(3, _tw_bbox(0.45), 1, FRAME)  # a kuruldu
    assert est._tw[3]["a"] is not None and est._tw[3]["b"] is None
    est.prune(frame_idx=100)  # uzun süre görülmedi
    assert 3 not in est._tw  # bayat durum temizlendi (recycled-id km/h hatası önlendi)


# ====================================================================== #
# 2) Tripwire dal kapsamı
# ====================================================================== #
def test_tripwire_crosses_a_not_b_no_speed(cfg):
    cfg.data["speed"]["mode"] = "tripwire"
    est = SpeedEstimator(cfg)
    est.fps = 30.0
    v = None
    for i, cy in enumerate([0.30, 0.45, 0.50, 0.55]):  # a geçti, b(0.70) geçmedi
        v = est.update(1, _tw_bbox(cy), i, FRAME).value_kmh
    assert v is None
    assert est._tw[1]["a"] is not None and est._tw[1]["b"] is None


def test_tripwire_wrong_direction_no_speed(cfg):
    """Yukarı doğru (uzaklaşan) hareket: prev < line koşulu hiç sağlanmaz → hız yok."""
    cfg.data["speed"]["mode"] = "tripwire"
    est = SpeedEstimator(cfg)
    est.fps = 30.0
    v = None
    for i, cy in enumerate([0.90, 0.75, 0.60, 0.45, 0.30]):  # aşağıdan yukarı
        v = est.update(1, _tw_bbox(cy), i, FRAME).value_kmh
    assert v is None
    assert est._tw[1]["a"] is None


def test_tripwire_independent_track_state(cfg):
    cfg.data["speed"]["mode"] = "tripwire"
    est = SpeedEstimator(cfg)
    est.fps = 30.0
    # track 1 tam geçiş yapsın (a sonra b)
    for i in range(21):
        cy = min(0.35 + 0.02 * i, 0.95)
        est.update(1, _tw_bbox(cy), i, FRAME)
    # track 2 hiç line_a geçmesin
    for i in range(5):
        est.update(2, _tw_bbox(0.30), i, FRAME)
    assert est._tw[1]["kmh"] is not None  # bağımsız: t1 hesapladı
    assert est._tw[2]["kmh"] is None  # t2 ayrı durum makinesi, hiç hesaplamadı


# ====================================================================== #
# 3) ipm modu — tamamen test edilmemiş yollar
# ====================================================================== #
def test_ipm_disabled_module_falls_back(cfg):
    """optional_modules.homography_ipm kapalı → disabled davranışı + tek-uyarı latch."""
    cfg.data["speed"]["mode"] = "ipm"
    cfg.data.setdefault("optional_modules", {})["homography_ipm"] = False
    est = SpeedEstimator(cfg)
    assert est._ipm_enabled is False
    s = est.update(1, _car(100.0), 0, FRAME)
    assert s.mode == "disabled" and s.value_kmh is None
    assert est._ipm_warned is True  # uyarı latch'lendi
    # ikinci kare yine disabled, latch tekrar uyarmaz (durum True kalır)
    s2 = est.update(1, _car(150.0), 1, FRAME)
    assert s2.mode == "disabled" and est._ipm_warned is True


def test_ipm_enabled_no_calibration_returns_none(cfg):
    """Modül açık ama calibration_file yok → ipm_speed None döner, mod 'ipm' kalır (uyarı yok)."""
    cfg.data["speed"]["mode"] = "ipm"
    cfg.data["speed"]["calibration_file"] = None
    cfg.data.setdefault("optional_modules", {})["homography_ipm"] = True
    est = SpeedEstimator(cfg)
    assert est._ipm_enabled is True
    s = est.update(1, _car(100.0), 0, FRAME)
    assert s.mode == "ipm" and s.value_kmh is None
    assert est._ipm_warned is False  # başarılı import → uyarı latch'i kurulmadı


def test_ipm_enabled_exception_swallowed_when_import_fails(cfg, monkeypatch):
    """Modül açık ama ipm_speed patlıyor → exception yutulur, disabled'a düşer, tek-uyarı."""
    cfg.data["speed"]["mode"] = "ipm"
    cfg.data.setdefault("optional_modules", {})["homography_ipm"] = True
    est = SpeedEstimator(cfg)
    import roadguard.optional.homography_ipm as mod

    def _boom(*a, **k):
        raise RuntimeError("ipm patladi")

    monkeypatch.setattr(mod, "ipm_speed", _boom)
    s = est.update(1, _car(100.0), 0, FRAME)
    assert s.mode == "disabled" and s.value_kmh is None
    assert est._ipm_warned is True  # exception yolu uyarı latch'ini kurdu
    # ikinci kare yine disabled, latch tekrar uyarmaz
    s2 = est.update(1, _car(150.0), 1, FRAME)
    assert s2.mode == "disabled" and est._ipm_warned is True


# ====================================================================== #
# 4) metric: lazy re-init guard + warmup göreli bayrak yine üretilir
# ====================================================================== #
def test_metric_lazy_reinit_when_metric_none(cfg):
    cfg.data["speed"]["mode"] = "metric"
    est = SpeedEstimator(cfg)
    est._metric = None  # zorla sıfırla → _metric_update lazy yeniden kurmalı
    est.update(1, _car(100.0), 0, FRAME)
    assert est._metric is not None


# ====================================================================== #
# 5) swerving: warmup kapısı, swerve_enabled=False, maxlen-rebuild, boundary
# ====================================================================== #
def _sw_cfg(cfg):
    cfg.data.setdefault("speed", {})["mode"] = "disabled"
    return cfg


def test_swerving_disabled_returns_false(cfg):
    _sw_cfg(cfg)
    cfg.data["speed"]["swerving"]["enabled"] = False
    est = SpeedEstimator(cfg)
    import math

    states = []
    for i in range(90):
        cx = 500 + 40.0 * math.sin(2 * math.pi * i / 30)
        states.append(est.update(1, _bbox(cx), i, (1080, 1920, 3)))
    assert not any(s.swerving for s in states)


def test_swerving_warmup_no_flag(cfg):
    _sw_cfg(cfg)
    est = SpeedEstimator(cfg)
    # maxlen = max(8, 3.0*30)=90; warmup kapısı len < 90//3=30 → False
    import math

    s = None
    for i in range(10):  # 10 < 30 → ısınma, bayrak olmamalı
        cx = 500 + 60.0 * math.sin(2 * math.pi * i / 4)  # agresif yalpalama bile
        s = est.update(1, _bbox(cx), i, (1080, 1920, 3))
    assert s.swerving is False
    assert len(est._lat_hist[1]) == 10  # geçmiş birikiyor


def test_swerving_maxlen_rebuild_preserves_history(cfg):
    _sw_cfg(cfg)
    est = SpeedEstimator(cfg)
    est.fps = 30.0
    for i in range(40):
        est.update(1, _bbox(500.0), i, (1080, 1920, 3))
    before = len(est._lat_hist[1])
    assert est._lat_hist[1].maxlen == 90
    # window_s değiştir → maxlen değişir, deque yeniden kurulur ama içerik korunur
    est.swerve_window_s = 1.0  # maxlen = max(8, 30) = 30
    est.update(1, _bbox(500.0), 40, (1080, 1920, 3))
    assert est._lat_hist[1].maxlen == 30
    # eski 40 + yeni 1 = 41 öğe ama maxlen 30 → son 30 tutulur
    assert len(est._lat_hist[1]) == 30
    assert before == 40


def test_swerving_single_overshoot_one_reversal_eliminated(cfg):
    """Tek şerit değişimi + overshoot = en çok 1 dönüş → min_flips(2) altında → bayrak yok."""
    _sw_cfg(cfg)
    est = SpeedEstimator(cfg)
    est.fps = 30.0
    # düz → sağa kay → küçük geri oturma (overshoot) → düz: 1 dönüş üretir
    path = [500.0] * 30
    path += [500.0 + min((i + 1) * 8.0, 200.0) for i in range(30)]  # sağa
    path += [700.0 - 15.0] * 30  # küçük overshoot geri (1 dönüş)
    states = [est.update(1, _bbox(cx), i, (1080, 1920, 3)) for i, cx in enumerate(path)]
    assert not any(s.swerving for s in states)


# ====================================================================== #
# 6) ScaleField kenar durumları
# ====================================================================== #
def test_scalefield_ppm_at_not_fitted_returns_none():
    sf = ScaleField(min_samples=4)
    sf.add(400.0, 100.0)
    assert sf.ppm_at(400.0) is None  # fit() çağrılmadı


def test_scalefield_add_rejects_invalid():
    sf = ScaleField(min_samples=2)
    sf.add(100.0, None)  # None
    sf.add(100.0, 0.0)  # <=0
    sf.add(100.0, -5.0)  # negatif
    sf.add(100.0, float("inf"))  # non-finite
    sf.add(100.0, 50.0, weight=0.0)  # weight<=0
    assert sf.n_samples == 0
    sf.add(100.0, 50.0, weight=1.0)  # geçerli
    assert sf.n_samples == 1


def test_scalefield_fit_below_min_samples_returns_false():
    sf = ScaleField(min_samples=6)
    for _ in range(3):
        sf.add(400.0, 100.0)
    assert sf.fit() is False and sf.is_ready is False


def test_scalefield_negative_prediction_falls_back_to_median():
    """Eğim aşağı eğimli → büyük y'de tahmin <=0 → median_ppm'e düşer."""
    sf = ScaleField(min_samples=4)
    # ppm y arttıkça AZALSIN (negatif eğim): y=100→200 .. y=500→40
    for y, p in [(100, 200), (200, 160), (300, 120), (400, 80), (500, 40)]:
        sf.add(float(y), float(p))
    assert sf.fit() is True
    # çok büyük y'de doğru <=0 verir → median'a düşer (>0)
    out = sf.ppm_at(5000.0)
    assert out is not None and out > 0
    assert abs(out - sf._median_ppm) < 1e-6


def test_scalefield_outlier_rejection_refits():
    """Bir aykırı örnek enjekte et → ikinci polyfit (keep<n) tetiklenir, eğim düzelir."""
    sf = ScaleField(min_samples=4)
    for y in (100, 200, 300, 400, 500, 600, 700):
        sf.add(float(y), 0.5 * y, weight=1.0)  # ppm = 0.5*y
    sf.add(350.0, 9999.0, weight=1.0)  # absürt aykırı
    assert sf.fit() is True
    # aykırı atıldıysa eğim ~0.5'e yakın kalır
    assert abs(sf._slope - 0.5) < 0.1
    assert abs(sf.ppm_at(300.0) - 150.0) < 10.0


def test_scalefield_constant_branch_with_weights():
    """y std ~0 (tek y) → ağırlıklı sabit ppm dalı (np.average weights)."""
    sf = ScaleField(min_samples=4)
    sf.add(400.0, 100.0, weight=1.0)
    sf.add(400.0, 100.0, weight=1.0)
    sf.add(400.0, 200.0, weight=3.0)  # ağırlıklı ortalama 200'e yaklaşır
    sf.add(400.0, 200.0, weight=3.0)
    assert sf.fit() is True
    assert sf._slope == 0.0
    out = sf.ppm_at(400.0)
    assert 100.0 < out < 200.0  # ağırlıklı ortalama (200'e meyilli)


# ====================================================================== #
# 7) plate_ppm kenar durumları
# ====================================================================== #
def test_plate_ppm_none_and_zero_width_m():
    assert plate_ppm(None) is None
    box = BBox(x1=0, y1=0, x2=104, y2=24, conf=0.9, cls="plate")
    assert plate_ppm(box, plate_width_m=0.0) is None
    assert plate_ppm(box, plate_width_m=-1.0) is None


def test_plate_ppm_degenerate_box():
    tiny_w = BBox(x1=10, y1=0, x2=11, y2=24, conf=0.9, cls="plate")  # w=1
    assert plate_ppm(tiny_w) is None
    tiny_h = BBox(x1=0, y1=10, x2=104, y2=11, conf=0.9, cls="plate")  # h=1
    assert plate_ppm(tiny_h) is None


# ====================================================================== #
# 8) Homografi yolu — _step_meters / set_homography / estimate önceliği
# ====================================================================== #
class _FakeHomography:
    def __init__(self, is_valid=True, ground=None):
        self.is_valid = is_valid
        self._ground = ground  # callable veya None döndüren

    def to_ground(self, x, y):
        if self._ground is not None:
            return self._ground(x, y)
        return (x * 0.01, y * 0.01)  # 1px = 0.01m


def test_set_homography_rejects_invalid():
    m = MetricSpeedEstimator(_settings())
    m.set_homography(_FakeHomography(is_valid=False))
    assert m.homography is None  # geçersiz → atanmaz
    m.set_homography(None)
    assert m.homography is None


def test_set_homography_accepts_valid():
    m = MetricSpeedEstimator(_settings())
    h = _FakeHomography(is_valid=True)
    m.set_homography(h)
    assert m.homography is h


def test_estimate_uses_homography_priority():
    """Homografi varken ölçek hazır olmasa bile km/h üretilir (öncelik B kaynağı)."""
    m = MetricSpeedEstimator(_settings(speed_window_frames=6))
    m.set_homography(_FakeHomography(is_valid=True))  # 1px=0.01m
    assert m.scale.is_ready is False
    tr = SpeedTrack(1)
    # 100px/kare, fps eşdeğeri ts: 1px=0.01m → 100px=1.0m/kare; dt=1/25s → 25m/s=90km/h
    for i in range(8):
        tr.update((100.0 + 100.0 * i, 400.0), i / 25.0)
    val, is_cal = m.estimate(tr)
    assert is_cal is True
    assert val is not None and abs(val - 90.0) <= 1.0


def test_step_meters_homography_to_ground_none():
    m = MetricSpeedEstimator(_settings())
    m.set_homography(_FakeHomography(is_valid=True, ground=lambda x, y: None))
    assert m._step_meters((0.0, 0.0), (10.0, 10.0)) is None


# ====================================================================== #
# 9) _window_steps: ts None / dt<=0 atlama (dropped-frame / VFR)
# ====================================================================== #
def test_window_steps_skips_none_ts_and_nonpositive_dt():
    m = MetricSpeedEstimator(_settings(speed_window_frames=10))
    m.set_homography(_FakeHomography(is_valid=True))
    tr = SpeedTrack(1)
    tr.update((0.0, 400.0), 0.0)
    tr.update((100.0, 400.0), None)  # ts None → bu adım atlanır
    tr.update((200.0, 400.0), 0.1)
    tr.update((300.0, 400.0), 0.1)  # dt=0 → atlanır
    tr.update((400.0, 400.0), 0.2)
    steps = m._window_steps(tr)
    # geçerli adımlar: yalnız (0.1→0.2) çifti (None ve dt=0 elendi)
    assert len(steps) == 1
    dt, v = steps[0]
    assert dt == pytest.approx(0.1)


def test_window_steps_tail_slice_matches_full(cfg):
    """Perf-opt (kuyruk islice) davranış-koruma: window'dan uzun geçmişte sonuç aynı."""
    m = MetricSpeedEstimator(_settings(speed_window_frames=4))
    m.set_homography(_FakeHomography(is_valid=True))
    tr = SpeedTrack(1, maxlen=16)
    for i in range(16):
        tr.update((50.0 * i, 400.0), i / 25.0)
    steps = m._window_steps(tr)
    # window=4 → son 4 çift; her adım 50px=0.5m, dt=0.04 → 12.5 m/s
    assert len(steps) == 4
    for dt, v in steps:
        assert dt == pytest.approx(0.04)
        assert v == pytest.approx(12.5, abs=1e-6)


# ====================================================================== #
# 10) MetricSpeedEstimator.prune (dead-code idi) + KalmanSpeed1D yakınsama
# ====================================================================== #
def test_metric_prune_evicts_kalman_ema():
    m = MetricSpeedEstimator(_settings(speed_ema_alpha=0.3))
    m.set_homography(_FakeHomography(is_valid=True))
    for tid in (1, 2, 3):
        tr = SpeedTrack(tid)
        for i in range(6):
            tr.update((100.0 * i, 400.0), i / 25.0)
        m.estimate(tr)
    assert set(m._kalman) == {1, 2, 3} and set(m._ema) == {1, 2, 3}
    m.prune(active_ids={2})
    assert set(m._kalman) == {2} and set(m._ema) == {2}


def test_kalman_multistep_convergence_to_constant():
    k = KalmanSpeed1D(Q=3.0, R=8.0)
    k.update(0.0)  # init
    outs = [k.update(20.0) for _ in range(40)]
    assert outs[-1] == pytest.approx(20.0, abs=0.5)  # sabit ölçüme yakınsar
    assert all(
        a <= b + 1e-9 for a, b in zip(outs, outs[1:], strict=False)
    )  # monoton artış (sıçrama yok)


def test_kalman_higher_q_tracks_faster():
    slow = KalmanSpeed1D(Q=0.5, R=8.0)
    fast = KalmanSpeed1D(Q=10.0, R=8.0)
    slow.update(0.0)
    fast.update(0.0)
    s = slow.update(100.0)
    f = fast.update(100.0)
    assert f > s  # yüksek Q → ölçümü daha hızlı takip


# ---- swerving yardımcı bbox (test_swerving.py ile aynı biçim) ---- #
def _bbox(cx: float, w: float = 100.0) -> BBox:
    return BBox(x1=cx - w / 2, y1=400, x2=cx + w / 2, y2=500, conf=0.9, cls="car")
