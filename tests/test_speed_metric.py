"""Metric (oto-kalibrasyon) hız: sentetik kontrollü doğruluk + ısınma davranışı.

gercek_hiz_plani.md §8.1: bilinen ppm + bilinen piksel/kare hızı → beklenen km/h
analitik bilinir. Bu, formül ve BİRİM doğruluğunu kamera/sahne belirsizliği olmadan kanıtlar.
"""

from __future__ import annotations

from aura.config import load_config
from aura.schema import BBox
from aura.speed.calibration import KalmanSpeed1D, ScaleField, plate_ppm
from aura.speed.estimator import SpeedEstimator

FRAME = (720, 1280, 3)


def _metric_cfg():
    cfg = load_config()
    cfg.data["speed"]["mode"] = "metric"
    return cfg


def _car(x_left: float, y2: float = 400.0, width: float = 180.0, height: float = 130.0) -> BBox:
    return BBox(x1=x_left, y1=y2 - height, x2=x_left + width, y2=y2, conf=0.9, cls="car")


def test_metric_known_ppm_gives_known_kmh():
    """ppm = 180px / 1.80m = 100 px/m; dx=50px/kare; fps=25 → 0.5 m/kare → 12.5 m/s = 45 km/h."""
    cfg = _metric_cfg()
    est = SpeedEstimator(cfg)
    est.fps = 25.0
    val = None
    is_cal = False
    for i in range(16):
        s = est.update(1, _car(100.0 + 50.0 * i), i, FRAME)
        val, is_cal = s.value_kmh, s.is_calibrated
        assert s.mode == "metric"
    assert is_cal is True
    assert val is not None and abs(val - 45.0) <= 2.0  # analitik 45 km/h


def test_metric_warmup_no_claim_but_relative_flag():
    """Isınma (min_samples öncesi): km/h yok + is_calibrated False; göreli bayrak yine üretilir."""
    cfg = _metric_cfg()
    est = SpeedEstimator(cfg)
    est.fps = 25.0
    s = est.update(7, _car(100.0), 0, FRAME)
    assert s.mode == "metric" and s.value_kmh is None and s.is_calibrated is False


def test_metric_faster_vehicle_higher_kmh():
    """İki kat piksel hızı ≈ iki kat km/h (monotonluk)."""
    est_slow = SpeedEstimator(_metric_cfg())
    est_fast = SpeedEstimator(_metric_cfg())
    est_slow.fps = est_fast.fps = 25.0
    v_slow = v_fast = None
    for i in range(16):
        v_slow = est_slow.update(1, _car(100.0 + 25.0 * i), i, FRAME).value_kmh
        v_fast = est_fast.update(1, _car(100.0 + 60.0 * i), i, FRAME).value_kmh
    assert v_slow and v_fast and v_fast > v_slow


def test_scalefield_constant_ppm_fit():
    sf = ScaleField(min_samples=4)
    for _ in range(6):
        sf.add(400.0, 100.0, weight=1.0)
    assert sf.fit() is True and sf.is_ready
    assert abs(sf.ppm_at(400.0) - 100.0) < 1e-6


def test_scalefield_linear_ppm_by_y():
    """y arttıkça ppm artar (yakın = büyük ppm) — eğim yakalanır."""
    sf = ScaleField(min_samples=4)
    for y in (100, 200, 300, 400, 500, 600):
        sf.add(float(y), 0.5 * y, weight=1.0)  # ppm = 0.5*y
    assert sf.fit() is True
    assert abs(sf.ppm_at(300.0) - 150.0) < 1.0


def test_plate_ppm_frontal_and_rejected():
    # Cepheden (520/120≈4.33): kabul, ppm = genişlik / 0.520
    frontal = BBox(x1=0, y1=0, x2=104, y2=24, conf=0.9, cls="plate")  # 104/24≈4.33
    assert abs(plate_ppm(frontal) - 104.0 / 0.520) < 1e-6
    # Aşırı eğik (kare gibi): foreshortening → None
    skew = BBox(x1=0, y1=0, x2=40, y2=40, conf=0.9, cls="plate")
    assert plate_ppm(skew) is None


def test_metric_border_holds_last_speed():
    """Araç kadraj kenarına (50px ölü bölge) girince yeni hız hesaplanmaz; son geçerli
    hız tutulur (kadrajdan çıkarken bbox kırpılıp hız aniden düşmesin)."""
    cfg = _metric_cfg()
    cfg.data["speed"]["frame_margin_px"] = 50
    est = SpeedEstimator(cfg)
    est.fps = 25.0
    # Kadraj içinde ilerle → kalibre + geçerli hız
    last = None
    for i in range(16):
        s = est.update(1, _car(100.0 + 50.0 * i), i, FRAME)
        if s.value_kmh is not None:
            last = s.value_kmh
    assert last is not None
    # Şimdi araç sağ kenara değsin (x2 >= 1280-50): hız DONDURULMALI = son geçerli değer
    edge = BBox(x1=1200.0, y1=270.0, x2=1260.0, y2=400.0, conf=0.9, cls="car")  # x2=1260 ≥ 1230
    s_edge = est.update(1, edge, 16, FRAME)
    assert s_edge.value_kmh == last  # aniden düşmedi, son geçerli hız korundu


def test_metric_ema_smooths_more_than_kalman_alone():
    """EMA açıkken ardışık km/h değişimi (titreme) EMA kapalıya göre daha küçük olmalı."""

    def run(alpha):
        cfg = _metric_cfg()
        cfg.data["speed"]["speed_ema_alpha"] = alpha
        est = SpeedEstimator(cfg)
        est.fps = 25.0
        vals = []
        # değişken (titreşimli) piksel hızı: dx 30/70 dönüşümlü
        x = 100.0
        for i in range(24):
            x += 30.0 if i % 2 == 0 else 70.0
            s = est.update(1, _car(x), i, FRAME)
            if s.value_kmh is not None:
                vals.append(s.value_kmh)
        diffs = [abs(b - a) for a, b in zip(vals, vals[1:])]
        return sum(diffs) / len(diffs) if diffs else 0.0

    assert run(0.2) < run(1.0)  # düşük alpha (güçlü EMA) daha az dalgalanma


def test_kalman_damps_jitter():
    k = KalmanSpeed1D(Q=3.0, R=8.0)
    assert k.update(10.0) == 10.0  # ilk ölçüm aynen
    out = k.update(40.0)  # ani sıçrama → tam takip etmez (yumuşatma)
    assert 10.0 < out < 40.0
