"""GRUP I3-complexity-hw: karmaşıklık (calibration maybe_fit refit-throttle) +
donanım/ağ dayanıklılığı (kamera kopma/yeniden-bağlanma/timeout, YOLO cihaz
çalışma-zamanı fallback). Hepsi MOCK/CPU — gerçek model/kamera GEREKMEZ.

- CL: MetricSpeedEstimator.maybe_fit() artık kare-başı O(N) polyfit yapmaz; ilk
  eşik geçişinde uydurur, sonra yalnız `calib_refit_every` yeni örnekte yeniden
  uydurur. DAVRANIŞ-KORUYAN (is_ready/ppm aynı kalitede, daha az iş).
- HW-002: YOLO26Detector._track() seçili cihaz çalışma-zamanında çökerse bir KEZ
  CPU'ya düşer (tek kare hatası akışı durdurmaz); zaten CPU'daysa hata sızar.
- HW-001: StreamManager canlı kamera (non-file) kopma/timeout'unda cap'i yeniden
  açar (üstel backoff); dosya kaynağında davranış değişmez (başa sar).
"""

from __future__ import annotations

import os

os.environ.setdefault("ROADGUARD_AUTOSTART", "0")
os.environ.setdefault("ROADGUARD_CAMERA_PROBE", "0")
os.environ.setdefault("AI_MODE", "mock")

import types  # noqa: E402

import numpy as np  # noqa: E402

from roadguard.config import load_config  # noqa: E402
from roadguard.detection.yolo import YOLO26Detector  # noqa: E402
from roadguard.speed.calibration import MetricSpeedEstimator  # noqa: E402
from services.inference_api.state import StreamManager  # noqa: E402


# --------------------------------------------------------------------------- #
# CL: calibration maybe_fit refit-throttle
# --------------------------------------------------------------------------- #
def _settings(**kw):
    base = dict(calib_min_samples=4, calib_refit_every=10)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _add_samples(est: MetricSpeedEstimator, n: int, y0: float = 100.0) -> None:
    # düzgün ppm(y) örnekleri (lineer) — fit deterministik/davranış-korur
    for k in range(n):
        est.scale.add(y0 + k, 50.0 + 0.1 * k, weight=1.0)


def test_maybe_fit_does_not_refit_before_threshold():
    """İlk min_samples eşiği geçilince uydurulur; sonraki birkaç örnekte (refit_every
    altında) fit() YENİDEN ÇAĞRILMAZ — kare-başı O(N) polyfit önlenir."""
    est = MetricSpeedEstimator(_settings())
    calls = {"n": 0}
    orig_fit = est.scale.fit

    def _counting_fit():
        calls["n"] += 1
        return orig_fit()

    est.scale.fit = _counting_fit  # type: ignore[method-assign]

    # min_samples (4) altında: hiç fit yok
    _add_samples(est, 3)
    est.maybe_fit()
    assert calls["n"] == 0
    assert est.scale.is_ready is False

    # eşiği geç (4 örnek): bir kez uydurulur, is_ready True
    _add_samples(est, 1)
    est.maybe_fit()
    assert calls["n"] == 1
    assert est.scale.is_ready is True

    # refit_every (10) altında ek 5 örnek + birçok maybe_fit çağrısı → YENİ fit yok
    for _ in range(20):
        est.maybe_fit()
    _add_samples(est, 5)
    for _ in range(20):
        est.maybe_fit()
    assert calls["n"] == 1  # hâlâ tek fit


def test_maybe_fit_refits_after_enough_new_samples():
    """Son uydurmadan bu yana refit_every yeni örnek birikince yeniden uydurulur."""
    est = MetricSpeedEstimator(_settings())
    calls = {"n": 0}
    orig_fit = est.scale.fit

    def _counting_fit():
        calls["n"] += 1
        return orig_fit()

    est.scale.fit = _counting_fit  # type: ignore[method-assign]

    _add_samples(est, 4)
    est.maybe_fit()
    assert calls["n"] == 1
    # refit_every (10) yeni örnek ekle → ikinci fit tetiklenir
    _add_samples(est, 10)
    est.maybe_fit()
    assert calls["n"] == 2


def test_maybe_fit_behavior_preserved_ppm_matches_eager():
    """DAVRANIŞ-KORUYAN: throttle'lı estimator'ın uyguladığı ppm(y), aynı örneklerle
    her çağrıda eager fit yapan kontrole birebir eşit (sonuç değişmez, yalnız maliyet)."""
    est = MetricSpeedEstimator(_settings(calib_refit_every=10))
    _add_samples(est, 14)
    est.maybe_fit()  # eşik + (14-4>=10) → güncel uydurma
    ppm_throttled = est.scale.ppm_at(150.0)

    ctrl = MetricSpeedEstimator(_settings(calib_refit_every=1))
    _add_samples(ctrl, 14)
    ctrl.maybe_fit()
    ppm_eager = ctrl.scale.ppm_at(150.0)

    assert ppm_throttled is not None and ppm_eager is not None
    assert abs(ppm_throttled - ppm_eager) < 1e-9


# --------------------------------------------------------------------------- #
# HW-002: YOLO cihaz çalışma-zamanı fallback
# --------------------------------------------------------------------------- #
def _bare_detector(device: str) -> YOLO26Detector:
    """__init__'i (YOLO yükler) atlayıp _track için gereken alanları elle kur."""
    det = YOLO26Detector.__new__(YOLO26Detector)
    det.device = device
    det.conf = 0.35
    det.iou = 0.45
    det.imgsz = 640
    det.tracker_yaml = "bytetrack.yaml"
    return det


class _FakeModel:
    """track() ilk çağrıda (CPU dışı cihaz) patlar, CPU'da başarılı döner."""

    def __init__(self):
        self.calls = []

    def track(self, frame, **kw):
        self.calls.append(kw.get("device"))
        if kw.get("device") != "cpu":
            raise RuntimeError("no kernel image is available for execution")
        return ["ok"]


def test_track_falls_back_to_cpu_on_runtime_error():
    """MPS/CUDA çalışma-zamanı hatası → bir KEZ CPU'ya düşer, sonuç döner, cihaz CPU kalır."""
    det = _bare_detector("mps")
    det.model = _FakeModel()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    out = det._track(frame)
    assert out == ["ok"]
    assert det.device == "cpu"
    assert det.model.calls == ["mps", "cpu"]  # önce mps (patlar), sonra cpu (başarı)


def test_track_reraises_when_already_cpu():
    """Zaten CPU'dayken hata → gizlenecek yedek yok, hata sızar."""
    det = _bare_detector("cpu")

    class _AlwaysFail:
        def track(self, frame, **kw):
            raise RuntimeError("boom")

    det.model = _AlwaysFail()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        det._track(frame)
        raised = False
    except RuntimeError:
        raised = True
    assert raised is True


# --------------------------------------------------------------------------- #
# HW-001: StreamManager canlı kamera reconnect / timeout
# --------------------------------------------------------------------------- #
class _FakeCap:
    def __init__(self, opened=True):
        self._opened = opened
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self._opened

    def release(self):
        self.released = True

    def set(self, prop, val):
        self.set_calls.append((prop, val))
        return True


def _mgr():
    cfg = load_config()
    m = StreamManager(cfg)
    # backoff'u testte hızlandır
    m._reconnect_backoff = 0.0
    m._reconnect_backoff_max = 0.0
    m._max_reconnect = 3
    return m


def test_open_capture_sets_timeout_only_for_live_camera(monkeypatch):
    """Canlı kamera (int indeks) → timeout property'leri set edilir; dosya (str) → edilmez."""
    fake = _FakeCap(opened=True)
    monkeypatch.setattr("services.inference_api.state.cv2.VideoCapture", lambda src: fake)
    m = _mgr()
    # canlı kamera: indeks 0 → timeout set çağrıları olur (property mevcutsa)
    fake.set_calls.clear()
    m._open_capture(0)
    # dosya: str kaynak → timeout set edilmez
    file_calls_before = len(fake.set_calls)
    m._open_capture("video.mp4")
    assert len(fake.set_calls) == file_calls_before  # dosyada ek set yok


def test_reconnect_returns_new_cap_on_success(monkeypatch):
    """Kopan cap release edilir, yeni açık cap döner; _running True iken backoff geçilir."""
    old = _FakeCap(opened=True)
    new = _FakeCap(opened=True)
    monkeypatch.setattr("services.inference_api.state.cv2.VideoCapture", lambda src: new)
    m = _mgr()
    m._running = True
    res = m._reconnect(old, 0, attempt=1)
    assert old.released is True
    assert res is new


def test_reconnect_returns_none_when_reopen_fails(monkeypatch):
    """Yeniden açılan cap isOpened()=False → None döner (çağıran sayaca göre durur)."""
    old = _FakeCap(opened=True)
    dead = _FakeCap(opened=False)
    monkeypatch.setattr("services.inference_api.state.cv2.VideoCapture", lambda src: dead)
    m = _mgr()
    m._running = True
    res = m._reconnect(old, 0, attempt=1)
    assert res is None
    assert dead.released is True  # başarısız açılan cap serbest bırakılır


def test_reconnect_aborts_immediately_when_stopped(monkeypatch):
    """Backoff sırasında _running False ise reconnect denenmeden None döner."""
    opens = {"n": 0}

    def _vc(src):
        opens["n"] += 1
        return _FakeCap(opened=True)

    monkeypatch.setattr("services.inference_api.state.cv2.VideoCapture", _vc)
    m = _mgr()
    m._running = False  # durdurulmuş
    res = m._reconnect(_FakeCap(), 0, attempt=1)
    assert res is None
    assert opens["n"] == 0  # yeni cap açma denemesi YOK


def test_worker_reconnects_live_camera_then_stops(monkeypatch):
    """Canlı kamera read()=False döndürünce worker yeniden bağlanmayı dener; max
    denemede pes eder ve _running False olur (sonsuz döngü yok). Pipeline mock'lanır."""

    class _NeverReadCap:
        def __init__(self):
            self._open = True
            self.reads = 0

        def isOpened(self):
            return self._open

        def read(self):
            self.reads += 1
            return False, None  # canlı kamera koptu

        def get(self, prop):
            return 30.0

        def set(self, prop, val):
            return True

        def release(self):
            self._open = False

    caps = [_NeverReadCap() for _ in range(10)]
    idx = {"i": 0}

    def _vc(src):
        c = caps[min(idx["i"], len(caps) - 1)]
        idx["i"] += 1
        return c

    monkeypatch.setattr("services.inference_api.state.cv2.VideoCapture", _vc)

    # Pipeline'ı tamamen mock'la (ağır model yüklenmesin)
    class _FakePipeline:
        def __init__(self, cfg):
            self.fps = 0.0
            self.speed = types.SimpleNamespace(fps=0.0)
            self.emitter = types.SimpleNamespace(
                on_event=lambda *a: None, on_annotation=lambda *a: None
            )

        def process_frame(self, frame, i):
            return types.SimpleNamespace(tracks=[], signs=[], scene={}), []

    monkeypatch.setattr("services.inference_api.state.Pipeline", _FakePipeline)

    m = _mgr()
    m.source = 0  # canlı kamera (int)
    m._running = True
    m._worker()
    # Tüm reconnect denemeleri tükendi → temiz duruş
    assert m._running is False


def test_worker_invalid_source_stops_cleanly(monkeypatch):
    """Açılamayan kaynak → worker pipeline kurar ama cap açılmaz → _running False (mevcut sözleşme)."""

    class _FakePipeline:
        def __init__(self, cfg):
            self.fps = 0.0
            self.speed = types.SimpleNamespace(fps=0.0)
            self.emitter = types.SimpleNamespace(
                on_event=lambda *a: None, on_annotation=lambda *a: None
            )

    monkeypatch.setattr("services.inference_api.state.Pipeline", _FakePipeline)
    monkeypatch.setattr(
        "services.inference_api.state.cv2.VideoCapture",
        lambda src: _FakeCap(opened=False),
    )
    m = _mgr()
    m.source = "yok_boyle_dosya.mp4"
    m._running = True
    m._worker()
    assert m._running is False
