"""I2 — bellek-sızıntısı + concurrency birim testleri (MEM-001..006, DF-001/002,
CA-001/002). Model gerektirmez (mock/CPU). İki sınıf davranış doğrulanır:

  1) UZUN-AKIŞ bellek SINIRLI: çok sayıda benzersiz track ekledikten sonra prune
     çağrısı per-track sözlükleri max_age'den eski track'ler için DÜŞÜRÜR (boyut
     düşer). DAVRANIŞ-KORUYAN: yalnız max_age'den eski (artık görünmeyen) track
     durumu düşer; max_age grace içindeki track KORUNUR.
  2) EŞ-ZAMANLI oku/yaz TUTARLI: emitter (CA-001) ve accumulator (CA-002) bir
     yazar + bir okuyucu iş parçacığı altında istisna ÜRETMEZ ("deque/dict mutated
     during iteration" yarışı kapalı).
"""

from __future__ import annotations

import threading

import numpy as np

from roadguard.accumulator import Accumulator
from roadguard.events.emitter import EventEmitter
from roadguard.plate.reader import PlateReader
from roadguard.schema import AnnotationFrame, BBox, make_event
from roadguard.stability.class_vote import TrackClassVoter
from roadguard.stability.state_machine import StabilityTracker


class _FakeOCR:
    def read(self, plate_roi, vehicle_crop=None):
        return ("", 0.0)  # konsensüs yok → durum 'pending' kalır, oy havuzu birikir


def _bbox():
    return BBox(x1=0, y1=0, x2=10, y2=10, conf=0.9, cls="car")


class _Cfg:
    """Minimal config: .get(key, default) — gerçek load_config gerektirmez."""

    def __init__(self, data=None):
        self._d = data or {}

    def get(self, key, default=None):
        return self._d.get(key, default)


# --------------------------------------------------------------------------- #
# 1) UZUN-AKIŞ — bellek sınırlı (prune sonrası dict boyutu düşer)
# --------------------------------------------------------------------------- #
def test_accumulator_prune_bounds_memory(cfg):
    acc = Accumulator(cfg)
    # 200 benzersiz track'i frame 0'da ekle, sonra çok ileri bir frame'de prune et.
    for tid in range(200):
        acc.update_track(tid, frame_idx=0, bbox=_bbox(), vehicle_class="car")
    assert len(acc.tracks) == 200
    acc.prune(frame_idx=1000, max_age=30)  # hepsi max_age'den (30) çok eski
    assert len(acc.tracks) == 0  # bellek serbest bırakıldı


def test_accumulator_prune_keeps_recent(cfg):
    acc = Accumulator(cfg)
    acc.update_track(1, frame_idx=100, bbox=_bbox(), vehicle_class="car")  # taze
    acc.update_track(2, frame_idx=10, bbox=_bbox(), vehicle_class="car")  # eski
    acc.prune(frame_idx=120, max_age=30)  # track1 grace içinde (20), track2 değil (110)
    assert 1 in acc.tracks and 2 not in acc.tracks  # davranış-koruyan


def test_class_voter_prune_aged_bounds_memory():
    v = TrackClassVoter(_Cfg({"tracking.class_vote": {}}))
    for tid in range(150):
        v.update(tid, "car", 0.9, frame_idx=0)
    assert len(v._votes) == 150
    v.prune_aged(frame_idx=500)  # max_age=30 default → hepsi düşer
    assert len(v._votes) == 0 and len(v._last_seen) == 0


def test_class_voter_prune_aged_keeps_recent():
    v = TrackClassVoter(_Cfg({"tracking.class_vote": {}}))
    v.update(7, "car", 0.9, frame_idx=100)
    v.update(8, "bus", 0.9, frame_idx=10)
    v.prune_aged(frame_idx=120)  # 7 grace içinde, 8 değil
    assert v.stable_class(7) == "car" and v.stable_class(8) is None


def test_stability_prune_aged_bounds_memory():
    st = StabilityTracker(_Cfg({}))
    for tid in range(120):
        st.update(f"{tid}:speed.rel", True, frame_idx=0)
        st.update(f"{tid}:speed.swerve", False, frame_idx=0)
    assert len(st._windows) == 240
    st.prune_aged(frame_idx=500)  # max_age=30 default → hepsi düşer
    assert len(st._windows) == 0 and len(st._track_last_seen) == 0


def test_stability_prune_aged_keeps_recent_and_nontrack_keys():
    st = StabilityTracker(_Cfg({}))
    st.update("5:speed.rel", True, frame_idx=100)  # taze track
    st.update("9:speed.rel", True, frame_idx=10)  # eski track
    st.update("global_flag", True, frame_idx=10)  # track-bağlı OLMAYAN anahtar
    st.prune_aged(frame_idx=120)
    assert "5:speed.rel" in st._windows  # grace içinde korunur
    assert "9:speed.rel" not in st._windows  # eski düşer
    assert "global_flag" in st._windows  # track-prefix yok → her zaman korunur


def test_stability_prune_aged_noop_without_frame_idx():
    # frame_idx geçilmezse _track_last_seen boş → prune hiçbir şey düşürmez (geriye uyum)
    st = StabilityTracker(_Cfg({}))
    st.update("3:speed.rel", True)  # frame_idx YOK
    st.prune_aged(frame_idx=10_000)
    assert "3:speed.rel" in st._windows


def test_plate_reader_prune_bounds_memory(cfg):
    # OCR enjekte → model/LP yüklenmez (saf CPU). Sweet-spot tüm kareyi kapsar (default).
    reader = PlateReader(cfg, ocr=_FakeOCR())
    roi = np.zeros((40, 80, 3), dtype=np.uint8)  # min_pixel_height üstü
    vbbox = BBox(x1=10, y1=10, x2=300, y2=300, conf=0.9, cls="car")
    for tid in range(100):
        reader.update(tid, roi, vbbox, (360, 640, 3), frame_idx=0)
    # per-track sözlüklerin EN AZ BİRİ doldu (oy havuzu/durum/last_seen birikti)
    assert len(reader._state) == 100
    assert len(reader._last_seen) == 100
    reader.prune(frame_idx=1000)  # max_age=30 default → hepsi eski
    assert len(reader._state) == 0
    assert len(reader._pools) == 0
    assert len(reader._reads_since_eval) == 0
    assert len(reader._last_seen) == 0


def test_plate_reader_prune_keeps_recent(cfg):
    reader = PlateReader(cfg, ocr=_FakeOCR())
    roi = np.zeros((40, 80, 3), dtype=np.uint8)
    vbbox = BBox(x1=10, y1=10, x2=300, y2=300, conf=0.9, cls="car")
    reader.update(1, roi, vbbox, (360, 640, 3), frame_idx=100)  # taze
    reader.update(2, roi, vbbox, (360, 640, 3), frame_idx=10)  # eski
    reader.prune(frame_idx=120)  # 1 grace içinde, 2 değil
    assert 1 in reader._state and 2 not in reader._state  # davranış-koruyan


def test_plate_reader_prune_noop_without_frame_idx(cfg):
    # frame_idx geçilmezse _last_seen boş → prune hiçbir şeyi düşürmez (geriye uyum)
    reader = PlateReader(cfg, ocr=_FakeOCR())
    roi = np.zeros((40, 80, 3), dtype=np.uint8)
    vbbox = BBox(x1=10, y1=10, x2=300, y2=300, conf=0.9, cls="car")
    reader.update(3, roi, vbbox, (360, 640, 3))  # frame_idx YOK
    reader.prune(frame_idx=10_000)
    assert 3 in reader._state


# --------------------------------------------------------------------------- #
# 2) EŞ-ZAMANLI oku/yaz — tutarlı (CA-001 emitter, CA-002 accumulator)
# --------------------------------------------------------------------------- #
def test_emitter_concurrent_read_write_consistent():
    em = EventEmitter(maxlen=500)
    errors: list[Exception] = []
    stop = threading.Event()

    def writer():
        try:
            for i in range(5000):
                em.emit_event(make_event(i, "DETECTION_UPDATE", {"n": i}))
                em.emit_annotation(AnnotationFrame(frame_id=i, tracks=[]))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                _ = em.recent_events(50)  # snapshot — append ile yarışmamalı
                _ = em.latest_annotation()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    tw, tr = threading.Thread(target=writer), threading.Thread(target=reader)
    tr.start()
    tw.start()
    tw.join()
    tr.join()
    assert not errors, f"eş-zamanlı erişimde istisna: {errors}"
    assert len(em.recent_events(50)) == 50  # deque maxlen ile sınırlı (bellek sabit)


def test_accumulator_concurrent_update_prune_read_consistent(cfg):
    acc = Accumulator(cfg)
    errors: list[Exception] = []
    stop = threading.Event()

    def writer():
        try:
            for i in range(4000):
                acc.update_track(i % 50, frame_idx=i, bbox=_bbox(), vehicle_class="car")
                if i % 10 == 0:
                    acc.prune(frame_idx=i, max_age=30)  # eş-zamanlı del
        except Exception as e:  # noqa: BLE001
            errors.append(e)
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                _ = acc.active_tracks()  # list(dict.values()) prune'la yarışmamalı
                _ = acc.get(0)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    tw, tr = threading.Thread(target=writer), threading.Thread(target=reader)
    tr.start()
    tw.start()
    tw.join()
    tr.join()
    assert not errors, f"eş-zamanlı erişimde istisna: {errors}"


def test_emitter_memory_bounded_by_maxlen():
    # CA-001 yanında bellek sınırı: deque(maxlen) uzun akışta sabit kalır.
    em = EventEmitter(maxlen=100)
    for i in range(10_000):
        em.emit_event(make_event(i, "DETECTION_UPDATE", {}))
    assert len(em.events) == 100  # sınırsız büyümez
