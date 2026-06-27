"""Pipeline uçtan-uca birim testleri (mock-mod, model gerektirmez).

test_integration.py yalnız gerçek-ağırlık gerektiren (CI'da skip) testleri kapsar.
Bu dosya process_frame'i KONTROLLÜ bir stub dedektörle sürerek davranışı izole eder:
  - çift kapı: min_track_frames (heavy-stage) ile min_output_frames (çıktı)
  - takipsiz tespit (track_id=None) ağır aşamaya/çıktıya girmez
  - orphan-person bastırma: çıktıya girmeyen araca person bağlanmaz
  - sahne enjeksiyonu: set_scene araç döngüsünden ÖNCE çağrılır (active_speed_limit)
  - accumulator + QoD event'lerinin AYNI frame-saat ts'ini taşıması
"""

from __future__ import annotations

import numpy as np

from roadguard.detection.detector import Detection, Person
from roadguard.pipeline import Pipeline
from roadguard.schema import BBox


class _StubDetector:
    """Pipeline'a enjekte edilen kontrollü dedektör (mock muadili)."""

    def __init__(self):
        self.queue: list[list[Detection]] = []
        self.last_persons: list[Person] = []
        self.last_signs: list = []
        self.last_aux: list = []

    def detect(self, frame):
        return self.queue.pop(0) if self.queue else []

    def close(self):
        pass


def _det(tid, *, x1=100, y1=100, x2=200, y2=200, cls="car", conf=0.9):
    return Detection(bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, cls=cls), track_id=tid)


def _pipe(cfg, stub):
    cfg.data.setdefault("runtime", {})["ai_mode"] = "mock"
    p = Pipeline(cfg)
    p.detector = stub  # kontrollü tespit akışı
    return p


def _frame():
    return np.zeros((360, 640, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Çift kapı: çıktı min_output_frames'e kadar bastırılır
# --------------------------------------------------------------------------- #
def test_output_gate_suppresses_young_track(cfg):
    stub = _StubDetector()
    p = _pipe(cfg, stub)
    n_out = p.min_output_frames
    # n_out-1 frame boyunca aynı track → çıktı YOK (genç track bastırılır).
    for i in range(n_out - 1):
        stub.queue.append([_det(1)])
        anno, _ = p.process_frame(_frame(), i)
        assert anno.tracks == [], f"frame {i}: genç track çıktıya sızmamalı"
    # Eşiğe ulaşan frame → çıktı belirir.
    stub.queue.append([_det(1)])
    anno, _ = p.process_frame(_frame(), n_out - 1)
    assert len(anno.tracks) == 1 and anno.tracks[0]["track_id"] == 1


# --------------------------------------------------------------------------- #
# Takipsiz tespit (track_id=None) hiçbir aşamaya/çıktıya girmez
# --------------------------------------------------------------------------- #
def test_untracked_detection_never_emitted(cfg):
    stub = _StubDetector()
    p = _pipe(cfg, stub)
    for i in range(p.min_output_frames + 2):
        stub.queue.append([_det(None)])  # track_id None
        anno, ev = p.process_frame(_frame(), i)
        assert anno.tracks == []
        assert all(e.track_id != -1 for e in ev)


# --------------------------------------------------------------------------- #
# Orphan-person bastırma: çıktıya girmeyen araca person bağlanmaz
# --------------------------------------------------------------------------- #
def test_orphan_person_suppressed_for_young_vehicle(cfg):
    stub = _StubDetector()
    p = _pipe(cfg, stub)
    # İlk frame: araç henüz genç (çıktıya girmez) ama bir person var.
    stub.last_persons = [Person(bbox=BBox(x1=120, y1=110, x2=160, y2=190), track_id=50)]
    stub.queue.append([_det(1)])
    anno, _ = p.process_frame(_frame(), 0)
    # Araç çıktıya girmedi → ona bağlı person da çıktıya girmemeli (orphan yok).
    assert anno.tracks == []
    assert anno.persons == []


# --------------------------------------------------------------------------- #
# Sahne enjeksiyonu: SignTracker → accumulator.active_speed_limit
# --------------------------------------------------------------------------- #
def test_scene_active_limit_injected_into_accumulator(cfg):
    stub = _StubDetector()
    p = _pipe(cfg, stub)

    # SignTracker.update'i zorla: sahne bağlamı aktif limit 50 döndürsün.
    def _fake_update(signs, idx, now=None):
        from roadguard.schema import SceneContext

        return SceneContext(active_speed_limit_kmh=50), []

    p.sign_tracker.update = _fake_update
    stub.queue.append([_det(1)])
    p.process_frame(_frame(), 0)
    assert p.acc.active_speed_limit == 50


# --------------------------------------------------------------------------- #
# Frame-saat ts: bir frame'deki tüm event'ler aynı ts ekseninde
# --------------------------------------------------------------------------- #
def test_events_share_frame_clock_ts(cfg):
    stub = _StubDetector()
    p = _pipe(cfg, stub)
    p.fps = 30.0
    n = p.min_output_frames
    last_events = []
    for i in range(n):
        stub.queue.append([_det(1)])
        _, ev = p.process_frame(_frame(), i)
        last_events = ev
    expected_ts = (n - 1) / 30.0
    # Üretilen event'ler wall-clock değil frame-saat ts'i taşımalı.
    assert last_events, "olgunlaşan track event üretmeli"
    assert all(abs(e.ts - expected_ts) < 1e-9 for e in last_events)
