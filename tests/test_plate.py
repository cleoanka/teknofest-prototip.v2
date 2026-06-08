"""Plaka: voting buffer, sweet-spot gating, regex, ret→QoD tetiği (model gerektirmez)."""

from __future__ import annotations

import numpy as np

from aura.plate.reader import PlateReader
from aura.plate.voting import VotingBuffer
from aura.schema import BBox

FRAME_SHAPE = (360, 640, 3)


class FakeOCR:
    def __init__(self, seq):
        self.seq = list(seq)
        self.i = 0

    def read(self, plate_roi, vehicle_crop=None):
        v = self.seq[self.i % len(self.seq)]
        self.i += 1
        return v


class FakeQoD:
    def __init__(self):
        self.calls = []

    def request_quality(self, track_id, reason):
        self.calls.append((track_id, reason))


def _center_bbox():  # sweet-spot içi (cx≈320, cy≈235)
    return BBox(x1=300, y1=210, x2=340, y2=260, conf=0.9, cls="car")


def _outside_bbox():  # sağ şerit, sweet-spot dışı (cx≈580 > 0.70*640)
    return BBox(x1=560, y1=210, x2=600, y2=260, conf=0.9, cls="car")


def _roi():
    return np.zeros((30, 40, 3), np.uint8)


# --- voting buffer ------------------------------------------------------- #
def test_voting_consensus():
    buf = VotingBuffer(5, 0.6)
    for t in ["34ABC123"] * 4 + ["06XY99"]:
        buf.add(t)
    val, frac = buf.consensus()
    assert val == "34ABC123" and abs(frac - 0.8) < 1e-9


def test_voting_reject_no_consensus():
    buf = VotingBuffer(5, 0.6)
    for t in ["A", "B", "C", "D", "E"]:
        buf.add(t)
    assert buf.consensus()[0] is None


# --- sweet spot + reader ------------------------------------------------- #
def test_sweet_spot_gates_ocr(cfg):
    r = PlateReader(cfg, ocr=FakeOCR([("34ABC123", 0.9)]))
    st = r.update(1, _roi(), _outside_bbox(), FRAME_SHAPE)
    assert st.status == "pending" and st.value is None  # OCR pasif


def test_plate_confirm_and_early_exit(cfg):
    r = PlateReader(cfg, ocr=FakeOCR([("34ABC123", 0.9)]))
    bbox = _center_bbox()
    st = None
    for _ in range(cfg.get("plate.voting_buffer_size")):
        st = r.update(1, _roi(), bbox, FRAME_SHAPE)
    assert st.status == "confirmed" and st.value == "34ABC123" and st.ocr_disabled is True


def test_plate_reject_triggers_qod(cfg):
    q = FakeQoD()
    r = PlateReader(cfg, qod=q, ocr=FakeOCR([("34ABC123", 0.9), ("06XY999", 0.9)]))
    bbox = _center_bbox()
    for _ in range(cfg.get("plate.voting_buffer_size") + 1):
        r.update(1, _roi(), bbox, FRAME_SHAPE)
    assert any(reason == "consensus_fail" for _, reason in q.calls)


def test_regex_rejects_invalid_plate(cfg):
    r = PlateReader(cfg, ocr=FakeOCR([("INVALID", 0.9)]))
    bbox = _center_bbox()
    st = None
    for _ in range(cfg.get("plate.voting_buffer_size")):
        st = r.update(1, _roi(), bbox, FRAME_SHAPE)
    assert st.status == "rejected"  # konsensüs var ama regex geçmez
