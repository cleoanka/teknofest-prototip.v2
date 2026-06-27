"""§6.0 sözleşmeleri + accumulator/risk mantığı (model gerektirmez)."""

from __future__ import annotations

from roadguard.accumulator import Accumulator
from roadguard.schema import (
    AnnotationFrame,
    BBox,
    DriverState,
    PlateState,
    SpeedState,
    TrackRecord,
    make_event,
)


def test_core_models_defaults():
    assert PlateState().status == "pending"
    assert DriverState(no_seatbelt=True).active_flags() == ["no_seatbelt"]
    assert SpeedState().mode == "disabled"
    b = BBox(x1=0, y1=0, x2=10, y2=20, conf=0.5, cls="car")
    assert b.width == 10 and b.height == 20 and b.center == (5, 10)
    t = TrackRecord(track_id=1, bbox=b)
    assert t.plate.status == "pending" and t.driver.phone is False


def test_make_event_and_annotation():
    e = make_event(7, "SPEED", {"value_kmh": 50}, ts=123.0)
    assert (
        e.type == "SPEED"
        and e.track_id == 7
        and e.ts == 123.0
        and e.source == "roadguard-inference"
    )
    a = AnnotationFrame(frame_id=3, tracks=[{"track_id": 1}])
    assert a.frame_id == 3 and a.tracks[0]["track_id"] == 1


def test_accumulator_new_track_emits_detection(cfg):
    acc = Accumulator(cfg)
    b = BBox(x1=0, y1=0, x2=10, y2=10, conf=0.9, cls="car")
    rec, ev = acc.update_track(1, frame_idx=0, bbox=b, vehicle_class="car")
    assert rec.track_id == 1 and rec.vehicle_class == "car"
    assert any(e.type == "DETECTION_UPDATE" for e in ev)


def test_accumulator_driver_change_and_risk(cfg):
    acc = Accumulator(cfg)
    b = BBox(x1=0, y1=0, x2=10, y2=10, conf=0.9, cls="car")
    acc.update_track(1, frame_idx=0, bbox=b, vehicle_class="car")
    rec, ev = acc.update_track(1, frame_idx=1, bbox=b, driver=DriverState(no_seatbelt=True))
    types = {e.type for e in ev}
    assert "DRIVER_STATE" in types
    assert "RISK_ALERT" in types  # config'teki 'unbelted' kuralı
    assert "unbelted" in rec.risk_flags


def test_accumulator_no_duplicate_risk_alert(cfg):
    acc = Accumulator(cfg)
    b = BBox(x1=0, y1=0, x2=10, y2=10, conf=0.9, cls="car")
    acc.update_track(1, frame_idx=0, bbox=b, driver=DriverState(no_seatbelt=True))
    # ikinci kez aynı durum → yeni RISK_ALERT olmamalı (sadece tetiklenince)
    _, ev = acc.update_track(1, frame_idx=1, bbox=b, driver=DriverState(no_seatbelt=True))
    assert not any(e.type == "RISK_ALERT" for e in ev)
