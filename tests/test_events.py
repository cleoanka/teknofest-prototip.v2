"""AuraEvent/AnnotationFrame şema doğrulama + EventEmitter + StreamManager push."""

from __future__ import annotations

import asyncio

from aura.events import EventEmitter
from aura.schema import AnnotationFrame, AuraEvent, make_event


def test_event_roundtrip():
    e = make_event(5, "PLATE_CONFIRMED", {"value": "34ABC123"}, ts=1.0)
    d = e.model_dump()
    assert d["type"] == "PLATE_CONFIRMED" and d["track_id"] == 5 and d["source"] == "aura-inference"
    assert AuraEvent.model_validate(d).payload["value"] == "34ABC123"


def test_annotation_roundtrip():
    a = AnnotationFrame(frame_id=1, ts=2.0, tracks=[{"track_id": 1, "bbox": [0, 0, 1, 1]}])
    assert AnnotationFrame.model_validate(a.model_dump()).tracks[0]["bbox"] == [0, 0, 1, 1]


def test_emitter_callbacks_and_buffer():
    em = EventEmitter(maxlen=10)
    seen = []
    em.on_event(seen.append)
    for i in range(3):
        em.emit_event(make_event(i, "DETECTION_UPDATE"))
    assert len(seen) == 3 and len(em.recent_events()) == 3
    em.emit_annotation(AnnotationFrame(frame_id=0))
    assert em.latest_annotation().frame_id == 0


def test_invalid_event_type_rejected():
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        AuraEvent(track_id=1, type="NOT_A_TYPE")


def test_stream_manager_threadsafe_push(cfg):
    from services.inference_api.state import StreamManager

    sm = StreamManager(cfg)
    loop = asyncio.new_event_loop()
    try:
        sm.attach_loop(loop)
        q = sm.subscribe_events()
        sm._push(sm._event_queues, {"hello": 1})
        loop.call_soon(loop.stop)
        loop.run_forever()
        assert not q.empty() and q.get_nowait() == {"hello": 1}
    finally:
        loop.close()
