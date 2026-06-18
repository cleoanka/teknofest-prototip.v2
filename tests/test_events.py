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


def test_off_event_unsubscribes():
    em = EventEmitter()
    seen = []
    cb = seen.append
    em.on_event(cb)
    em.emit_event(make_event(1, "DETECTION_UPDATE"))
    em.off_event(cb)
    em.emit_event(make_event(2, "DETECTION_UPDATE"))
    assert len(seen) == 1  # abonelik kalktıktan sonra çağrılmadı


def test_off_event_unknown_callback_noop():
    em = EventEmitter()
    em.off_event(lambda e: None)  # kayıtlı değil → sessiz no-op


def test_off_annotation_unsubscribes():
    em = EventEmitter()
    seen = []
    cb = seen.append
    em.on_annotation(cb)
    em.emit_annotation(AnnotationFrame(frame_id=0))
    em.off_annotation(cb)
    em.emit_annotation(AnnotationFrame(frame_id=1))
    assert len(seen) == 1


def test_maxlen_ring_buffer_overflow():
    em = EventEmitter(maxlen=3)
    for i in range(5):
        em.emit_event(make_event(i, "DETECTION_UPDATE"))
    ev = list(em.events)
    assert len(ev) == 3  # halka tampon: yalnız son 3
    assert [e.track_id for e in ev] == [2, 3, 4]


def test_recent_events_slicing():
    em = EventEmitter(maxlen=100)
    for i in range(10):
        em.emit_event(make_event(i, "DETECTION_UPDATE"))
    last3 = em.recent_events(3)
    assert [e.track_id for e in last3] == [7, 8, 9]
    # n buffer'dan büyükse tümünü döner.
    assert len(em.recent_events(50)) == 10


def test_callback_exception_isolated():
    """Bir abone patlasa bile diğer aboneler çalışmaya devam etmeli (izolasyon)."""
    em = EventEmitter()
    calls = []

    def bad(_e):
        raise RuntimeError("patlama")

    em.on_event(bad)
    em.on_event(calls.append)
    em.emit_event(make_event(1, "DETECTION_UPDATE"))  # exception yutulmalı
    assert len(calls) == 1  # ikinci abone yine çağrıldı


def test_annotation_callback_exception_isolated():
    em = EventEmitter()
    calls = []
    em.on_annotation(lambda _a: (_ for _ in ()).throw(RuntimeError("x")))
    em.on_annotation(calls.append)
    em.emit_annotation(AnnotationFrame(frame_id=0))
    assert len(calls) == 1


def test_latest_annotation_none_when_empty():
    em = EventEmitter()
    assert em.latest_annotation() is None
