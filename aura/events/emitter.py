"""Event + annotation yayıncısı.

Pipeline `AuraEvent` (durum değişimleri) ve `AnnotationFrame` (kare başına bbox)
üretir. M2: in-memory halka tampon + callback kayıt defteri. M7: WS/SSE köprüsü
bu callback'lere abone olur (iki-kanal tasarımı: events ayrı, annotations ayrı).
"""
from __future__ import annotations

from collections import deque
from typing import Callable

from aura.schema import AnnotationFrame, AuraEvent


class EventEmitter:
    def __init__(self, maxlen: int = 500):
        self.events: deque[AuraEvent] = deque(maxlen=maxlen)
        self.annotations: deque[AnnotationFrame] = deque(maxlen=maxlen)
        self._event_cbs: list[Callable[[AuraEvent], None]] = []
        self._annot_cbs: list[Callable[[AnnotationFrame], None]] = []

    # --- abonelik ---------------------------------------------------------- #
    def on_event(self, cb: Callable[[AuraEvent], None]) -> None:
        self._event_cbs.append(cb)

    def on_annotation(self, cb: Callable[[AnnotationFrame], None]) -> None:
        self._annot_cbs.append(cb)

    def off_event(self, cb) -> None:
        if cb in self._event_cbs:
            self._event_cbs.remove(cb)

    def off_annotation(self, cb) -> None:
        if cb in self._annot_cbs:
            self._annot_cbs.remove(cb)

    # --- yayın ------------------------------------------------------------- #
    def emit_event(self, event: AuraEvent) -> None:
        self.events.append(event)
        for cb in list(self._event_cbs):
            try:
                cb(event)
            except Exception:  # noqa: BLE001 - bir abone diğerlerini engellemesin
                pass

    def emit_annotation(self, anno: AnnotationFrame) -> None:
        self.annotations.append(anno)
        for cb in list(self._annot_cbs):
            try:
                cb(anno)
            except Exception:  # noqa: BLE001
                pass

    # --- okuma ------------------------------------------------------------- #
    def recent_events(self, n: int = 50) -> list[AuraEvent]:
        return list(self.events)[-n:]

    def latest_annotation(self) -> AnnotationFrame | None:
        return self.annotations[-1] if self.annotations else None
