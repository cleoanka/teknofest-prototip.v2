"""Event + annotation yayıncısı.

Pipeline `RoadGuardEvent` (durum değişimleri) ve `AnnotationFrame` (kare başına bbox)
üretir. M2: in-memory halka tampon + callback kayıt defteri. M7: WS/SSE köprüsü
bu callback'lere abone olur (iki-kanal tasarımı: events ayrı, annotations ayrı).
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

from roadguard.schema import AnnotationFrame, RoadGuardEvent


class EventEmitter:
    def __init__(self, maxlen: int = 500):
        self.events: deque[RoadGuardEvent] = deque(maxlen=maxlen)
        self.annotations: deque[AnnotationFrame] = deque(maxlen=maxlen)
        self._event_cbs: list[Callable[[RoadGuardEvent], None]] = []
        self._annot_cbs: list[Callable[[AnnotationFrame], None]] = []
        # CA-001: pipeline iş parçacığı deque'lere YAZARKEN WS/SSE okuyucu iş
        # parçacığı recent_events()/latest_annotation() ile OKUYOR. deque.append
        # atomik olsa da snapshot (list(deque)) eş-zamanlı append ile "deque mutated
        # during iteration" / yarı-tutarlı görüntü üretebilir. Bu kilit yaz (append)
        # ile oku (snapshot) işlemlerini serileştirir. Callback'ler kilit DIŞINDA
        # çağrılır (uzun/bloklayan abone tüm yayını kilitlemesin; deadlock önlenir).
        self._lock = threading.Lock()

    # --- abonelik ---------------------------------------------------------- #
    def on_event(self, cb: Callable[[RoadGuardEvent], None]) -> None:
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
    def emit_event(self, event: RoadGuardEvent) -> None:
        with self._lock:  # CA-001: yazma okuyucu snapshot'larıyla serileşsin
            self.events.append(event)
        for cb in list(self._event_cbs):  # callback'ler kilit DIŞINDA (bloklamayı izole et)
            try:
                cb(event)
            except Exception:  # noqa: BLE001 - bir abone diğerlerini engellemesin
                pass

    def emit_annotation(self, anno: AnnotationFrame) -> None:
        with self._lock:
            self.annotations.append(anno)
        for cb in list(self._annot_cbs):
            try:
                cb(anno)
            except Exception:  # noqa: BLE001
                pass

    # --- okuma ------------------------------------------------------------- #
    def recent_events(self, n: int = 50) -> list[RoadGuardEvent]:
        with self._lock:  # CA-001: tutarlı snapshot (eş-zamanlı append ile yarış yok)
            return list(self.events)[-n:]

    def latest_annotation(self) -> AnnotationFrame | None:
        with self._lock:
            return self.annotations[-1] if self.annotations else None

    def snapshot_annotations(self) -> list[AnnotationFrame]:
        """annotations deque'sinin KİLİTLİ, tutarlı kopyası.

        CA-001: deque'i doğrudan (kilitsiz) iterleyen tüketiciler, pipeline thread'i
        eş-zamanlı `append` yaparken "deque mutated during iteration" (RuntimeError)
        alır. Tüm tarama gerektiren okuyucular bu metodu kullanmalı.
        """
        with self._lock:
            return list(self.annotations)
