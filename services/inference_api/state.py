"""StreamManager — pipeline'ı arka planda koşturan ve MJPEG + WS besleyen runtime.

İki-kanal mimari: ham JPEG (MJPEG) ve `AnnotationFrame`/`AuraEvent` WS akışları
ayrı yayılır. Pipeline arka plan thread'inde çalışır; WS push'ları event loop'a
`call_soon_threadsafe` ile güvenli aktarılır.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import cv2

from aura.config import load_config
from aura.pipeline import Pipeline

log = logging.getLogger("aura.api.stream")


class StreamManager:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.pipeline: Pipeline | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.source = None
        self.device = None
        self.bbox_overlay = True
        self.loop: asyncio.AbstractEventLoop | None = None
        self._raw_jpeg: bytes | None = None
        self._annot_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self.frame_count = 0
        self.fps_actual = 0.0
        self._event_queues: set[asyncio.Queue] = set()
        self._annot_queues: set[asyncio.Queue] = set()
        self.started_at = 0.0

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    # --- WS abonelikleri --------------------------------------------------- #
    def subscribe_events(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._event_queues.add(q)
        return q

    def unsubscribe_events(self, q) -> None:
        self._event_queues.discard(q)

    def subscribe_annotations(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._annot_queues.add(q)
        return q

    def unsubscribe_annotations(self, q) -> None:
        self._annot_queues.discard(q)

    def _push(self, queues: set, data: dict) -> None:
        if not self.loop:
            return
        for q in list(queues):
            try:
                self.loop.call_soon_threadsafe(q.put_nowait, data)
            except Exception:  # noqa: BLE001 - dolu/kapalı kuyruğu yok say
                pass

    # --- yaşam döngüsü ----------------------------------------------------- #
    @property
    def running(self) -> bool:
        return self._running

    def start(self, source=None, device=None, bbox_overlay=True) -> None:
        self.stop()
        self.source = source if source not in (None, "") else self.cfg.get("runtime.source")
        self.device = device or self.cfg.get("runtime.device")
        self.bbox_overlay = bbox_overlay
        if device:
            self.cfg.data.setdefault("runtime", {})["device"] = device
        self.pipeline = Pipeline(self.cfg)
        self.pipeline.emitter.on_event(lambda e: self._push(self._event_queues, e.model_dump()))
        self.pipeline.emitter.on_annotation(
            lambda a: self._push(self._annot_queues, a.model_dump())
        )
        self._running = True
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info("Stream başladı: source=%s device=%s", self.source, self.device)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.pipeline = None
        self.frame_count = 0

    # --- arka plan worker -------------------------------------------------- #
    def _worker(self) -> None:
        src = self.source
        src = int(src) if isinstance(src, str) and src.isdigit() else src
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error("Kaynak açılamadı: %s", self.source)
            self._running = False
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.pipeline.fps = fps
        self.pipeline.speed.fps = fps
        is_file = isinstance(src, str)
        delay = 1.0 / fps if fps > 0 else 0.033
        i, t0 = 0, time.time()
        while self._running:
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    i = 0
                    continue
                break
            anno, _events = self.pipeline.process_frame(frame, i)
            ok1, raw = cv2.imencode(".jpg", frame)
            ok2, annot = cv2.imencode(".jpg", self._draw(frame, anno))
            with self._lock:
                if ok1:
                    self._raw_jpeg = raw.tobytes()
                if ok2:
                    self._annot_jpeg = annot.tobytes()
                self.frame_count = i
            i += 1
            if i % 30 == 0:
                dt = time.time() - t0
                self.fps_actual = round(30.0 / dt, 1) if dt > 0 else 0.0
                t0 = time.time()
            time.sleep(delay)
        cap.release()
        self._running = False
        log.info("Stream worker durdu (%d kare)", i)

    @staticmethod
    def _draw(frame, anno):
        img = frame.copy()
        for t in anno.tracks:
            x1, y1, x2, y2 = (int(v) for v in t["bbox"])
            color = (68, 68, 255) if t.get("risk_flags") else (136, 255, 0)  # BGR
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"ID{t['track_id']}"
            if t.get("plate"):
                label += f" {t['plate']}"
            cv2.putText(
                img,
                label,
                (x1, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return img

    # --- okuma ------------------------------------------------------------- #
    def latest_jpeg(self, bbox: bool) -> bytes | None:
        with self._lock:
            return self._annot_jpeg if bbox else self._raw_jpeg

    def status(self) -> dict:
        return {
            "running": self._running,
            "source": str(self.source) if self.source is not None else None,
            "device": self.device,
            "bbox_overlay": self.bbox_overlay,
            "frame_count": self.frame_count,
            "fps": self.fps_actual,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            "active_tracks": len(self.pipeline.acc.active_tracks()) if self.pipeline else 0,
            "qod_active_sessions": (
                len(self.pipeline.qod.active_sessions()) if self.pipeline else 0
            ),
        }
