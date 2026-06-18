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

from aura.config import load_config, resolve_source
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

    def _emit_event(self, e) -> None:
        # Yalnız abone varsa serialize et + fan-out yap (boşken model_dump israfı yok).
        if self._event_queues and self.loop:
            self._push(self._event_queues, e.model_dump())

    def _emit_annotation(self, a) -> None:
        if self._annot_queues and self.loop:
            self._push(self._annot_queues, a.model_dump())

    @staticmethod
    def _safe_put(q: asyncio.Queue, data: dict) -> None:
        # Event loop içinde çalışır. Dolu kuyruk (yavaş WS client'ı) put_nowait'te
        # QueueFull fırlatır; bu callback loop'ta koştuğu için _push'taki try/except
        # YAKALAMAZ → loop "exception in callback" log'u spam'lerdi. Burada yutarız:
        # en yeni kareyi düşürmek, akışı/log'u bozmaktan iyidir.
        try:
            q.put_nowait(data)
        except Exception:  # noqa: BLE001 - dolu/kapalı kuyruğu yok say
            pass

    def _push(self, queues: set, data: dict) -> None:
        loop = self.loop
        if not loop or not queues:
            return
        # `call_soon_threadsafe` yalnız loop kapalı/zamanlama hatalarını fırlatır;
        # gerçek put QueueFull'ı _safe_put içinde (loop thread'inde) yakalanır.
        for q in tuple(queues):
            try:
                loop.call_soon_threadsafe(self._safe_put, q, data)
            except Exception:  # noqa: BLE001 - loop kapalı/durmuş → yok say
                pass

    # --- yaşam döngüsü ----------------------------------------------------- #
    @property
    def running(self) -> bool:
        return self._running

    def start(self, source=None, device=None, bbox_overlay=True) -> None:
        self.stop()
        self.source = source if source not in (None, "") else resolve_source(self.cfg)
        self.device = device or self.cfg.get("runtime.device")
        self.bbox_overlay = bbox_overlay
        if device:
            self.cfg.data.setdefault("runtime", {})["device"] = device
        if source not in (None, ""):
            # Kaynak config'e de yazılır: Pipeline kurulurken resolve_ai_mode gerçek
            # kaynağa bakar; aksi halde auto modu config'teki örnek videoya göre karar
            # verip gerçek kaynağı mock'ta işleyebiliyordu (CLI'daki D3'ün servis eşi).
            self.cfg.data.setdefault("runtime", {})["source"] = source
        self._running = True
        self.started_at = time.time()
        # Pipeline (ağır model yüklemeleri: YOLO/EasyOCR ısınması) worker thread'inde
        # kurulur → sunucu başlangıcını ve /stream/start isteğini BLOKLAMAZ.
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info("Stream başlatılıyor: source=%s device=%s", self.source, self.device)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.pipeline = None
        self.frame_count = 0

    # --- arka plan worker -------------------------------------------------- #
    def _worker(self) -> None:
        # Ağır model yüklemeleri burada (arka planda) yapılır → başlangıcı bloklamaz.
        try:
            self.pipeline = Pipeline(self.cfg)
            # Hot path mikro-opt: abone YOKKEN (yaygın "dashboard yalnız MJPEG" durumu)
            # kare başına pydantic model_dump()/fan-out yapma. Abone varken davranış
            # birebir aynı. model_dump yalnızca gerçekten gönderilecekse hesaplanır.
            self.pipeline.emitter.on_event(self._emit_event)
            self.pipeline.emitter.on_annotation(self._emit_annotation)
        except Exception:  # noqa: BLE001 - model/ağırlık hatası → akışı temiz durdur
            log.exception("Pipeline kurulamadı — stream durduruluyor")
            self._running = False
            return

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
        i, t0, errors = 0, time.time(), 0
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    if is_file:  # dosya bitti → başa sar (sürekli demo döngüsü)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        i = 0
                        continue
                    break
                # Ham kareyi HEMEN yayınla: video akışı, (yavaş olabilen) tespitten
                # bağımsız akıcı kalsın. Annotated kanal tespit bitince güncellenir.
                ok_raw, raw = cv2.imencode(".jpg", frame)
                if ok_raw:
                    with self._lock:
                        self._raw_jpeg = raw.tobytes()
                # Tek bir karenin hatası tüm akışı düşürmesin (robustluk).
                try:
                    anno, _events = self.pipeline.process_frame(frame, i)
                    ok_an, annot = cv2.imencode(".jpg", self._draw(frame, anno))
                    if ok_an:
                        with self._lock:
                            self._annot_jpeg = annot.tobytes()
                    errors = 0
                except Exception:  # noqa: BLE001
                    errors += 1
                    log.exception("Kare işleme hatası (#%d, kare %d)", errors, i)
                    if errors >= 30:
                        log.error("Ardışık 30 kare işleme hatası — stream durduruluyor")
                        break
                with self._lock:
                    self.frame_count = i
                i += 1
                if i % 30 == 0:
                    dt = time.time() - t0
                    self.fps_actual = round(30.0 / dt, 1) if dt > 0 else 0.0
                    t0 = time.time()
                time.sleep(delay)
        finally:
            cap.release()
            self._running = False
            log.info("Stream worker durdu (%d kare)", i)

    @staticmethod
    def _draw(frame, anno):
        signs = getattr(anno, "signs", None) or []
        lim = (getattr(anno, "scene", None) or {}).get("active_speed_limit_kmh")
        # Çizilecek hiçbir şey yoksa frame.copy() (kare-başı tahsis) gereksiz; ham
        # kareyi olduğu gibi döndür. imencode kaynağı değiştirmez → davranış aynı.
        if not anno.tracks and not signs and lim is None:
            return frame
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
        # Trafik tabelaları (sahne-seviyesi) — BGR ~ açık mavi (#33ccff)
        sign_color = (255, 204, 51)
        for s in signs:
            sx1, sy1, sx2, sy2 = (int(v) for v in s["bbox"])
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), sign_color, 2)
            lbl = (
                f"{s['speed_limit_kmh']} km/h"
                if s.get("speed_limit_kmh") is not None
                else s.get("cls", "tabela")
            )
            cv2.putText(
                img,
                lbl,
                (sx1, max(12, sy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                sign_color,
                1,
                cv2.LINE_AA,
            )
        # Aktif hız limiti banner'ı (sol-üst)
        if lim is not None:
            cv2.putText(
                img,
                f"LIMIT {lim}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                sign_color,
                2,
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
