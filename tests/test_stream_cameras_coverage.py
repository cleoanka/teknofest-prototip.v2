"""S2-stream-cameras kapsamlı birim/endpoint testleri (TestClient, mock-mod).

S1-api-core'un (test_api_core_coverage.py) kapsamadığı boşluklara odaklanır:
  - cameras.py: _macos_camera_names parse mantığı, _name_for Darwin index>len fallback,
    enumerate_cameras cv2 hata-yutma (except pass) + isOpened()/release dalları
  - StreamManager._push / _safe_put: dolu Queue(maxsize) → QueueFull loop callback'inde
    YAKALANIR (Bug: yorumun aksine eski kod yakalamıyordu) + çoklu abone fan-out
  - _draw: çizilecek bir şey yokken frame.copy() yapmadan ham kareyi döndürür (perf)
  - _draw: tracks/signs/limit dolu iken bbox/etiket çizer (davranış korunur)
  - stream_start geçersiz kaynak → worker pipeline/cap açılmaz → _running False
  - status() pipeline=None vs pipeline dolu iken active_tracks/qod_active_sessions
  - WS slow-consumer: dolu kuyruk push'u sessizce drop eder, exception sızmaz
"""

from __future__ import annotations

import os

os.environ.setdefault("AURA_AUTOSTART", "0")
os.environ.setdefault("AURA_CAMERA_PROBE", "0")
os.environ.setdefault("AI_MODE", "mock")

import asyncio  # noqa: E402

import numpy as np  # noqa: E402

from aura.config import load_config  # noqa: E402
from aura.schema import AnnotationFrame  # noqa: E402
from services.inference_api.routers import cameras as cam_mod  # noqa: E402
from services.inference_api.state import StreamManager  # noqa: E402


# --- cameras: _macos_camera_names parse ----------------------------------- #
def test_macos_camera_names_parses_only_name_headers(monkeypatch):
    """system_profiler çıktısından yalnız kamera adı başlıkları çekilir."""
    sample = (
        "Camera:\n"
        "\n"
        "    FaceTime HD Camera:\n"
        "      Model ID: UVC Camera\n"
        "      Unique ID: 0x12345\n"
        "      Manufacturer: Apple Inc.\n"
        "\n"
        "    iPhone Camera:\n"
        "      Model ID: iPhone\n"
    )

    class _Res:
        stdout = sample

    monkeypatch.setattr(cam_mod.subprocess, "run", lambda *a, **k: _Res())
    names = cam_mod._macos_camera_names()
    # "Camera:" başlığı (Camera ile başlar) ve Model/Unique/Manufacturer satırları elenir.
    assert "FaceTime HD Camera" in names
    assert "iPhone Camera" in names
    assert all("Model" not in n and "Unique" not in n for n in names)


def test_macos_camera_names_swallows_subprocess_error(monkeypatch):
    def _boom(*a, **k):
        raise OSError("system_profiler yok")

    monkeypatch.setattr(cam_mod.subprocess, "run", _boom)
    assert cam_mod._macos_camera_names() == []  # except → [] (boş, çökme yok)


# --- cameras: _name_for fallback ------------------------------------------ #
def test_name_for_darwin_in_range(monkeypatch):
    monkeypatch.setattr(cam_mod.platform, "system", lambda: "Darwin")
    assert cam_mod._name_for(1, ["A", "B", "C"]) == "B"


def test_name_for_darwin_index_out_of_range_fallback(monkeypatch):
    monkeypatch.setattr(cam_mod.platform, "system", lambda: "Darwin")
    # index >= len(mac_names) → generic isim (fallback dalı).
    assert cam_mod._name_for(5, ["A"]) == "Camera 5"


def test_name_for_non_darwin_uses_generic(monkeypatch):
    monkeypatch.setattr(cam_mod.platform, "system", lambda: "Linux")
    assert cam_mod._name_for(0, ["ShouldIgnore"]) == "Camera 0"


# --- cameras: enumerate_cameras probe dalları ----------------------------- #
def test_enumerate_cameras_probe_disabled(monkeypatch):
    monkeypatch.setenv("AURA_CAMERA_PROBE", "0")
    assert cam_mod.enumerate_cameras() == []


def test_enumerate_cameras_opened_and_closed(monkeypatch):
    """Probe açık: açılan indeksler CameraInfo'ya dönüşür, kapalılar atlanır."""
    monkeypatch.setenv("AURA_CAMERA_PROBE", "1")
    monkeypatch.setattr(cam_mod.platform, "system", lambda: "Linux")

    class _FakeCap:
        def __init__(self, idx):
            self.idx = idx
            self.released = False

        def isOpened(self):
            return self.idx == 0  # yalnız 0 açık

        def get(self, prop):
            if prop == cam_mod.cv2.CAP_PROP_FRAME_WIDTH:
                return 1280
            if prop == cam_mod.cv2.CAP_PROP_FRAME_HEIGHT:
                return 720
            return 0

        def release(self):
            self.released = True

    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", lambda i: _FakeCap(i))
    cams = cam_mod.enumerate_cameras(max_index=3)
    assert len(cams) == 1
    assert cams[0].index == 0
    assert cams[0].width == 1280 and cams[0].height == 720
    assert cams[0].name == "Camera 0"


def test_enumerate_cameras_zero_dims_defaults(monkeypatch):
    """get() 0 döndürürse 640x480 default'a düşülür (`or` dalı)."""
    monkeypatch.setenv("AURA_CAMERA_PROBE", "1")
    monkeypatch.setattr(cam_mod.platform, "system", lambda: "Linux")

    class _Cap:
        def __init__(self, i):
            self.i = i

        def isOpened(self):
            return self.i == 0

        def get(self, prop):
            return 0  # 0 → default tetikler

        def release(self):
            pass

    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", lambda i: _Cap(i))
    cams = cam_mod.enumerate_cameras(max_index=1)
    assert cams[0].width == 640 and cams[0].height == 480


def test_enumerate_cameras_swallows_cv2_exception(monkeypatch):
    """VideoCapture fırlatırsa except pass → o indeks atlanır, çökme yok."""
    monkeypatch.setenv("AURA_CAMERA_PROBE", "1")
    monkeypatch.setattr(cam_mod.platform, "system", lambda: "Linux")

    def _boom(i):
        raise RuntimeError("kamera erişim hatası")

    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", _boom)
    assert cam_mod.enumerate_cameras(max_index=2) == []


# --- StreamManager._push / _safe_put : QueueFull yutma -------------------- #
def test_safe_put_swallows_queue_full():
    sm = StreamManager(load_config())
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    q.put_nowait({"a": 0})  # kuyruğu doldur
    # _safe_put loop thread'inde koşar; dolu kuyruk → QueueFull yutulmalı (çökme yok).
    sm._safe_put(q, {"a": 1})
    assert q.qsize() == 1  # ikinci öğe düşürüldü


def test_push_full_queue_does_not_raise_in_loop():
    """Bug doğrulama: dolu kuyruk push'u event loop'ta yakalanmamış istisna üretmez.

    Eski kod put_nowait'i call_soon_threadsafe'e doğrudan veriyordu → QueueFull
    callback'te patlıyordu. Yeni kod _safe_put sarmalayıcısı ile yutar.
    """
    loop = asyncio.new_event_loop()
    errors = []
    loop.set_exception_handler(lambda lp, ctx: errors.append(ctx))
    sm = StreamManager(load_config())
    sm.attach_loop(loop)
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    q.put_nowait({"x": 0})  # dolu
    queues = {q}

    async def _drive():
        sm._push(queues, {"x": 1})  # dolu kuyruğa push
        await asyncio.sleep(0)  # scheduled callback'in koşmasına izin ver

    try:
        loop.run_until_complete(_drive())
    finally:
        loop.close()
    assert errors == []  # loop'a hiçbir yakalanmamış istisna sızmadı
    assert q.qsize() == 1


def test_push_fans_out_to_multiple_subscribers():
    loop = asyncio.new_event_loop()
    sm = StreamManager(load_config())
    sm.attach_loop(loop)
    q1 = sm.subscribe_annotations()
    q2 = sm.subscribe_annotations()

    async def _drive():
        sm._emit_annotation(AnnotationFrame(frame_id=11, tracks=[]))
        await asyncio.sleep(0)
        return await q1.get(), await q2.get()

    try:
        a, b = loop.run_until_complete(_drive())
    finally:
        loop.close()
    assert a["frame_id"] == 11 and b["frame_id"] == 11


# --- StreamManager._draw : perf no-op + çizim davranışı ------------------- #
def _blank(h=20, w=20):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_returns_same_frame_when_nothing_to_draw():
    frame = _blank()
    anno = AnnotationFrame(frame_id=0, tracks=[])
    out = StreamManager._draw(frame, anno)
    # Çizilecek bir şey yok → frame.copy() yapılmaz; aynı nesne döner (tahsis yok).
    assert out is frame


def test_draw_copies_and_draws_when_tracks_present():
    frame = _blank(40, 40)
    anno = AnnotationFrame(
        frame_id=1,
        tracks=[{"track_id": 1, "bbox": [1, 1, 10, 10], "plate": "34TC8532"}],
    )
    out = StreamManager._draw(frame, anno)
    assert out is not frame  # kopya alınmış
    # En az bir piksel çizilmiş (ham frame sıfırdı).
    assert int(out.sum()) > 0


def test_draw_with_risk_flag_branch():
    frame = _blank(40, 40)
    anno = AnnotationFrame(
        frame_id=2,
        tracks=[{"track_id": 2, "bbox": [2, 2, 15, 15], "risk_flags": ["X"]}],
    )
    out = StreamManager._draw(frame, anno)
    assert out is not frame and int(out.sum()) > 0


# --- StreamManager.status() : pipeline None vs dolu ----------------------- #
def test_status_pipeline_none_zero_counts():
    sm = StreamManager(load_config())
    st = sm.status()
    assert st["active_tracks"] == 0 and st["qod_active_sessions"] == 0
    assert st["running"] is False and st["source"] is None


def test_status_with_pipeline_reports_counts():
    sm = StreamManager(load_config())

    class _Acc:
        def active_tracks(self):
            return [1, 2, 3]

    class _Qod:
        def active_sessions(self):
            return {"a": 1, "b": 2}

    class _Pipe:
        acc = _Acc()
        qod = _Qod()

    sm.pipeline = _Pipe()
    sm.source = "0"
    sm.device = "cpu"
    st = sm.status()
    assert st["active_tracks"] == 3
    assert st["qod_active_sessions"] == 2
    assert st["source"] == "0" and st["device"] == "cpu"


# --- stream_start geçersiz kaynak → _running False ------------------------ #
def test_start_invalid_source_sets_not_running():
    """Açılamayan kaynak: worker cap.isOpened() False → _running False olur.

    Pipeline mock-mod'da kurulur; cv2.VideoCapture açılamaz → temiz durur.
    Thread join ile worker'ın tamamlanması beklenir.
    """
    sm = StreamManager(load_config())
    sm.start(source="/nonexistent/__yok__.mp4", device="cpu", bbox_overlay=False)
    # Worker thread arka planda; kısa süre içinde _running False'a düşmeli.
    for _ in range(200):
        if not sm.running:
            break
        time_sleep()
    sm.stop()
    assert sm.running is False


def time_sleep():
    import time

    time.sleep(0.02)
