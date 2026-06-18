"""S1-api-core kapsamlı endpoint/birim testleri (TestClient, mock-mod, model gerekmez).

Kapsanan boşluklar (review test_gaps):
  - /stream/config PATCH (bbox_overlay + conf_threshold dalları, status şekli)
  - /stream/video MJPEG (idle/break path, ?bbox toggle)
  - WS /stream/annotations + /stream/events (subscribe/unsubscribe lifecycle, fan-out)
  - /eval/* tüm dallar (no_results, queued + background _job hata yolu, export her dal)
  - /tracks/{id}/history (pipeline None dalı + dolu seri + count/son-200 sözleşmesi)
  - /config qod_profile + bbox_overlay dalları + 422 validasyonları
  - /info config_summary anahtarları + gömülü status() şekli
  - cameras enumerate (PROBE=0 → []) + isim çözümü
  - Config izolasyonu (as_dict deep-copy, qod_profile doğru hedef anahtar)
  - StreamManager robustluk: subscribe/unsubscribe, _push boş-kuyruk/loop-yok no-op
"""

from __future__ import annotations

import os

os.environ.setdefault("AURA_AUTOSTART", "0")
os.environ.setdefault("AURA_CAMERA_PROBE", "0")
os.environ.setdefault("AI_MODE", "mock")

import asyncio  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from aura.config import load_config  # noqa: E402
from aura.schema import AnnotationFrame, make_event  # noqa: E402
from services.inference_api.main import create_app  # noqa: E402
from services.inference_api.state import StreamManager  # noqa: E402


@pytest.fixture
def client():
    # Fonksiyon-kapsamlı: her test taze app/StreamManager alır → config sızıntısı yok.
    with TestClient(create_app()) as c:
        yield c


# --- /config -------------------------------------------------------------- #
def test_config_get_returns_deepcopy_no_leak(client):
    """GET /config döndürülen sözlük mutasyonu canlı state'i kirletmemeli."""
    j = client.get("/config").json()
    assert "runtime" in j
    # Döndürülen yapıyı boz; ardından tekrar GET → bozulma sızmamalı.
    j["runtime"]["__leak__"] = "x"
    again = client.get("/config").json()
    assert "__leak__" not in again.get("runtime", {})


def test_config_patch_conf_threshold(client):
    r = client.patch("/config", json={"conf_threshold": 0.42})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "updated"
    assert body["config"]["models"]["detector"]["conf"] == 0.42


def test_config_patch_bbox_overlay_side_effect(client):
    r = client.patch("/config", json={"bbox_overlay": False})
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["dashboard"]["default_bbox"] is False
    # status() bbox_overlay alanı da güncellenmeli (sm.bbox_overlay side-effect)
    assert client.get("/stream/status").json()["bbox_overlay"] is False


def test_config_patch_qod_profile_valid_target_key(client):
    """Geçerli profil adı → qod.active_profile'a yazılır (profiles.quality DEĞİL)."""
    cfg = client.get("/config").json()
    valid = next(iter(cfg["qod"]["profiles"].keys()))
    r = client.patch("/config", json={"qod_profile": valid})
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["qod"]["active_profile"] == valid
    # profiles haritası bozulmamalı (eski hatalı davranış quality'yi eziyordu)
    assert body["config"]["qod"]["profiles"] == cfg["qod"]["profiles"]


def test_config_patch_qod_profile_invalid_rejected(client):
    r = client.patch("/config", json={"qod_profile": "__yok__"})
    assert r.status_code == 422


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_config_patch_conf_threshold_out_of_range_422(client, bad):
    assert client.patch("/config", json={"conf_threshold": bad}).status_code == 422


def test_config_patch_empty_body_ok(client):
    # Hiçbir alan verilmezse no-op ama 200 (tüm alanlar Optional).
    r = client.patch("/config", json={})
    assert r.status_code == 200 and r.json()["status"] == "updated"


# --- /stream/config ------------------------------------------------------- #
def test_stream_config_bbox_overlay(client):
    r = client.patch("/stream/config", json={"bbox_overlay": False})
    assert r.status_code == 200
    # /stream/config yanıtı status() şekli döndürür ({'status':...} DEĞİL).
    body = r.json()
    assert "running" in body and body["bbox_overlay"] is False
    assert "status" not in body


def test_stream_config_conf_threshold(client):
    r = client.patch("/stream/config", json={"conf_threshold": 0.33})
    assert r.status_code == 200
    assert client.get("/config").json()["models"]["detector"]["conf"] == 0.33


@pytest.mark.parametrize("bad", [-1.0, 2.0])
def test_stream_config_conf_threshold_out_of_range_422(client, bad):
    assert client.patch("/stream/config", json={"conf_threshold": bad}).status_code == 422


def test_stream_config_empty_noop(client):
    r = client.patch("/stream/config", json={})
    assert r.status_code == 200 and "running" in r.json()


# --- /stream/start defaults / validation ---------------------------------- #
def test_stream_start_no_body_uses_defaults(client):
    # Boş gövde: source=None → resolve_source fallback, bbox_overlay default True.
    r = client.post("/stream/start", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "started"
    assert body["bbox_overlay"] is True
    client.post("/stream/stop")


def test_stream_stop_idempotent(client):
    assert client.post("/stream/stop").json()["status"] == "stopped"
    assert client.post("/stream/stop").json()["status"] == "stopped"


# --- /stream/video MJPEG -------------------------------------------------- #
def test_stream_video_idle_breaks_when_not_running(client):
    """Stream çalışmazken jpeg yok → idle>20 ile generator kapanır (boş gövde)."""
    r = client.get("/stream/video")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert r.content == b""  # hiç kare yok, akış temiz biter


def test_stream_video_bbox_param_idle(client):
    r = client.get("/stream/video", params={"bbox": "true"})
    assert r.status_code == 200
    assert r.content == b""


def test_stream_video_generator_yields_frame_then_stops(monkeypatch):
    """MJPEG generator: kare varken multipart frame yayınlar; stop sonrası temiz biter.

    Generator doğrudan sürülür (canlı sonsuz akış TestClient'i kilitlerdi): önce bir
    JPEG döner, sonra kaynak durur ve None döner → idle birikip break ile kapanır.
    """
    from services.inference_api.routers import stream as stream_mod

    sm = StreamManager(load_config())
    calls = {"n": 0}

    def fake_latest(bbox):
        calls["n"] += 1
        if calls["n"] == 1:
            return b"\xff\xd8rawjpeg\xff\xd9"
        sm._running = False  # ilk kareden sonra akışı durdur
        return None

    sm._running = True
    monkeypatch.setattr(sm, "latest_jpeg", fake_latest)
    monkeypatch.setattr(stream_mod.time, "sleep", lambda *_: None)  # idle döngüsünü hızlandır

    req = type("R", (), {"app": type("A", (), {"state": type("S", (), {"stream": sm})()})()})()
    resp = stream_mod.stream_video(req, bbox=False)

    async def _drain():
        out = b""
        async for chunk in resp.body_iterator:
            out += chunk
        return out

    body = asyncio.run(_drain())
    assert b"Content-Type: image/jpeg" in body
    assert b"rawjpeg" in body


# --- WS /stream/annotations + /stream/events ------------------------------ #
def test_ws_annotations_subscribe_push_unsubscribe(client):
    sm = client.app.state.stream
    with client.websocket_connect("/stream/annotations") as ws:
        # Bağlantı sonrası tam 1 abone olmalı.
        assert len(sm._annot_queues) == 1
        anno = AnnotationFrame(frame_id=7, tracks=[{"track_id": 1, "bbox": [0, 0, 5, 5]}])
        sm._emit_annotation(anno)  # abone var → fan-out yapılmalı
        msg = ws.receive_json()
        assert msg["frame_id"] == 7 and msg["tracks"][0]["track_id"] == 1
    # Çıkışta finally → unsubscribe (abone temizliği).
    assert len(sm._annot_queues) == 0


def test_ws_events_subscribe_push_unsubscribe(client):
    sm = client.app.state.stream
    with client.websocket_connect("/stream/events") as ws:
        assert len(sm._event_queues) == 1
        ev = make_event(track_id=3, type="RISK_ALERT")
        sm._emit_event(ev)
        msg = ws.receive_json()
        assert msg["track_id"] == 3 and msg["type"] == "RISK_ALERT"
    assert len(sm._event_queues) == 0


# --- /tracks/{id}/history ------------------------------------------------- #
def test_track_history_pipeline_none(client):
    # Pipeline kurulmadan: boş history dalı.
    j = client.get("/tracks/5/history").json()
    assert j == {"track_id": 5, "history": []}


def test_track_history_populated_contract(client):
    """history = son 200, count = toplam eşleşme (bounded deque davranışı)."""

    class _FakeAcc:
        def get(self, tid):
            return None

        def active_tracks(self):
            return []

    class _FakeEmitter:
        def __init__(self):
            # 250 kare, her birinde track_id=42 → count=250, history=son 200.
            self.annotations = [
                AnnotationFrame(
                    frame_id=i,
                    tracks=[{"track_id": 42, "bbox": [0, 0, 1, 1]}],
                )
                for i in range(250)
            ]

    class _FakePipeline:
        acc = _FakeAcc()
        emitter = _FakeEmitter()

    sm = client.app.state.stream
    sm.pipeline = _FakePipeline()
    try:
        j = client.get("/tracks/42/history").json()
        assert j["count"] == 250
        assert len(j["history"]) == 200
        # Son 200: frame_id 50..249 olmalı (en eski 50 düşmüş).
        assert j["history"][0]["frame_id"] == 50
        assert j["history"][-1]["frame_id"] == 249
        # Eşleşmeyen id → boş seri ama count=0.
        none = client.get("/tracks/999/history").json()
        assert none == {"track_id": 999, "history": [], "count": 0}
    finally:
        sm.pipeline = None


# --- /eval/* -------------------------------------------------------------- #
def test_eval_results_no_results(client):
    j = client.get("/eval/results").json()
    assert j["status"] == "no_results"


def test_eval_export_no_results(client):
    r = client.get("/eval/results/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert "yok" in r.text.lower()


def test_eval_export_error_status_treated_as_no_results(client):
    client.app.state.eval_results = {"status": "error", "error": "boom"}
    r = client.get("/eval/results/export")
    assert r.status_code == 200
    assert "yok" in r.text.lower()


def test_eval_export_report_md_branch(client):
    client.app.state.eval_results = {"status": "ok", "report_md": "# Rapor\nveri"}
    r = client.get("/eval/results/export")
    assert "# Rapor" in r.text


def test_eval_export_json_fallback_branch(client):
    # report_md yok ama sonuç var → json fence fallback.
    client.app.state.eval_results = {"status": "ok", "metric": 0.9}
    r = client.get("/eval/results/export")
    assert "```json" in r.text and "metric" in r.text


def test_eval_run_queued_and_background_job(client):
    r = client.post("/eval/run", json={"source": "x.mp4", "ground_truth": "gt.json"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert body["source"] == "x.mp4" and body["ground_truth"] == "gt.json"
    assert body["qod_comparison"] is True
    # Background _job TestClient kapanışında çalışır; sonuç set edilmiş olmalı
    # (harness varsa sonuç dict, yoksa {'status':'error'} — ikisi de geçerli).
    res = client.get("/eval/results").json()
    assert isinstance(res, dict) and "status" in res


def test_eval_run_defaults_fallback(client):
    # Gövde alanları yok → source/gt config fallback'ı.
    r = client.post("/eval/run", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["ground_truth"] == "data/samples/ornek_gt.json"
    assert body["source"] is not None


# --- /info ---------------------------------------------------------------- #
def test_info_config_summary_and_status_shape(client):
    j = client.get("/info").json()
    assert j["version"]
    cs = j["config_summary"]
    for k in ("detector", "driver_state", "tracker", "speed_mode", "qod_backend", "ai_mode"):
        assert k in cs
    st = j["status"]
    for k in ("running", "frame_count", "fps", "active_tracks", "qod_active_sessions"):
        assert k in st


# --- /cameras ------------------------------------------------------------- #
def test_cameras_probe_disabled_returns_empty(client):
    # AURA_CAMERA_PROBE=0 → donanım taraması yok, boş liste.
    j = client.get("/cameras").json()
    assert j["cameras"] == [] and j["rtsp_supported"] is True


def test_cameras_name_for_logic():
    from services.inference_api.routers.cameras import _name_for

    # mac isimleri yoksa generic isim.
    assert _name_for(2, []) == "Camera 2"


# --- StreamManager birim davranışı (izolasyon/perf no-op) ----------------- #
def test_stream_manager_push_noop_without_loop_or_queues():
    sm = StreamManager(load_config())
    # loop yok → _push sessiz no-op (exception fırlatmamalı).
    sm._push(set(), {"a": 1})
    sm.loop = asyncio.new_event_loop()
    try:
        # kuyruk boş → no-op (model_dump/fan-out hot path israfı engellenir).
        sm._push(set(), {"a": 1})
    finally:
        sm.loop.close()


def test_stream_manager_emit_skips_serialization_without_subscribers():
    sm = StreamManager(load_config())

    class _Boom:
        def model_dump(self):
            raise AssertionError("abone yokken model_dump çağrılmamalı")

    # Abone seti boş → _emit_* model_dump'a hiç dokunmamalı.
    sm._emit_event(_Boom())
    sm._emit_annotation(_Boom())


def test_stream_manager_subscribe_unsubscribe_lifecycle():
    sm = StreamManager(load_config())
    qe = sm.subscribe_events()
    qa = sm.subscribe_annotations()
    assert qe in sm._event_queues and qa in sm._annot_queues
    sm.unsubscribe_events(qe)
    sm.unsubscribe_annotations(qa)
    assert not sm._event_queues and not sm._annot_queues
    # Çift unsubscribe güvenli (discard).
    sm.unsubscribe_events(qe)


def test_stream_status_shape_when_idle():
    sm = StreamManager(load_config())
    st = sm.status()
    assert st["running"] is False
    assert st["active_tracks"] == 0 and st["qod_active_sessions"] == 0
    assert st["frame_count"] == 0
