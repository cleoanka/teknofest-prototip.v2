"""S3-tracks-eval kapsamlı endpoint/birim testleri (TestClient, mock-mod, model gerekmez).

Kapsanan boşluklar (review test_gaps, test_api_core_coverage'da OLMAYANLAR):
  - /tracks (list_tracks): pipeline None → {tracks:[],count:0}; dolu → TrackRecord şeması
  - /tracks/{id} (get_track): pipeline None → 404 (KASITLI sözleşme); var-track → 200 + tam şema
  - /tracks/{id}/history: track_id yol-parametre tip-zorlaması (numerik-olmayan → 422)
  - /eval/run: queued yanıtı + echo (source/ground_truth/qod_comparison) + _job'un
    harness'i çağırıp eval_results'a yazması; harness HATA dalı (import/çalışma) →
    eval_results={'status':'error'} ve ardından export'un '# Eval sonucu yok' dönmesi
  - /eval/run qod_comparison=False echo'su
"""

from __future__ import annotations

import os

os.environ.setdefault("AURA_AUTOSTART", "0")
os.environ.setdefault("AURA_CAMERA_PROBE", "0")
os.environ.setdefault("AI_MODE", "mock")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from aura.schema import AnnotationFrame, BBox, TrackRecord  # noqa: E402
from services.inference_api.main import create_app  # noqa: E402


@pytest.fixture
def client():
    # Fonksiyon-kapsamlı: her test taze app/StreamManager → state sızıntısı yok.
    with TestClient(create_app()) as c:
        yield c


# --- Sahte pipeline yardımcıları ----------------------------------------- #
class _FakeAcc:
    def __init__(self, records=None):
        self._records = records or {}

    def active_tracks(self):
        return list(self._records.values())

    def get(self, tid):
        return self._records.get(tid)


class _FakeEmitter:
    def __init__(self, annotations=None):
        self.annotations = annotations or []


class _FakePipeline:
    def __init__(self, records=None, annotations=None):
        self.acc = _FakeAcc(records)
        self.emitter = _FakeEmitter(annotations)


def _rec(track_id=1):
    return TrackRecord(track_id=track_id, bbox=BBox(x1=0, y1=0, x2=10, y2=10))


# --- /tracks (list_tracks) ----------------------------------------------- #
def test_list_tracks_pipeline_none_empty(client):
    # Pipeline kurulmadan: boş liste, count=0 (404 DEĞİL — koleksiyon semantiği).
    j = client.get("/tracks").json()
    assert j == {"tracks": [], "count": 0}


def test_list_tracks_populated_schema(client):
    sm = client.app.state.stream
    sm.pipeline = _FakePipeline(records={1: _rec(1), 2: _rec(2)})
    try:
        r = client.get("/tracks")
        assert r.status_code == 200
        j = r.json()
        assert j["count"] == 2 and len(j["tracks"]) == 2
        # Her eleman TrackRecord.model_dump() şeması olmalı (anahtar varlığı).
        for t in j["tracks"]:
            for k in ("track_id", "vehicle_class", "bbox", "plate", "driver", "speed"):
                assert k in t
    finally:
        sm.pipeline = None


# --- /tracks/{id} (get_track) -------------------------------------------- #
def test_get_track_pipeline_none_404(client):
    # KASITLI sözleşme: pipeline yokken tekil sorgu 404 (track yok) döner.
    r = client.get("/tracks/7")
    assert r.status_code == 404
    assert r.json()["detail"] == "track not found"


def test_get_track_missing_id_404(client):
    sm = client.app.state.stream
    sm.pipeline = _FakePipeline(records={1: _rec(1)})
    try:
        assert client.get("/tracks/999").status_code == 404
    finally:
        sm.pipeline = None


def test_get_track_found_full_schema(client):
    sm = client.app.state.stream
    sm.pipeline = _FakePipeline(records={42: _rec(42)})
    try:
        r = client.get("/tracks/42")
        assert r.status_code == 200
        body = r.json()
        assert body["track_id"] == 42
        # model_dump tam şema: iç içe state nesneleri serileşmeli.
        assert body["bbox"]["x2"] == 10
        for k in ("plate", "driver", "speed", "risk_flags", "qod_active"):
            assert k in body
    finally:
        sm.pipeline = None


# --- /tracks/{id}/history tip-zorlaması ---------------------------------- #
def test_track_history_non_numeric_id_422(client):
    # Yol parametresi int; numerik-olmayan ID FastAPI varsayılanı ile 422.
    assert client.get("/tracks/abc/history").status_code == 422


def test_get_track_non_numeric_id_422(client):
    assert client.get("/tracks/abc").status_code == 422


def test_track_history_populated_spread_schema(client):
    """history kayıtları frame_id/ts/track_id ve track alanlarını yaymalı."""
    sm = client.app.state.stream
    anns = [
        AnnotationFrame(
            frame_id=i,
            ts=float(i),
            tracks=[{"track_id": 5, "bbox": [0, 0, 1, 1]}],
        )
        for i in range(3)
    ]
    sm.pipeline = _FakePipeline(records={}, annotations=anns)
    try:
        j = client.get("/tracks/5/history").json()
        assert j["track_id"] == 5 and j["count"] == 3
        assert len(j["history"]) == 3
        first = j["history"][0]
        for k in ("frame_id", "ts", "track_id", "bbox"):
            assert k in first
        assert first["track_id"] == 5
    finally:
        sm.pipeline = None


# --- /eval/run ------------------------------------------------------------ #
def test_eval_run_queued_echo_and_job_writes_results(client, monkeypatch):
    """queued yanıtı + echo; _job harness'i çağırıp eval_results'a yazmalı."""
    import services.inference_api.routers.eval as eval_mod

    captured = {}

    def fake_run_eval(cfg, source, gt, qod_comparison):
        captured["args"] = (source, gt, qod_comparison)
        return {"status": "ok", "report_md": "# OK", "from_fake": True}

    # harness modülünü monkeypatch'le (lazy import bu modülü hedefler).
    import types

    fake_harness = types.ModuleType("aura.eval.harness")
    fake_harness.run_eval = fake_run_eval
    monkeypatch.setitem(__import__("sys").modules, "aura.eval.harness", fake_harness)

    r = client.post(
        "/eval/run",
        json={"source": "vid.mp4", "ground_truth": "gt.json", "qod_comparison": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "status": "queued",
        "source": "vid.mp4",
        "ground_truth": "gt.json",
        "qod_comparison": True,
    }
    # Background _job request response sonrası çalıştı → eval_results yazıldı.
    assert captured["args"] == ("vid.mp4", "gt.json", True)
    res = client.get("/eval/results").json()
    assert res.get("from_fake") is True and res["status"] == "ok"
    # eval_mod referansı kullanılıyor (import lint'i için).
    assert hasattr(eval_mod, "eval_run")


def test_eval_run_qod_comparison_false_echo(client, monkeypatch):
    import types

    fake_harness = types.ModuleType("aura.eval.harness")
    fake_harness.run_eval = lambda *a, **k: {"status": "ok"}
    monkeypatch.setitem(__import__("sys").modules, "aura.eval.harness", fake_harness)
    r = client.post("/eval/run", json={"qod_comparison": False})
    assert r.status_code == 200
    assert r.json()["qod_comparison"] is False


def test_eval_run_harness_error_path_sets_error_then_export_empty(client, monkeypatch):
    """harness HATA dalı: _job exception → eval_results={'status':'error'};
    ardından /eval/results/export '# Eval sonucu yok' döner."""
    import types

    def boom(*a, **k):
        raise RuntimeError("harness patladi")

    fake_harness = types.ModuleType("aura.eval.harness")
    fake_harness.run_eval = boom
    monkeypatch.setitem(__import__("sys").modules, "aura.eval.harness", fake_harness)

    r = client.post("/eval/run", json={"source": "x.mp4"})
    assert r.status_code == 200 and r.json()["status"] == "queued"
    # _job hata yakaladı → error state.
    res = client.get("/eval/results").json()
    assert res["status"] == "error" and "harness patladi" in res["error"]
    # error state'te export markdown'u '# Eval sonucu yok' olmalı.
    exp = client.get("/eval/results/export")
    assert exp.status_code == 200
    assert exp.headers["content-type"].startswith("text/markdown")
    assert "yok" in exp.text.lower()


def test_eval_run_defaults_source_and_gt(client, monkeypatch):
    """source verilmezse cfg fallback; gt verilmezse modül-sabit _DEFAULT_GT."""
    import types

    import services.inference_api.routers.eval as eval_mod

    fake_harness = types.ModuleType("aura.eval.harness")
    fake_harness.run_eval = lambda *a, **k: {"status": "ok"}
    monkeypatch.setitem(__import__("sys").modules, "aura.eval.harness", fake_harness)

    r = client.post("/eval/run", json={})
    body = r.json()
    assert body["ground_truth"] == eval_mod._DEFAULT_GT
    assert body["source"] is not None
