"""API sözleşmeleri — inference_api + qod_mock + nv_mock (TestClient, model gerektirmez)."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("AURA_AUTOSTART", "0")  # lifespan otomatik stream başlatmasın
os.environ.setdefault("AURA_CAMERA_PROBE", "0")  # donanım kamerası taranmasın

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from services.inference_api.main import create_app  # noqa: E402
from services.nv_mock.main import app as nv_app  # noqa: E402
from services.qod_mock.main import app as qod_app  # noqa: E402

SAMPLE = Path("data/samples/ornek.mp4")


def _ensure_sample():
    if not SAMPLE.exists():
        from aura.synthetic import generate

        generate(SAMPLE.parent, 90, 30, 640, 360)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope="module")
def qod():
    with TestClient(qod_app) as c:
        yield c


@pytest.fixture(scope="module")
def nv():
    with TestClient(nv_app) as c:
        yield c


# --- inference_api ------------------------------------------------------- #
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["service"] == "inference_api"


def test_info_and_cameras(client):
    assert client.get("/info").status_code == 200
    j = client.get("/cameras").json()
    assert "cameras" in j and j["rtsp_supported"] is True


def test_config_get_patch(client):
    assert "runtime" in client.get("/config").json()
    r = client.patch("/config", json={"conf_threshold": 0.5})
    assert r.status_code == 200 and r.json()["status"] == "updated"


def test_stream_lifecycle_produces_tracks(client):
    _ensure_sample()
    r = client.post("/stream/start", json={"source": str(SAMPLE), "bbox_overlay": True})
    assert r.status_code == 200 and r.json()["status"] == "started"
    fc = 0
    deadline = time.time() + 8
    while time.time() < deadline:
        st = client.get("/stream/status").json()
        fc = st["frame_count"]
        if fc > 0 and st["active_tracks"] > 0:
            break
        time.sleep(0.1)
    assert fc > 0
    tr = client.get("/tracks").json()
    assert tr["count"] >= 1
    tid = tr["tracks"][0]["track_id"]
    assert client.get(f"/tracks/{tid}").status_code == 200
    assert client.get("/tracks/999999").status_code == 404
    assert client.post("/stream/stop").json()["status"] == "stopped"


# --- qod_mock (CAMARA sözleşmesi) ---------------------------------------- #
def test_qod_session_lifecycle(qod):
    assert qod.get("/health").json()["status"] == "ok"
    r = qod.post("/sessions", json={"profile": "HIGH_THROUGHPUT", "device_id": "dev1"})
    assert r.status_code == 201
    body = r.json()
    sid = body["session_id"]
    assert body["granted_profile"] == "HIGH_THROUGHPUT" and body["status"] == "ACTIVE"
    assert qod.get(f"/sessions/{sid}").status_code == 200
    assert qod.get("/sessions").json()["count"] >= 1
    assert qod.delete(f"/sessions/{sid}").status_code == 200
    assert qod.get(f"/sessions/{sid}").status_code == 404


# --- nv_mock (Number Verification) --------------------------------------- #
def test_nv_verify_silent(nv):
    ok = nv.post("/verify", json={"phone_number": "+90 555 111 2233", "sim_token": "tok"}).json()
    assert ok["verified"] is True and ok["latency_ms"] >= 0
    bad = nv.post("/verify", json={"phone_number": "+1 555 111 2233", "sim_token": "tok"}).json()
    assert bad["verified"] is False
    no_token = nv.post("/verify", json={"phone_number": "+905551112233"}).json()
    assert no_token["verified"] is False
