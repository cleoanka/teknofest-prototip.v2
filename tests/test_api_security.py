"""SEC-001/002/003 guvenlik testleri (inference_api).

DEMO-KORUMA dogrulamasi: token UNSET iken mutasyon uclari hala 200 doner
(yerel demo bozulmaz). Token SET iken 401/200 davranisi, SSRF source 400 ve
ground_truth path-traversal 400 dogrulanir. Tum testler model gerektirmez
(AI_MODE=mock, autostart kapali); start kaynak DOGRULAMASI worker'dan once
istek thread'inde calistigi icin agir model yuklemez.
"""

from __future__ import annotations

import os

os.environ.setdefault("ROADGUARD_AUTOSTART", "0")
os.environ.setdefault("ROADGUARD_CAMERA_PROBE", "0")
os.environ.setdefault("AI_MODE", "mock")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from services.inference_api.main import create_app  # noqa: E402
from services.inference_api.security import cors_origins, validate_source  # noqa: E402

_TOKEN = "s3cr3t-token"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def with_token(monkeypatch):
    monkeypatch.setenv("ROADGUARD_API_TOKEN", _TOKEN)


@pytest.fixture
def with_protect_reads(monkeypatch):
    """OPT-IN PII okuma korumasi: token + ROADGUARD_API_PROTECT_READS birlikte set."""
    monkeypatch.setenv("ROADGUARD_API_TOKEN", _TOKEN)
    monkeypatch.setenv("ROADGUARD_API_PROTECT_READS", "1")


# --- SEC-001: token auth (ENV-GATED) ------------------------------------- #
def test_mutation_open_when_token_unset(client, monkeypatch):
    """Token UNSET → mutasyon uclari acik (yerel demo bozulmaz)."""
    monkeypatch.delenv("ROADGUARD_API_TOKEN", raising=False)
    r = client.patch("/config", json={"conf_threshold": 0.5})
    assert r.status_code == 200
    assert client.post("/stream/stop").status_code == 200
    assert client.patch("/stream/config", json={"bbox_overlay": True}).status_code == 200


def test_mutation_401_when_token_set_and_missing(client, with_token):
    """Token SET ama baslik yok → 401."""
    assert client.patch("/config", json={"conf_threshold": 0.4}).status_code == 401
    assert client.post("/stream/stop").status_code == 401
    assert client.patch("/stream/config", json={"bbox_overlay": True}).status_code == 401
    assert client.post("/eval/run", json={}).status_code == 401


def test_mutation_401_when_token_wrong(client, with_token):
    r = client.patch("/config", json={"conf_threshold": 0.4}, headers={"X-RoadGuard-Token": "nope"})
    assert r.status_code == 401


def test_mutation_200_when_token_correct(client, with_token):
    h = {"X-RoadGuard-Token": _TOKEN}
    assert client.patch("/config", json={"conf_threshold": 0.4}, headers=h).status_code == 200
    assert client.post("/stream/stop", headers=h).status_code == 200
    assert client.patch("/stream/config", json={"bbox_overlay": True}, headers=h).status_code == 200


def test_read_endpoints_unauthenticated_ok(client, with_token):
    """Okuma uclari token SET olsa bile acik kalir (sozlesme korunur)."""
    assert client.get("/config").status_code == 200
    assert client.get("/stream/status").status_code == 200


def test_pii_reads_open_by_default_even_with_token(client, with_token):
    """Geriye uyum: PROTECT_READS YOK iken token set olsa bile PII okuma uclari (tracks/
    info) ACIK (co-located dashboard sozlesmesi). Yalniz opt-in bayrak bunu kapatir."""
    assert client.get("/tracks").status_code == 200
    assert client.get("/info").status_code == 200


def test_protect_reads_optin_blocks_pii(client, with_protect_reads):
    """OPT-IN (ROADGUARD_API_PROTECT_READS=1): token'siz PII okuma -> 401; header VEYA
    ?token= query-param (MJPEG <img> baslik gonderemez) ile -> gecer."""
    # token YOK -> 401 (PII sizdirilmaz)
    assert client.get("/tracks").status_code == 401
    assert client.get("/tracks/1").status_code == 401
    assert client.get("/info").status_code == 401
    assert client.get("/stream/video").status_code == 401
    # header token -> gecer
    h = {"X-RoadGuard-Token": _TOKEN}
    assert client.get("/tracks", headers=h).status_code == 200
    assert client.get("/info", headers=h).status_code == 200
    # MJPEG: query-param token -> gecer (200), yanlis -> 401
    assert client.get("/stream/video", params={"token": _TOKEN}).status_code == 200
    assert client.get("/stream/video", params={"token": "nope"}).status_code == 401
    # mutasyon ucu hala header token ister (degismedi)
    assert client.patch("/config", json={"conf_threshold": 0.4}, headers=h).status_code == 200


def test_cors_not_wildcard():
    origins = cors_origins()
    assert "*" not in origins
    assert "http://localhost:8080" in origins


def test_cors_env_extend(monkeypatch):
    monkeypatch.setenv("ROADGUARD_CORS_ORIGINS", "https://demo.example.com")
    assert "https://demo.example.com" in cors_origins()


# --- SEC-002: SSRF source guard ------------------------------------------ #
def test_ssrf_http_source_rejected(client, monkeypatch):
    monkeypatch.delenv("ROADGUARD_API_TOKEN", raising=False)
    monkeypatch.delenv("ROADGUARD_ALLOW_NET_SOURCE", raising=False)
    r = client.post("/stream/start", json={"source": "http://169.254.169.254/latest/meta-data"})
    assert r.status_code == 400
    r2 = client.post("/eval/run", json={"source": "http://169.254.169.254/"})
    assert r2.status_code == 400


def test_ssrf_file_scheme_rejected(client, monkeypatch):
    monkeypatch.delenv("ROADGUARD_API_TOKEN", raising=False)
    r = client.post("/stream/start", json={"source": "file:///etc/passwd"})
    assert r.status_code == 400


def test_source_traversal_rejected(client, monkeypatch):
    monkeypatch.delenv("ROADGUARD_API_TOKEN", raising=False)
    r = client.post("/stream/start", json={"source": "../../../../etc/passwd"})
    assert r.status_code == 400


def test_camera_index_allowed():
    assert validate_source("0") == "0"
    assert validate_source("12") == "12"


def test_rtsp_allowed():
    assert validate_source("rtsp://cam.local/stream") == "rtsp://cam.local/stream"


def test_net_source_opt_in(monkeypatch):
    monkeypatch.setenv("ROADGUARD_ALLOW_NET_SOURCE", "1")
    assert validate_source("http://example.com/x.mp4") == "http://example.com/x.mp4"


# --- SEC-003: ground_truth path-traversal guard -------------------------- #
def test_gt_traversal_rejected(client, monkeypatch):
    monkeypatch.delenv("ROADGUARD_API_TOKEN", raising=False)
    r = client.post("/eval/run", json={"ground_truth": "../../../../etc/passwd"})
    assert r.status_code == 400


def test_gt_absolute_outside_rejected(client, monkeypatch):
    monkeypatch.delenv("ROADGUARD_API_TOKEN", raising=False)
    r = client.post("/eval/run", json={"ground_truth": "/etc/passwd"})
    assert r.status_code == 400


def test_gt_inside_data_dir_accepted(client, monkeypatch):
    """data/ altindaki GT kabul edilir (queued)."""
    monkeypatch.delenv("ROADGUARD_API_TOKEN", raising=False)
    r = client.post("/eval/run", json={"ground_truth": "data/samples/ornek_gt.json"})
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
