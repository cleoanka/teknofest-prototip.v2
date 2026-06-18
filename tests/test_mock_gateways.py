"""HTTP-seviye kapsamlı testler: qod_mock (CAMARA QoD) + nv_mock (Number Verification).

TestClient ile çalışır, gerçek model/ağ gerektirmez. Sözleşme yüzeyi (status kod +
response şema) + hata yolları + kenar durumları kapsanır.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.nv_mock.main import app as nv_app
from services.qod_mock.main import _sessions
from services.qod_mock.main import app as qod_app

# Tüm SessionResponse anahtarları — POST/GET/LIST/DELETE arasında tutarlı olmalı.
_SESSION_KEYS = {"session_id", "status", "granted_profile", "device_id", "created_at"}


@pytest.fixture
def qod():
    _sessions.clear()
    with TestClient(qod_app) as c:
        yield c
    _sessions.clear()


@pytest.fixture(scope="module")
def nv():
    with TestClient(nv_app) as c:
        yield c


# --- qod_mock ------------------------------------------------------------- #
def test_qod_health_active_sessions_counts(qod):
    h = qod.get("/health").json()
    assert h == {"status": "ok", "service": "qod_mock", "active_sessions": 0}
    qod.post("/sessions", json={"profile": "LOW_LATENCY", "device_id": "d"})
    qod.post("/sessions", json={"profile": "HIGH_THROUGHPUT", "device_id": "d"})
    assert qod.get("/health").json()["active_sessions"] == 2


def test_qod_create_low_latency_echoes_profile(qod):
    r = qod.post("/sessions", json={"profile": "LOW_LATENCY", "device_id": "dev-ll"})
    assert r.status_code == 201
    body = r.json()
    assert body["granted_profile"] == "LOW_LATENCY"
    assert body["status"] == "ACTIVE"
    assert body["device_id"] == "dev-ll"
    assert isinstance(body["created_at"], (int, float))
    assert body["session_id"]


def test_qod_create_high_throughput_echoes_profile(qod):
    r = qod.post("/sessions", json={"profile": "HIGH_THROUGHPUT", "device_id": "dev-ht"})
    assert r.status_code == 201
    assert r.json()["granted_profile"] == "HIGH_THROUGHPUT"


def test_qod_response_schema_consistent_across_endpoints(qod):
    """Aynı kaynak POST/GET/LIST/DELETE'te aynı anahtar kümesini döndürmeli (Bug 1)."""
    post_body = qod.post(
        "/sessions", json={"profile": "LOW_LATENCY", "device_id": "d", "duration_seconds": 120}
    ).json()
    sid = post_body["session_id"]
    assert set(post_body.keys()) == _SESSION_KEYS

    get_body = qod.get(f"/sessions/{sid}").json()
    assert set(get_body.keys()) == _SESSION_KEYS

    list_body = qod.get("/sessions").json()
    assert set(list_body["sessions"][0].keys()) == _SESSION_KEYS

    del_body = qod.delete(f"/sessions/{sid}").json()
    assert set(del_body.keys()) == _SESSION_KEYS
    # duration_seconds artık hiçbir response'ta sızmıyor.
    assert "duration_seconds" not in get_body
    assert "duration_seconds" not in list_body["sessions"][0]


def test_qod_list_shape_and_count(qod):
    assert qod.get("/sessions").json() == {"sessions": [], "count": 0}
    qod.post("/sessions", json={"profile": "LOW_LATENCY", "device_id": "a"})
    qod.post("/sessions", json={"profile": "HIGH_THROUGHPUT", "device_id": "b"})
    lst = qod.get("/sessions").json()
    assert lst["count"] == 2
    assert len(lst["sessions"]) == 2


def test_qod_get_missing_returns_404(qod):
    r = qod.get("/sessions/does-not-exist")
    assert r.status_code == 404
    assert r.json()["detail"] == "session not found"


def test_qod_delete_missing_returns_404(qod):
    r = qod.delete("/sessions/does-not-exist")
    assert r.status_code == 404
    assert r.json()["detail"] == "session not found"


def test_qod_delete_marks_status_deleted_and_removes(qod):
    sid = qod.post("/sessions", json={"profile": "LOW_LATENCY", "device_id": "d"}).json()[
        "session_id"
    ]
    del_body = qod.delete(f"/sessions/{sid}").json()
    assert del_body["status"] == "DELETED"
    assert del_body["session_id"] == sid
    # idempotent değil: ikinci silme 404.
    assert qod.delete(f"/sessions/{sid}").status_code == 404
    assert qod.get(f"/sessions/{sid}").status_code == 404
    assert qod.get("/sessions").json()["count"] == 0


def test_qod_missing_required_field_returns_422(qod):
    # device_id zorunlu (default yok) -> 422.
    assert qod.post("/sessions", json={"profile": "LOW_LATENCY"}).status_code == 422
    # profile zorunlu.
    assert qod.post("/sessions", json={"device_id": "d"}).status_code == 422


def test_qod_duration_defaults_when_omitted(qod):
    # duration_seconds opsiyonel; eksikse 201 (sözleşme kabul ediyor).
    r = qod.post("/sessions", json={"profile": "LOW_LATENCY", "device_id": "d"})
    assert r.status_code == 201


def test_qod_unique_session_ids(qod):
    s1 = qod.post("/sessions", json={"profile": "LOW_LATENCY", "device_id": "d"}).json()
    s2 = qod.post("/sessions", json={"profile": "LOW_LATENCY", "device_id": "d"}).json()
    assert s1["session_id"] != s2["session_id"]


# --- nv_mock -------------------------------------------------------------- #
def test_nv_health(nv):
    assert nv.get("/health").json() == {"status": "ok", "service": "nv_mock"}


def test_nv_verified_plus90(nv):
    r = nv.post("/verify", json={"phone_number": "+90 555 111 2233", "sim_token": "tok"})
    body = r.json()
    assert body["verified"] is True
    assert body["latency_ms"] == 40
    # phone_number HAM (normalize edilmemiş) input'u yansıtır.
    assert body["phone_number"] == "+90 555 111 2233"


def test_nv_verified_bare_90_prefix(nv):
    body = nv.post("/verify", json={"phone_number": "905551112233", "sim_token": "t"}).json()
    assert body["verified"] is True


def test_nv_verified_zero_prefix(nv):
    body = nv.post("/verify", json={"phone_number": "05551112233", "sim_token": "t"}).json()
    assert body["verified"] is True


def test_nv_token_present_non_turkish_prefix_false(nv):
    # AND kısa-devre: token var ama prefix Türk değil -> False.
    body = nv.post("/verify", json={"phone_number": "+1 555 111 2233", "sim_token": "t"}).json()
    assert body["verified"] is False


def test_nv_no_token_false_even_if_turkish(nv):
    body = nv.post("/verify", json={"phone_number": "+905551112233"}).json()
    assert body["verified"] is False
    body2 = nv.post("/verify", json={"phone_number": "+905551112233", "sim_token": ""}).json()
    assert body2["verified"] is False


def test_nv_empty_phone_false(nv):
    body = nv.post("/verify", json={"phone_number": "", "sim_token": "t"}).json()
    assert body["verified"] is False
    assert body["phone_number"] == ""


def test_nv_latency_constant(nv):
    body = nv.post("/verify", json={"phone_number": "+90555", "sim_token": "t"}).json()
    assert body["latency_ms"] == 40


def test_nv_phone_number_echoes_raw_input(nv):
    raw = "+90 5 5 5"
    body = nv.post("/verify", json={"phone_number": raw, "sim_token": "t"}).json()
    assert body["phone_number"] == raw


def test_nv_missing_phone_returns_422(nv):
    assert nv.post("/verify", json={"sim_token": "t"}).status_code == 422


def test_nv_loose_prefix_false_positives_documented(nv):
    """Mock kuralı: bare '90'/'0' ile başlayan her numara Türk sayılır (bilinen gevşeklik)."""
    assert (
        nv.post("/verify", json={"phone_number": "901234", "sim_token": "t"}).json()["verified"]
        is True
    )
    assert (
        nv.post("/verify", json={"phone_number": "0555", "sim_token": "t"}).json()["verified"]
        is True
    )
