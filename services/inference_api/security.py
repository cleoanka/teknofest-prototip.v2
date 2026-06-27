"""inference_api guvenlik katmani — SEC-001/002/003 remediation.

Tasarim ilkesi (DEMO-KORUMA): tum sertlestirmeler ENV-GATED. Varsayilan
(env set degil) durumda davranis yerel demoyla birebir aynidir; uretim/saha
icin ilgili ortam degiskenleri set edilince koruma devreye girer.

- SEC-001: ``ROADGUARD_API_TOKEN`` set ise mutasyon uclari ``X-RoadGuard-Token`` ister.
  CORS allowlist ``ROADGUARD_CORS_ORIGINS`` ile yapilandirilir (varsayilan localhost).
- SEC-002: kaynak (camera index | rtsp:// | ROOT altinda dosya) dogrulamasi;
  http://, file:// ve serbest semalar varsayilan kapali, ``ROADGUARD_ALLOW_NET_SOURCE``
  ile acilir → SSRF kapatilir.
- SEC-003: ground_truth path-traversal guard (izinli dizine resolve + commonpath).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import Header, HTTPException, Query, status

from roadguard.config import ROOT

_TOKEN_HEADER = "X-RoadGuard-Token"
_CAMERA_INDEX_RE = re.compile(r"^\d{1,2}$")

# Path-traversal guard'lari icin izinli kok dizin (repo koku, sunucu sabiti).
# ground_truth ve dosya-kaynaklar bu agacin DISINA cikamaz; ic-iceki goreli
# yollar (or. "gt.json", "data/samples/...") aynen gecerli sayilir → mevcut
# API sozlesmesi (echo + run_eval argumanlari ham deger) korunur.
_ALLOWED_ROOT = ROOT.resolve()


def _expected_token() -> str | None:
    """Beklenen token; ``ROADGUARD_API_TOKEN`` bos/unset ise None (auth kapali)."""
    tok = os.environ.get("ROADGUARD_API_TOKEN")
    return tok if tok else None


def verify_token(x_roadguard_token: str | None = Header(default=None)) -> None:
    """ENV-GATED token auth (FastAPI dependency).

    ``ROADGUARD_API_TOKEN`` set DEGILSE: hicbir kontrol yok (yerel demo bozulmaz).
    Set ISE: ``X-RoadGuard-Token`` basligi token ile birebir eslesmeli, aksi halde 401.
    """
    expected = _expected_token()
    if expected is None:
        return
    if not x_roadguard_token or x_roadguard_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Gecersiz veya eksik {_TOKEN_HEADER}",
            headers={"WWW-Authenticate": _TOKEN_HEADER},
        )


def _read_auth_enabled() -> bool:
    """Okuma-ucu korumasi yalnız ``ROADGUARD_API_TOKEN`` VE ``ROADGUARD_API_PROTECT_READS``
    set ise aktiftir.

    Varsayilan (bayrak yok): okuma uclari ACIK kalir → co-located dashboard ve
    ``test_read_endpoints_unauthenticated_ok`` sozlesmesi korunur. Uretimde PII
    (canli plaka/goruntu) korumasi icin opt-in; tasarim ilkesiyle (DEMO-KORUMA,
    ENV-GATED) tutarli.
    """
    if _expected_token() is None:
        return False
    flag = os.environ.get("ROADGUARD_API_PROTECT_READS", "").strip().lower()
    return flag not in ("", "0", "false", "no", "off")


def verify_token_read(
    x_roadguard_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    """OPT-IN okuma-ucu auth (PII koruma; SEC-001 genisletmesi).

    ``ROADGUARD_API_TOKEN`` + ``ROADGUARD_API_PROTECT_READS`` set ISE: ``X-RoadGuard-Token``
    basligi VEYA ``?token=`` query-param (MJPEG ``<img>`` baslik gonderemez) beklenen
    token ile birebir eslesmeli; aksi halde 401. Bayraklardan biri yoksa no-op
    (varsayilan demo + dashboard + acik-okuma sozlesmesi BOZULMAZ).
    """
    if not _read_auth_enabled():
        return
    expected = _expected_token()
    supplied = x_roadguard_token or token
    if not supplied or supplied != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Gecersiz veya eksik token ({_TOKEN_HEADER} veya ?token=); okuma korumasi acik",
            headers={"WWW-Authenticate": _TOKEN_HEADER},
        )


def cors_origins() -> list[str]:
    """CORS allowlist (varsayilan localhost; ``ROADGUARD_CORS_ORIGINS`` ile genislet).

    Dashboard same-origin sunuldugu icin varsayilan localhost listesi yerel
    demoyu bozmaz; '*' kullanilmaz.
    """
    extra = os.environ.get("ROADGUARD_CORS_ORIGINS", "")
    origins = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    for o in extra.split(","):
        o = o.strip()
        if o and o not in origins:
            origins.append(o)
    return origins


def _net_sources_allowed() -> bool:
    return os.environ.get("ROADGUARD_ALLOW_NET_SOURCE", "0") not in ("0", "", "false", "False")


def _contained_in(path: Path, base: Path) -> bool:
    """``path`` ``base`` altinda mi (commonpath ile, symlink-resolve sonrasi)."""
    try:
        return os.path.commonpath([str(path), str(base)]) == str(base)
    except ValueError:  # farkli surucu/gecersiz → icermez
        return False


def validate_source(source):
    """Akis/eval kaynagini dogrula (SEC-002 SSRF guard).

    Izinli:
      (a) kamera indeksi (``"0"``..``"99"``),
      (b) ``rtsp://`` URL'leri,
      (c) ROOT altina resolve edilip disari cikmadigi teyit edilen dosya yolu.
    Varsayilan KAPALI: ``http://``/``https://``/``file://`` ve diger semalar
    (``ROADGUARD_ALLOW_NET_SOURCE`` set ise ``http(s)/rtmp/rtp`` acilir).

    Dogrulama gecerse girdi DEGISTIRILMEDEN dondurulur (API sozlesmesi: echo ve
    StreamManager.start/run_eval argumanlari ham deger kalir). None/bos → None.
    Gecersiz → HTTP 400.
    """
    if source in (None, ""):
        return None
    if isinstance(source, int):
        return source
    s = str(source).strip()

    if _CAMERA_INDEX_RE.match(s):
        return source

    if "://" in s:
        scheme = s.split("://", 1)[0].lower()
        if scheme == "rtsp":
            return source
        if scheme in ("http", "https", "rtmp", "rtp") and _net_sources_allowed():
            return source
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Izin verilmeyen kaynak semasi: {scheme}:// (SSRF korumasi)",
        )

    # Yerel dosya yolu: ROOT altina resolve edilip disari cikmadigi teyit edilir.
    if not _resolves_inside(s, _ALLOWED_ROOT):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kaynak izinli dizin disinda (path traversal korumasi)",
        )
    return source


def resolve_ground_truth(ground_truth):
    """ground_truth'u path-traversal'a karsi koru (SEC-003 guard).

    Izinli kok (repo koku) altina resolve edilip disari cikmadigi commonpath ile
    teyit edilir; cikan (``../...``, ``/etc/passwd`` vb.) 400 ile reddedilir.
    Gecerli girdi DEGISTIRILMEDEN dondurulur (API sozlesmesi korunur).
    None/bos → None.
    """
    if ground_truth in (None, ""):
        return None
    s = str(ground_truth).strip()
    if not _resolves_inside(s, _ALLOWED_ROOT):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ground_truth izinli dizin disinda (path traversal korumasi)",
        )
    return ground_truth


def _resolves_inside(s: str, base: Path) -> bool:
    """``s`` (goreli → ROOT'a, mutlak → kendi) ``base`` altina resolve oluyor mu."""
    p = Path(s)
    if not p.is_absolute():
        p = ROOT / p
    return _contained_in(p.resolve(), base)
