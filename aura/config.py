"""Merkezi config yükleyici.

`config/default.yaml` tek doğruluk kaynağıdır. Hiçbir eşik/flag koda gömülmez.
Belirli env değişkenleri (AI_MODE, AURA_DEVICE, port'lar) YAML'ı override eder.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"
SAMPLE_VIDEO = ROOT / "data" / "samples" / "ornek.mp4"  # pakete gömülü sentetik demo

log = logging.getLogger("aura.config")


class Config:
    """Noktalı erişimli (`cfg.get("plate.voting_buffer_size")`) ince sözlük sarmalayıcı."""

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self._data = data
        self.path = path

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for key in dotted.split("."):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def as_dict(self) -> dict[str, Any]:
        return self._data


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Seçili env değişkenleri YAML'ı override eder (config/README.md'de belgeli)."""
    ai_mode = os.environ.get("AI_MODE")
    if ai_mode:
        data.setdefault("runtime", {})["ai_mode"] = ai_mode
    device = os.environ.get("AURA_DEVICE")
    if device:
        data.setdefault("runtime", {})["device"] = device

    services = data.setdefault("services", {})
    for env_key, cfg_key in (
        ("AURA_INFERENCE_PORT", "inference_api"),
        ("AURA_QOD_MOCK_PORT", "qod_mock"),
        ("AURA_NV_MOCK_PORT", "nv_mock"),
    ):
        val = os.environ.get(env_key)
        if val and val.isdigit():
            services[cfg_key] = int(val)
    return data


def load_config(path: str | Path | None = None) -> Config:
    """Config'i yükle. `path` verilmezse `config/default.yaml`."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config bulunamadı: {cfg_path}. Önce `python bootstrap.py` çalıştırın."
        )
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = _apply_env_overrides(data)
    return Config(data, cfg_path)


def resolve_source(cfg: Config) -> str | int:
    """`runtime.source`'u açılabilir bir kaynağa çöz.

    - Kamera indeksi (``"0"``) veya URL (``rtsp://...``) → olduğu gibi geçer.
    - Dosya yolu mevcutsa mutlak yol döner.
    - Dosya yolu YOKSA paketteki örnek videoya düşer: sessiz "ölü akış" yerine
      net bir uyarı + çalışan demo. Yapılandırılan dosya sonradan eklenirse
      otomatik olarak ona geri dönülür (config değişmez).
    """
    src = cfg.get("runtime.source", "data/samples/ornek.mp4")
    if isinstance(src, int):
        return src
    s = str(src)
    if s.isdigit() or "://" in s:  # kamera indeksi / akış URL'si → dokunma
        return s
    p = Path(s)
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        return str(p)
    fallback = ROOT / "data" / "samples" / "ornek.mp4"
    if fallback.exists():
        log.warning("Kaynak bulunamadı: %s → örnek videoya düşülüyor: %s", s, fallback)
        return str(fallback)
    log.error("Kaynak bulunamadı ve örnek video da yok: %s", s)
    return s


def is_synthetic_source(cfg: Config) -> bool:
    """Çözülen kaynak, pakete gömülü sentetik örnek video mu?

    Sentetik örnek (koyu asfalt üzerinde renkli bloklar) yalnızca mock dedektörle
    anlamlı tespit üretir; COCO-eğitimli gerçek YOLO bu blokları araç olarak
    GÖRMEZ (0 tespit). `ai_mode=auto` bu durumu algılayıp mock'a düşer — gerçek
    footage/kamera kaynağında ise gerçek YOLO kullanılır.
    """
    try:
        return Path(str(resolve_source(cfg))).resolve() == SAMPLE_VIDEO.resolve()
    except OSError:
        return False
