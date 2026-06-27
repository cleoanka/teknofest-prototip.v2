"""Merkezi config yükleyici.

`config/default.yaml` tek doğruluk kaynağıdır. Hiçbir eşik/flag koda gömülmez.
Belirli env değişkenleri (AI_MODE, ROADGUARD_DEVICE, port'lar) YAML'ı override eder.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"
PROFILES_DIR = ROOT / "config" / "profiles"  # default.yaml üzerine derin-merge edilen overlay'ler
SAMPLE_VIDEO = ROOT / "data" / "samples" / "ornek.mp4"  # pakete gömülü sentetik demo

log = logging.getLogger("roadguard.config")


class Config:
    """Noktalı erişimli (`cfg.get("plate.voting_buffer_size")`) ince sözlük sarmalayıcı."""

    def __init__(self, data: dict[str, Any], path: Path | None = None, profile: str | None = None):
        self._data = data
        self.path = path
        self.profile = profile  # uygulanan overlay adı (varsa) — loglama/teşhis için

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
        # Derin kopya: dışarıya verilen sözlük serbestçe değiştirilse/serialize
        # edilse bile canlı iç state'i kirletmesin (GET /config artık by-reference
        # canlı _data'yı sızdırmaz; PATCH için kasıtlı mutasyon `.data` üzerinden).
        return copy.deepcopy(self._data)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Seçili env değişkenleri YAML'ı override eder (config/README.md'de belgeli)."""
    ai_mode = os.environ.get("AI_MODE")
    if ai_mode:
        data.setdefault("runtime", {})["ai_mode"] = ai_mode
    device = os.environ.get("ROADGUARD_DEVICE")
    if device:
        data.setdefault("runtime", {})["device"] = device

    services = data.setdefault("services", {})
    for env_key, cfg_key in (
        ("ROADGUARD_INFERENCE_PORT", "inference_api"),
        ("ROADGUARD_QOD_MOCK_PORT", "qod_mock"),
        ("ROADGUARD_NV_MOCK_PORT", "nv_mock"),
    ):
        val = os.environ.get(env_key)
        if val and val.isdigit():
            services[cfg_key] = int(val)
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """`overlay`'i `base` üzerine özyinelemeli birleştir (yeni sözlük döner).

    İç içe sözlükler birleştirilir; skaler/liste değerleri overlay TAMAMEN ezer
    (ör. ``vehicle_classes`` listesi profil tarafından tümüyle değiştirilir — kısmi
    liste birleştirme sürpriz davranış yaratırdı). Mutasyonsuz: kaynaklar değişmez.
    """
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def resolve_profile_path(profile: str | Path) -> Path:
    """Profil adını/yolunu `config/profiles/<ad>.yaml`'a çöz.

    - Mevcut bir dosya yolu verilirse olduğu gibi kullanılır.
    - ``server`` gibi çıplak ad → ``config/profiles/server.yaml``.
    """
    p = Path(profile)
    if p.exists():
        return p
    candidate = PROFILES_DIR / f"{p.name}.yaml" if p.suffix != ".yaml" else PROFILES_DIR / p.name
    return candidate


def available_profiles() -> list[str]:
    """`config/profiles/` altındaki tüm profil adları (alfabetik)."""
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def load_config(path: str | Path | None = None, profile: str | None = None) -> Config:
    """Config'i yükle: `default.yaml` (taban) + opsiyonel profil overlay + env override.

    Çözüm sırası (sonraki öncekini ezer):
      1. ``path`` (verilmezse ``config/default.yaml``) — taban katman.
      2. Profil overlay — ``profile`` argümanı > ``ROADGUARD_PROFILE`` env >
         taban içindeki ``profile:`` anahtarı. ``config/profiles/<ad>.yaml`` derin-merge.
      3. Env override'lar (``AI_MODE``, ``ROADGUARD_DEVICE``, port'lar).

    Profil mekanizması geriye dönük uyumludur: profil seçilmezse davranış değişmez.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config bulunamadı: {cfg_path}. Önce `python bootstrap.py` çalıştırın."
        )
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        # Üst-düzey girinti hatası tüm dosyayı liste/skalar yapabilir → aşağıdaki
        # data.get/setdefault ham AttributeError ile çökerdi. Yönlendirici hata ver.
        raise ValueError(f"Config kökü sözlük olmalı, {type(data).__name__} bulundu: {cfg_path}")

    # Profil çözümü: açık argüman > env > taban config'teki 'profile' anahtarı.
    chosen = profile or os.environ.get("ROADGUARD_PROFILE") or data.get("profile")
    applied: str | None = None
    if chosen:
        prof_path = resolve_profile_path(chosen)
        if prof_path.exists():
            with open(prof_path, encoding="utf-8") as f:
                overlay = yaml.safe_load(f) or {}
            if not isinstance(overlay, dict):
                raise ValueError(
                    f"Profil overlay kökü sözlük olmalı, {type(overlay).__name__} bulundu: {prof_path}"
                )
            overlay.pop("profile", None)  # overlay kendini tekrar tetiklemesin
            data = _deep_merge(data, overlay)
            applied = Path(chosen).stem
            log.info("Config profili uygulandı: %s (%s)", applied, prof_path)
        else:
            log.warning(
                "Config profili bulunamadı: %s (%s). Mevcut profiller: %s",
                chosen,
                prof_path,
                ", ".join(available_profiles()) or "(yok)",
            )
    data.pop("profile", None)  # 'profile' anahtarı runtime verisi değil, meta
    data = _apply_env_overrides(data)
    return Config(data, cfg_path, profile=applied)


def resolve_repo_path(path: str | Path) -> Path:
    """Göreli yolu repo köküne göre mutlaklaştır (CWD-bağımsız model/ağırlık yükleme).

    ``weights/yolo26s.pt`` gibi config yolları, süreç hangi dizinden başlatılırsa
    başlatılsın repo köküne göre çözülür (hidden_prototip ``_resolve_model_path``
    dersi: CWD'ye bağlı yükleme, servis/CLI/IDE'den farklı davranıyordu).
    """
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


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
