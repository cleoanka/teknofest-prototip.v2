"""Opsiyonel modül yükleyici — lazy loading pattern.

`config.optional_modules.<name>` false iken modül **import bile edilmez**.
Sadece flag true olduğunda `importlib` ile yüklenir ve cache'lenir.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

log = logging.getLogger("roadguard.optional")

_cache: dict[str, ModuleType] = {}


def is_enabled(cfg, name: str) -> bool:
    return bool(cfg.get(f"optional_modules.{name}", False))


def get_optional(cfg, name: str) -> ModuleType | None:
    """Modülü döndür (flag açıksa); kapalıysa None — import YAPILMAZ."""
    if not is_enabled(cfg, name):
        return None
    mod = _cache.get(name)
    if mod is None:
        mod = importlib.import_module(f"roadguard.optional.{name}")
        _cache[name] = mod
        log.info("Opsiyonel modül yüklendi: %s", name)
    return mod


def _reset_cache() -> None:  # test yardımcısı
    _cache.clear()
