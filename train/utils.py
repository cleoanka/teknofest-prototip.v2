"""Eğitim yardımcıları — cihaz çözümleme, custom ağırlık swap."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger("aura.train")

ROOT = Path(__file__).resolve().parent.parent


def resolve_device(device: str | None):
    """'auto' → ultralytics otomatik seçsin (None); aksi halde değer."""
    return None if device in (None, "auto", "") else device


def export_best(results, dest_name: str) -> Path | None:
    """Eğitim çıktısındaki best.pt'yi weights/ altına kopyala (config swap için)."""
    best = None
    try:
        best = Path(getattr(results, "save_dir", "")) / "weights" / "best.pt"
    except Exception:  # noqa: BLE001
        pass
    if not best or not best.exists():
        log.warning("best.pt bulunamadı (eğitim çıktısı: %s)", getattr(results, "save_dir", "?"))
        return None
    dest = ROOT / "weights" / dest_name
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    log.info("Custom ağırlık kaydedildi: %s", dest)
    log.info("config/default.yaml → models.*.path değerini bununla güncelleyin.")
    return dest
