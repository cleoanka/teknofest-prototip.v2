"""Stage-2 sürücü durumu sınıflandırıcı arayüzü + fabrika.

- `YOLO26lDriverClassifier` (gerçek): cabin ROI üzerinde çoklu-etiket detection
  (phone/smoking/no_seatbelt/fatigue). MediaPipe/landmark KESİNLİKLE kullanılmaz —
  yorgunluk dahil tüm durumlar detection sınıfı olarak öğrenilir.
- `MockDriverClassifier` (deterministik): cabin ROI baskın rengini senaryo sürücü
  durumuna eşler → ağırlık olmadan anlamlı sürücü-durum event'leri üretilir.

Girdi yalnızca sürücü kabini ROI'sidir (asla tam kare).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from aura.config import is_synthetic_source
from aura.schema import DriverState

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("aura.driver_state")


class DriverClassifier(ABC):
    @abstractmethod
    def infer(self, cabin_roi: np.ndarray | None) -> DriverState:
        raise NotImplementedError


def _ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def resolve_driver_mode(cfg) -> str:
    mode = str(cfg.get("runtime.ai_mode", "auto")).lower()
    if mode in ("real", "mock"):
        return mode
    weight = Path(cfg.get("models.driver_state.path", "weights/yolo26l.pt"))
    if not weight.is_absolute():
        weight = Path(__file__).resolve().parents[2] / weight
    if not (_ultralytics_available() and weight.exists()):
        return "mock"
    # auto + ağırlık var: sentetik örnekte gerçek YOLO26l anlamlı sürücü-durumu
    # üretmez → mock (senaryo-bazlı zengin demo). Gerçek footage → gerçek model.
    return "mock" if is_synthetic_source(cfg) else "real"


def build_driver_classifier(cfg) -> DriverClassifier:
    if resolve_driver_mode(cfg) == "real":
        from aura.driver_state.yolo import YOLO26lDriverClassifier

        log.info("DriverState: YOLO26l (gerçek)")
        return YOLO26lDriverClassifier(cfg)
    from aura.driver_state.mock import MockDriverClassifier

    log.info("DriverState: deterministik MOCK")
    return MockDriverClassifier(cfg)
