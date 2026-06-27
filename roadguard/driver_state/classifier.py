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

from roadguard.config import is_synthetic_source
from roadguard.schema import DriverState

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("roadguard.driver_state")


class DriverClassifier(ABC):
    @abstractmethod
    def infer(self, cabin_roi: np.ndarray | None, track_id: int | None = None) -> DriverState:
        """ROI'den sürücü durumu üret.

        ``track_id`` opsiyoneldir: durum tutan backend'ler (ör. pose hibritinin
        telefon-nesnesi latch'i) kısa süreli kanıt belleğini track'e bağlamak için
        kullanır; durumsuz backend'ler yok sayar.
        """
        raise NotImplementedError


def _ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def _weight_exists(cfg, key: str, default: str) -> bool:
    weight = Path(cfg.get(key, default))
    if not weight.is_absolute():
        weight = Path(__file__).resolve().parents[2] / weight
    return weight.exists()


def _pose_weight_exists(cfg) -> bool:
    """Yapılandırılan pose ağırlığı VEYA stok s-pose fallback'i diskte mi?

    PoseDriverClassifier yapılandırılan ağırlık yoksa s-pose'a loglu düşer;
    backend seçimi de aynı gerçeği görmeli (l-pose inmemiş diye yolo'ya düşmesin).
    """
    return (
        _weight_exists(cfg, "models.driver_state.pose_path", "weights/yolo26l-pose.pt")
        or (Path(__file__).resolve().parents[2] / "weights/yolo26s-pose.pt").exists()
    )


def resolve_driver_mode(cfg) -> str:
    mode = str(cfg.get("runtime.ai_mode", "auto")).lower()
    if mode in ("real", "mock"):
        return mode
    has_yolo = _weight_exists(cfg, "models.driver_state.path", "weights/yolo26l.pt")
    has_pose = _pose_weight_exists(cfg)
    if not (_ultralytics_available() and (has_yolo or has_pose)):
        return "mock"
    # auto + ağırlık var: sentetik örnekte gerçek YOLO26l anlamlı sürücü-durumu
    # üretmez → mock (senaryo-bazlı zengin demo). Gerçek footage → gerçek model.
    return "mock" if is_synthetic_source(cfg) else "real"


def resolve_driver_backend(cfg) -> str:
    """real moddaki backend: ``pose`` | ``yolo`` (config: models.driver_state.backend).

    - ``yolo``: fine-tune edilmiş detection ağırlığı (phone/smoking/... sınıfları)
      varsa en doğru yol. STOK COCO yolo26l ile bu sınıflar ÜRETİLEMEZ.
    - ``pose``: YOLO26-pose keypoint geometrisi — fine-tune ağırlık gerektirmez
      (v1'in MediaPipe geometrisinin saf-YOLO portu). Bkz. driver_state/pose.py.
    - ``auto`` (varsayılan): pose ağırlığı diskte varsa pose, yoksa yolo.
    """
    backend = str(cfg.get("models.driver_state.backend", "auto")).lower()
    if backend in ("pose", "yolo"):
        return backend
    return "pose" if _pose_weight_exists(cfg) else "yolo"


def build_driver_classifier(cfg) -> DriverClassifier:
    if resolve_driver_mode(cfg) == "real":
        if resolve_driver_backend(cfg) == "pose":
            from roadguard.driver_state.pose import PoseDriverClassifier

            log.info("DriverState: YOLO26-pose keypoint geometrisi (gerçek)")
            return PoseDriverClassifier(cfg)
        from roadguard.driver_state.yolo import YOLO26lDriverClassifier

        log.info("DriverState: YOLO26l detection (gerçek)")
        return YOLO26lDriverClassifier(cfg)
    from roadguard.driver_state.mock import MockDriverClassifier

    log.info("DriverState: deterministik MOCK")
    return MockDriverClassifier(cfg)
