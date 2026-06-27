"""Stage-1 tespit + ROI kırpma arayüzü ve dedektör fabrikası.

- `YOLO26Detector` (gerçek): ultralytics YOLO + ByteTrack (lazy: roadguard/detection/yolo.py)
- `MockDetector` (deterministik numpy): parlak araç bloklarını eşikler, IoU-takip eder
  (lazy: roadguard/detection/mock.py) → model/ağırlık olmadan tüm hat uçtan-uca çalışır
- `StubDetector`: boş çıktı (test/iskelet)

`ai_mode` çözümlemesi: real | mock | auto (ultralytics+ağırlık varsa real, yoksa mock).
Tasarım kuralı: downstream'e asla tam kare gönderilmez; yalnızca ROI crop'lar.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from roadguard.config import is_synthetic_source
from roadguard.schema import BBox

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("roadguard.detection")


@dataclass
class Detection:
    """Bir araç tespiti + (opsiyonel) takip ID'si + ROI crop'ları."""

    bbox: BBox
    track_id: int | None = None
    cabin_roi: np.ndarray | None = field(default=None, repr=False)
    plate_roi: np.ndarray | None = field(default=None, repr=False)


@dataclass
class Person:
    """Bir kişi tespiti + ByteTrack takip ID'si (sürücü kilidi için kullanılır)."""

    bbox: BBox
    track_id: int | None = None


@dataclass
class Sign:
    """Bir trafik tabelası tespiti (sahne-seviyesi; araç/kişiden ayrı toplanır).

    Ham sınıf adı (``cls``, ör. ``speed_limit_50``) taşınır; km/h çözümü SignTracker'da
    ``sign.value_map`` ile yapılır. ID-merkezli karar akışına girmez.
    """

    bbox: BBox
    cls: str = ""
    track_id: int | None = None


class Detector(ABC):
    """Tespit motoru soyut arayüzü (gerçek/mock implementasyonlar bunu uygular).

    Alt sınıflar her ``detect()`` çağrısından sonra o karede bulunan kişileri
    ``last_persons``, tabelaları ``last_signs`` listesine yazar (sürücü kilidi ve
    SignTracker bunları tüketir). Üretmeyen implementasyonlar bunları boş bırakır.
    """

    #: Son karede tespit edilen kişiler (her detect() çağrısında güncellenir)
    last_persons: list[Person]
    #: Son karede tespit edilen trafik tabelaları (her detect() çağrısında güncellenir)
    last_signs: list[Sign]
    #: Son karede tespit edilen yardımcı kanıt nesneleri (kanonik adlarıyla, ör.
    #: 'phone'/'smoking') — pipeline bunları araç kabinine düşüyorsa sürücü durumuyla
    #: füzyon eder. Üretmeyen implementasyonlar boş bırakır.
    last_aux: list[BBox]

    def __init__(self) -> None:
        # ÖNEMLİ: bu listeler ÖRNEK-seviyesi olmalı. Sınıf-seviyesi mutable varsayılan
        # (eski hata) tüm dedektör örneklerinin AYNI listeyi paylaşmasına yol açıyordu
        # (klasik mutable-default tuzağı). Alt sınıflar super().__init__() çağırmasa da
        # güvende kalmak için detect() içinde de yeniden atanır; yine de taban burada
        # temiz bir başlangıç durumu sağlar.
        self.last_persons = []
        self.last_signs = []
        self.last_aux = []

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Kareyi işle → araç tespitleri (track_id atanmış olabilir)."""
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 - opsiyonel hook (alt sınıflar override edebilir)
        """Opsiyonel kaynak temizliği (gerçek dedektörler override eder)."""


class StubDetector(Detector):
    """Tespit üretmez (iskelet/test)."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []


# --------------------------------------------------------------------------- #
# Fabrika + ai_mode çözümleme
# --------------------------------------------------------------------------- #
def _ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def resolve_ai_mode(cfg) -> str:
    """real | mock — config.runtime.ai_mode'a ('auto' dahil) göre."""
    mode = str(cfg.get("runtime.ai_mode", "auto")).lower()
    if mode == "real":
        return "real"
    if mode == "mock":
        return "mock"
    # auto: ultralytics kurulu VE detector ağırlığı mevcutsa real, aksi halde mock
    weight = Path(cfg.get("models.detector.path", "weights/yolo26s.pt"))
    if not weight.is_absolute():
        weight = Path(__file__).resolve().parents[2] / weight
    if not (_ultralytics_available() and weight.exists()):
        return "mock"
    # auto + ağırlık var: gömülü sentetik örnekte (renkli bloklar) COCO-YOLO araç
    # göremez → mock (zengin, çalışan demo). Gerçek footage/kamera → gerçek YOLO.
    if is_synthetic_source(cfg):
        return "mock"
    return "real"


def build_detector(cfg) -> Detector:
    """Config'e göre dedektör kur (ağır backend'ler lazy import edilir)."""
    mode = resolve_ai_mode(cfg)
    if mode == "real":
        from roadguard.detection.yolo import YOLO26Detector

        log.info("Detector: YOLO26 (gerçek) + %s", cfg.get("tracking.tracker", "bytetrack"))
        return YOLO26Detector(cfg)
    from roadguard.detection.mock import MockDetector

    log.info("Detector: deterministik MOCK (ağırlık yok / ai_mode=mock)")
    return MockDetector(cfg)


# --------------------------------------------------------------------------- #
# ROI geometri (modelden bağımsız, saf hesap)
# --------------------------------------------------------------------------- #
def crop_rois(
    frame: np.ndarray, bbox: BBox, cabin_ratio: float = 0.55
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Araç bbox'ından iki ROI üret: (sürücü kabini=üst, plaka bölgesi=alt).

    YOLO26l ve OCR yalnızca bu küçük crop'lar üzerinde çalışır (zero-waste prensibi).
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.x1))
    y1 = max(0, int(bbox.y1))
    x2 = min(w, int(bbox.x2))
    y2 = min(h, int(bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None, None
    split = y1 + int((y2 - y1) * cabin_ratio)
    cabin = frame[y1:split, x1:x2].copy()
    plate = frame[split:y2, x1:x2].copy()
    cabin = cabin if cabin.size else None
    plate = plate if plate.size else None
    return cabin, plate


def cap_roi_to_area(
    frame: np.ndarray,
    roi_box: tuple[int, int, int, int],
    max_area_ratio: float,
    corner: tuple[float, float] = (1.0, 1.0),
) -> tuple[int, int, int, int] | None:
    """ROI kutusunu kare alanının ``max_area_ratio`` payına KIRP (sürücü köşesine doğru).

    Devasa sürücü ROI'si (kişi-kutusu yokken geometrik kabin fallback'i: araç üst
    ~%55, ön cam + yolcu yansımaları) modelin minimum-alan ilkesine aykırıdır ve FP
    kaynağıdır. ROI alanı eşiği AŞIYORSA, En-Boy oranı korunarak hedef alana ölçeklenir
    ve ``corner`` (DriverLock ile aynı sözleşme, vars. sağ-alt = sürücü) yönüne sabitlenir.

    ``max_area_ratio <= 0`` → kapalı (None). Eşik zaten sağlanıyorsa None (kırpma yok).
    Aksi halde yeni (x1, y1, x2, y2) kutusunu döndürür. K-004: kare-alanına göreli
    oran; videoya-özel sabit yok. Saf geometri (model gerektirmez)."""
    if max_area_ratio <= 0:
        return None
    h, w = frame.shape[:2]
    frame_area = float(h * w)
    if frame_area <= 0:
        return None
    rx1, ry1, rx2, ry2 = roi_box
    rx1 = max(0, min(int(rx1), w))
    ry1 = max(0, min(int(ry1), h))
    rx2 = max(0, min(int(rx2), w))
    ry2 = max(0, min(int(ry2), h))
    rw, rh = rx2 - rx1, ry2 - ry1
    if rw <= 0 or rh <= 0:
        return None
    roi_area = float(rw * rh)
    cap = frame_area * float(max_area_ratio)
    if roi_area <= cap:
        return None  # zaten yeterince küçük → kırpma gereksiz (dar ROI değişmez)
    scale = (cap / roi_area) ** 0.5  # alan oranı → kenar oranı (kare-kök)
    new_w = max(1, int(rw * scale))
    new_h = max(1, int(rh * scale))
    cx_t, cy_t = corner  # 0..1 köşe hedefi (sağ-alt = 1,1)
    # Köşe-hizalı yerleştirme: anchor noktasını köşeye sabitle, kalan alanı içeri al.
    nx1 = int(rx1 + (rw - new_w) * cx_t)
    ny1 = int(ry1 + (rh - new_h) * cy_t)
    nx1 = max(rx1, min(nx1, rx2 - new_w))
    ny1 = max(ry1, min(ny1, ry2 - new_h))
    return nx1, ny1, nx1 + new_w, ny1 + new_h


def crop_person_roi(frame: np.ndarray, bbox: BBox, pad_ratio: float = 0.15) -> np.ndarray | None:
    """Kilitli sürücünün kutusundan ROI kes (kenarlardan `pad_ratio` kadar pay bırakır).

    Geometrik 'üst %55 kabin' tahmini yerine, sürücü olarak kilitlenmiş kişinin
    gerçek kutusundan kırpar; Stage-2 (YOLO26l driver_state) yalnızca bu crop'ta çalışır.
    """
    h, w = frame.shape[:2]
    pad_x = bbox.width * pad_ratio
    pad_y = bbox.height * pad_ratio
    x1 = max(0, int(bbox.x1 - pad_x))
    y1 = max(0, int(bbox.y1 - pad_y))
    x2 = min(w, int(bbox.x2 + pad_x))
    y2 = min(h, int(bbox.y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2].copy()
    return roi if roi.size else None
