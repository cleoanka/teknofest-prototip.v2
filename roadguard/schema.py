"""Çekirdek veri sözleşmeleri (pydantic v2) — plan.md §6.0.

Downstream'in tamamı (accumulator, events, API, dashboard, eval) yalnızca bu
sözleşmeleri bilir. Bu modül değişmeden hiçbir katman sözleşme dışı veri beklemez.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Alt durum modelleri
# --------------------------------------------------------------------------- #


class PlateState(BaseModel):
    value: str | None = None
    confidence: float = 0.0
    status: Literal["pending", "confirmed", "rejected"] = "pending"
    votes: dict[str, int] = Field(default_factory=dict)
    ocr_disabled: bool = False  # erken çıkış flag'i (konsensüs sonrası OCR kapanır)
    # Kısmi plaka: tam TR formatı doğrulanamasa bile en güçlü aday burada raporlanır
    # (jüri/denetim kanıtı — "kanıtlanamayan hedef puanlanmaz" şartname kuralı için).
    partial: str | None = None


class DriverState(BaseModel):
    phone: bool = False
    smoking: bool = False
    no_seatbelt: bool = False
    fatigue: bool = False
    # HAM gözlem (ihlal DEĞİL): kemer şeridi görüldü mü. Model bunu tespit eder;
    # `no_seatbelt` ihlali Katman B'de bunun YOKLUĞUNDAN türetilir (DriverStateEngine).
    seatbelt: bool = False
    confidence: dict[str, float] = Field(default_factory=dict)

    def active_flags(self) -> list[str]:
        # Yalnızca İHLAL bayrakları (seatbelt presence bir ihlal değildir → dahil değil).
        return [k for k in ("phone", "smoking", "no_seatbelt", "fatigue") if getattr(self, k)]


class SpeedState(BaseModel):
    value_kmh: float | None = None
    mode: Literal["tripwire", "ipm", "disabled", "metric"] = "disabled"
    relative_velocity_flag: bool = False
    is_calibrated: bool = False  # metric mod: ppm(y) ölçek-alanı hazır mı (km/h gerçek mi)
    # Swerving (dikkatsiz sürüş): yanal yörüngede zigzag VEYA hızlı yanal kayma.
    # Ölçek-bağımsız ölçülür (araç genişliği birimi) — kalibrasyon gerektirmez.
    swerving: bool = False


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 0.0
    cls: str = ""

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


# --------------------------------------------------------------------------- #
# Sahne-seviyesi modeller (ID-merkezli DEĞİL — tüm kareye/sahneye ait bağlam)
# --------------------------------------------------------------------------- #


class SignDetection(BaseModel):
    """Bir trafik tabelası tespiti. Bir araca bağlı değildir (sahne-seviyesi).

    ``speed_limit_kmh`` yalnızca hız-limiti tabelalarında doludur; diğer tabelalarda
    (dur, yol ver vb.) None'dır ve tabela yalnızca çizim/kayıt amacıyla taşınır.
    """

    cls: str = ""  # ham sınıf adı (ör. "speed_limit_50", "sign")
    bbox: BBox
    speed_limit_kmh: int | None = None


class SceneContext(BaseModel):
    """Kare/sahne-seviyesi bağlam — ID-merkezli accumulator'ın YANINDA taşınır.

    Şu an yalnızca aktif hız limitini tutar (tabela geçildikten sonra da bir süre
    geçerli kalır; bkz. SignTracker). İleride hava/ışık/şerit-zon bilgisi eklenebilir.
    """

    active_speed_limit_kmh: int | None = None
    speed_limit_source_cls: str | None = None
    sign_count: int = 0


# --------------------------------------------------------------------------- #
# ID-merkezli kayıt
# --------------------------------------------------------------------------- #


class TrackRecord(BaseModel):
    track_id: int
    vehicle_class: str = ""
    first_frame: int = 0
    last_frame: int = 0
    bbox: BBox
    plate: PlateState = Field(default_factory=PlateState)
    driver: DriverState = Field(default_factory=DriverState)
    # Sürücü kimlik kilidi: araca eşlenen kişinin takip ID'si ve kilit durumu.
    # driver_locked=True olunca bu ID araca kalıcı bağlanır; başka kişi sürücü olamaz.
    driver_id: int | None = None
    driver_locked: bool = False
    speed: SpeedState = Field(default_factory=SpeedState)
    qod_active: bool = False
    qod_profile: str | None = None
    risk_flags: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Event ve annotation stream sözleşmeleri
# --------------------------------------------------------------------------- #

EventType = Literal[
    "DETECTION_UPDATE",
    "PLATE_CONFIRMED",
    "PLATE_REJECTED",
    "DRIVER_STATE",
    "DRIVER_LOCKED",
    "SPEED",
    "QOD_TRIGGER",
    "QOD_RELEASE",
    "RISK_ALERT",
    "SPEED_LIMIT_DETECTED",  # sahne: aktif hız limiti değişti (track_id=-1)
    "SPEED_LIMIT_VIOLATION",  # araç: hız aktif tabela limitini aştı
]


class RoadGuardEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = Field(default_factory=time.time)
    track_id: int
    type: EventType
    payload: dict = Field(default_factory=dict)
    source: str = "roadguard-inference"


class AnnotationFrame(BaseModel):
    frame_id: int
    ts: float = Field(default_factory=time.time)
    tracks: list[dict] = Field(default_factory=list)  # bbox + label + track_id + risk_flags
    persons: list[dict] = Field(
        default_factory=list
    )  # kişiler: bbox + role ("driver"/"passenger") + track_id + vehicle_id + locked
    signs: list[dict] = Field(
        default_factory=list
    )  # sahne tabelaları: bbox + cls + speed_limit_kmh
    scene: dict = Field(default_factory=dict)  # SceneContext.model_dump() (aktif hız limiti vb.)


def make_event(
    track_id: int, type: EventType, payload: dict | None = None, ts: float | None = None
) -> RoadGuardEvent:
    """RoadGuardEvent kısa-yolu (ts verilmezse şimdi)."""
    return RoadGuardEvent(
        track_id=track_id,
        type=type,
        payload=payload or {},
        ts=ts if ts is not None else time.time(),
    )
