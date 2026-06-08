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


class DriverState(BaseModel):
    phone: bool = False
    smoking: bool = False
    no_seatbelt: bool = False
    fatigue: bool = False
    confidence: dict[str, float] = Field(default_factory=dict)

    def active_flags(self) -> list[str]:
        return [k for k in ("phone", "smoking", "no_seatbelt", "fatigue") if getattr(self, k)]


class SpeedState(BaseModel):
    value_kmh: float | None = None
    mode: Literal["tripwire", "ipm", "disabled"] = "disabled"
    relative_velocity_flag: bool = False


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
    "SPEED",
    "QOD_TRIGGER",
    "QOD_RELEASE",
    "RISK_ALERT",
]


class AuraEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = Field(default_factory=time.time)
    track_id: int
    type: EventType
    payload: dict = Field(default_factory=dict)
    source: str = "aura-inference"


class AnnotationFrame(BaseModel):
    frame_id: int
    ts: float = Field(default_factory=time.time)
    tracks: list[dict] = Field(default_factory=list)  # bbox + label + track_id + risk_flags


def make_event(track_id: int, type: EventType, payload: dict | None = None,
               ts: float | None = None) -> AuraEvent:
    """AuraEvent kısa-yolu (ts verilmezse şimdi)."""
    return AuraEvent(
        track_id=track_id,
        type=type,
        payload=payload or {},
        ts=ts if ts is not None else time.time(),
    )
