"""Sürücü kimlik kilidi (Driver Lock).

Kural (kullanıcı talebi):
  1. Araç kabininde **sağ-alttaki ilk kişiyi** sürücü adayı kabul et.
  2. Aynı kişi (ByteTrack ID'si) **confirm_frames (vars. 5) ardışık karede** sürücü
     adayı kalırsa, o kişinin ID'sini **araca kilitle**.
  3. Kilit sonrası **başka hiç kimse** o aracın sürücüsü olamaz (aday değişse bile).

Tasarım: ID-merkezli (kare-merkezli değil). Kişiler Stage-1'de YOLO+ByteTrack ile
tüm karede tespit edilip takip edilir; bu modül kişileri araç kutusuna eşler, sağ-alt
adayı seçer, tutarlılığı sayar ve kilidi yönetir. Modelden bağımsız, saf hesap.

"sağ-alt" yönü config ile değiştirilebilir (`driver_lock.corner`): Türkiye soldan
direksiyondur; kamera açısına göre sürücü görüntüde farklı köşeye düşebilir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aura.detection.detector import Person
from aura.schema import BBox

log = logging.getLogger("aura.identity.driver_lock")

# corner adı → (hedef_nx, hedef_ny) normalize araç-içi köşe (0..1)
_CORNERS = {
    "bottom_right": (1.0, 1.0),
    "bottom_left": (0.0, 1.0),
    "top_right": (1.0, 0.0),
    "top_left": (0.0, 0.0),
}


@dataclass
class DriverAssignment:
    """Bir araç için sürücü kilidi anlık durumu."""

    vehicle_id: int
    driver_id: int | None = None  # kilitliyse kilitli kişi; değilse mevcut aday
    locked: bool = False
    candidate_id: int | None = None  # bu karedeki sağ-alt aday
    streak: int = 0  # adayın üst üste kaç karedir tutulduğu
    newly_locked: bool = False  # bu karede yeni mi kilitlendi (event tetikler)
    driver_bbox: BBox | None = None  # kilitli/aday sürücünün BU karedeki kutusu (ROI için)


def _containment(person: BBox, vehicle: BBox) -> float:
    """Kişi kutusunun araç kutusuyla örtüşme oranı (kişi alanına göre, 0..1)."""
    ix1, iy1 = max(person.x1, vehicle.x1), max(person.y1, vehicle.y1)
    ix2, iy2 = min(person.x2, vehicle.x2), min(person.y2, vehicle.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    parea = max(person.width * person.height, 1e-6)
    return inter / parea


class DriverLock:
    """Araç başına sürücü adayı seçimi, tutarlılık sayımı ve kalıcı kilit."""

    def __init__(self, cfg):
        self.enabled = bool(cfg.get("driver_lock.enabled", True))
        self.confirm_frames = max(1, int(cfg.get("driver_lock.confirm_frames", 5)))
        corner = str(cfg.get("driver_lock.corner", "bottom_right")).lower()
        self.target = _CORNERS.get(corner, _CORNERS["bottom_right"])
        self.min_containment = float(cfg.get("driver_lock.min_containment", 0.5))
        self.max_age = int(cfg.get("driver_lock.max_age", 30))

        # vehicle_id -> kilitli sürücü person_id
        self._locked: dict[int, int] = {}
        # vehicle_id -> [aday_person_id, streak]
        self._cand: dict[int, list[int]] = {}
        # vehicle_id -> en son görüldüğü kare (prune için)
        self._last_seen: dict[int, int] = {}

    # --- yardımcılar ------------------------------------------------------- #
    def persons_in_vehicle(self, vehicle: BBox, persons: list[Person]) -> list[Person]:
        """Kutusu araç kutusuyla yeterince örtüşen (kabindeki) kişiler."""
        return [
            p
            for p in persons
            if p.track_id is not None
            and _containment(p.bbox, vehicle) >= self.min_containment
        ]

    def _bbox_of(self, person_id: int, persons: list[Person]) -> BBox | None:
        """Verilen takip ID'sine sahip kişinin bu kareki kutusu (yoksa None)."""
        return next((p.bbox for p in persons if p.track_id == person_id), None)

    def select_candidate(self, vehicle: BBox, persons: list[Person]) -> Person | None:
        """Araç içindeki kişiler arasından hedef köşeye (vars. sağ-alt) en yakın olanı seç."""
        cand = self.persons_in_vehicle(vehicle, persons)
        if not cand:
            return None
        tx, ty = self.target
        vw, vh = max(vehicle.width, 1e-6), max(vehicle.height, 1e-6)

        def corner_score(p: Person) -> float:
            cx, cy = p.bbox.center
            nx = (cx - vehicle.x1) / vw  # 0..1 (araç içinde)
            ny = (cy - vehicle.y1) / vh
            # hedef köşeye uzaklık küçükse skor büyük → max alınır
            return -((nx - tx) ** 2 + (ny - ty) ** 2)

        # eşitlikte deterministik: önce skor, sonra küçük track_id
        return max(cand, key=lambda p: (corner_score(p), -p.track_id))

    # --- ana giriş noktası ------------------------------------------------- #
    def update(
        self, vehicle_id: int, vehicle: BBox, persons: list[Person], frame_idx: int
    ) -> DriverAssignment:
        """Bir araç için kilidi güncelle ve güncel atamayı döndür."""
        self._last_seen[vehicle_id] = frame_idx

        if not self.enabled:
            return DriverAssignment(vehicle_id=vehicle_id)

        # 1) Zaten kilitliyse: kilitli sürücüyü döndür, başka adayı YOK SAY.
        #    ROI için kilitli kişinin bu kareki kutusunu da getir (kaybolduysa None).
        if vehicle_id in self._locked:
            locked_id = self._locked[vehicle_id]
            return DriverAssignment(
                vehicle_id=vehicle_id,
                driver_id=locked_id,
                locked=True,
                candidate_id=locked_id,
                streak=self.confirm_frames,
                driver_bbox=self._bbox_of(locked_id, persons),
            )

        # 2) Kilit yok: sağ-alt adayı seç.
        cand = self.select_candidate(vehicle, persons)
        if cand is None:
            # Bu karede kabinde kişi yok → tutarlılık sıfırlanır.
            self._cand.pop(vehicle_id, None)
            return DriverAssignment(vehicle_id=vehicle_id)

        cand_id = cand.track_id
        prev = self._cand.get(vehicle_id)
        if prev is not None and prev[0] == cand_id:
            prev[1] += 1
        else:
            self._cand[vehicle_id] = [cand_id, 1]
        streak = self._cand[vehicle_id][1]

        # 3) confirm_frames sağlandıysa KİLİTLE.
        if streak >= self.confirm_frames:
            self._locked[vehicle_id] = cand_id
            self._cand.pop(vehicle_id, None)
            log.info(
                "Sürücü kilitlendi: araç=%s sürücü=%s (%d kare)",
                vehicle_id,
                cand_id,
                self.confirm_frames,
            )
            return DriverAssignment(
                vehicle_id=vehicle_id,
                driver_id=cand_id,
                locked=True,
                candidate_id=cand_id,
                streak=streak,
                newly_locked=True,
                driver_bbox=cand.bbox,
            )

        # Henüz kilit yok → güncel aday (geçici). ROI aday kutusundan kesilebilir.
        return DriverAssignment(
            vehicle_id=vehicle_id,
            driver_id=cand_id,
            locked=False,
            candidate_id=cand_id,
            streak=streak,
            driver_bbox=cand.bbox,
        )

    # --- sorgu / bakım ----------------------------------------------------- #
    def driver_of(self, vehicle_id: int) -> int | None:
        """Araca kilitli sürücü ID'si (yoksa None)."""
        return self._locked.get(vehicle_id)

    def is_locked(self, vehicle_id: int) -> bool:
        return vehicle_id in self._locked

    def prune(self, frame_idx: int) -> None:
        """max_age'den uzun süredir görünmeyen araçların kilidini/adayını unut."""
        dead = [
            vid
            for vid, seen in self._last_seen.items()
            if frame_idx - seen > self.max_age
        ]
        for vid in dead:
            self._locked.pop(vid, None)
            self._cand.pop(vid, None)
            self._last_seen.pop(vid, None)
