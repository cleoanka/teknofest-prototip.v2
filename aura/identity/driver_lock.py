"""Sürücü kimlik kilidi (Driver Lock).

Kural (kullanıcı talebi):
  1. Araç kabininde **sağ-alttaki ilk kişiyi** sürücü adayı kabul et.
  2. Aynı kişi (ByteTrack ID'si) **confirm_frames (vars. 5) ardışık karede** sürücü
     adayı kalırsa, o kişinin ID'sini **araca kilitle**.
  3. Kilit sonrası **başka hiç kimse** o aracın sürücüsü olamaz (aday değişse bile).
  4. **Global dışlama:** Bir kişi yalnızca **GERÇEKTEN içinde olduğu TEK araca**
     kilitlenir. Örtüşen/yan yana araçlarda aynı kişi iki aracın sürücüsü olamaz;
     kare başına her kişi en iyi (en çok örtüşen + merkeze en yakın) araca atanır.

Tasarım: ID-merkezli (kare-merkezli değil). Kişiler Stage-1'de YOLO+ByteTrack ile
tüm karede tespit edilip takip edilir; bu modül kişileri araç kutusuna eşler, sağ-alt
adayı seçer, tutarlılığı sayar ve kilidi yönetir. Modelden bağımsız, saf hesap.

Tek-araç (`update`) ve global (`assign_frame`) olmak üzere iki giriş noktası vardır;
pipeline her kare bir kez `assign_frame` çağırır (araçlar arası çift-sahiplenmeyi
ancak tüm araçları aynı anda görerek engelleyebiliriz). `update` geriye dönük
uyumluluk için (tek araç) korunur.

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
        # Araç atamasında merkeze-yakınlığın ağırlığı: örtüşme (containment) eşit
        # olduğunda (kişi iki kutuya da tam giriyorsa) merkezi daha yakın aracı seçer.
        self.center_bias = float(cfg.get("driver_lock.center_bias", 0.25))

        # vehicle_id -> kilitli sürücü person_id
        self._locked: dict[int, int] = {}
        # person_id -> vehicle_id: bir sürücü yalnızca TEK araca sahip olabilir (global dışlama)
        self._driver_vehicle: dict[int, int] = {}
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
            if p.track_id is not None and _containment(p.bbox, vehicle) >= self.min_containment
        ]

    def _bbox_of(self, person_id: int, persons: list[Person]) -> BBox | None:
        """Verilen takip ID'sine sahip kişinin bu kareki kutusu (yoksa None)."""
        return next((p.bbox for p in persons if p.track_id == person_id), None)

    def _owned_by_other(self, person_id: int | None, vehicle_id: int) -> bool:
        """Bu kişi BAŞKA bir araca kilitli mi? (kendi aracına kilitliyse engel değil)"""
        owner = self._driver_vehicle.get(person_id)
        return owner is not None and owner != vehicle_id

    def _pick_corner(self, vehicle: BBox, pool: list[Person]) -> Person | None:
        """Önceden filtrelenmiş havuzdan hedef köşeye (vars. sağ-alt) en yakın kişiyi seç."""
        if not pool:
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
        return max(pool, key=lambda p: (corner_score(p), -p.track_id))

    def select_candidate(self, vehicle: BBox, persons: list[Person]) -> Person | None:
        """Araç içindeki kişiler arasından hedef köşeye (vars. sağ-alt) en yakın olanı seç."""
        return self._pick_corner(vehicle, self.persons_in_vehicle(vehicle, persons))

    def _assign_score(self, person: BBox, vehicle: BBox) -> float | None:
        """Kişi↔araç eşleşme skoru (büyük = daha iyi); eşik geçmiyorsa None.

        Örtüşme (containment) birincil ölçüttür; kişi iki araca da tam giriyorsa
        (örtüşen kutular) örtüşme ~1.0'da doygunlaşır, bu yüzden merkeze-yakınlık ile
        ayrıştırırız: kişi GERÇEKTEN içinde olduğu aracın merkezine daha yakındır.
        """
        c = _containment(person, vehicle)
        if c < self.min_containment:
            return None
        vw, vh = max(vehicle.width, 1e-6), max(vehicle.height, 1e-6)
        pcx, pcy = person.center
        vcx, vcy = vehicle.center
        # araç boyutuna göre normalize merkez uzaklığı (0 = merkez, dışa doğru büyür)
        dn = (((pcx - vcx) / vw) ** 2 + ((pcy - vcy) / vh) ** 2) ** 0.5
        return c - self.center_bias * dn

    # --- kilit çekirdeği (dışlamalı havuz üzerinde tek araç) --------------- #
    def _lock_step(
        self,
        vehicle_id: int,
        vehicle: BBox,
        pool: list[Person],
        persons: list[Person],
    ) -> DriverAssignment:
        """Bu araç için adayı seç, tutarlılığı say ve gerekiyorsa kilitle.

        `pool`  → bu araca ÖZEL (dışlamalı) aday kişiler (başka araca kilitli olanlar yok).
        `persons` → tüm kişiler; yalnızca kilitli sürücünün kutusunu bulmak için kullanılır.
        """
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

        # 2) Kilit yok: havuzdan sağ-alt adayı seç.
        cand = self._pick_corner(vehicle, pool)
        if cand is None:
            # Bu karede araca atanmış kişi yok → tutarlılık sıfırlanır.
            self._cand.pop(vehicle_id, None)
            return DriverAssignment(vehicle_id=vehicle_id)

        cand_id = cand.track_id
        prev = self._cand.get(vehicle_id)
        if prev is not None and prev[0] == cand_id:
            prev[1] += 1
        else:
            self._cand[vehicle_id] = [cand_id, 1]
        streak = self._cand[vehicle_id][1]

        # 3) confirm_frames sağlandıysa KİLİTLE (ve kişiyi bu araca sahiplen).
        if streak >= self.confirm_frames:
            self._locked[vehicle_id] = cand_id
            self._driver_vehicle[cand_id] = vehicle_id  # kişi → araç sahipliği (global dışlama)
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

    # --- giriş noktası: tek araç (geriye dönük uyumluluk) ------------------ #
    def update(
        self, vehicle_id: int, vehicle: BBox, persons: list[Person], frame_idx: int
    ) -> DriverAssignment:
        """Tek bir araç için kilidi güncelle ve güncel atamayı döndür.

        Aday havuzu: araç kabinindeki kişiler eksi BAŞKA araca kilitli olanlar.
        (Çok-araçlı kareler için `assign_frame` tercih edilir; o, çift-sahiplenmeyi
        araçları aynı anda görerek baştan engeller.)
        """
        self._last_seen[vehicle_id] = frame_idx
        if not self.enabled:
            return DriverAssignment(vehicle_id=vehicle_id)
        pool = [
            p
            for p in self.persons_in_vehicle(vehicle, persons)
            if not self._owned_by_other(p.track_id, vehicle_id)
        ]
        return self._lock_step(vehicle_id, vehicle, pool, persons)

    # --- giriş noktası: global / dışlamalı (pipeline bunu kullanır) -------- #
    def assign_frame(
        self, vehicles: list[tuple[int, BBox]], persons: list[Person], frame_idx: int
    ) -> list[DriverAssignment]:
        """Kare içindeki TÜM araç↔sürücü eşleşmesini global ve dışlamalı çöz.

        `vehicles` : (vehicle_id, araç_bbox) listesi — pipeline'daki tespit sırası.
        Dönüş      : aynı sıradaki `DriverAssignment` listesi (vehicles[i] ↔ out[i]).

        Çift-sahiplenme engeli:
          • Zaten kilitli sürücüler yalnızca kendi araçlarına rezerve edilir; başka
            hiçbir aracın aday havuzuna girmezler.
          • Serbest (kilitsiz) her kişi, eşiği geçen araçlar arasında TEK bir araca
            (en iyi skor: örtüşme + merkez yakınlığı) atanır — iki aracın havuzunda
            birden bulunamaz, dolayısıyla aynı karede iki araca kilitlenemez.
        """
        for vid, _ in vehicles:
            self._last_seen[vid] = frame_idx
        if not self.enabled:
            return [DriverAssignment(vehicle_id=vid) for vid, _ in vehicles]

        valid = [p for p in persons if p.track_id is not None]
        # Pozisyona göre havuzlar (vehicle_id çakışabilir, ör. takipsiz araçlar -1).
        pools: list[list[Person]] = [[] for _ in vehicles]

        for p in valid:
            owner = self._driver_vehicle.get(p.track_id)
            if owner is not None:
                # Kilitli sürücü: SADECE sahibi olan aracın havuzuna (başka araçlardan dışlanır).
                for i, (vid, _) in enumerate(vehicles):
                    if vid == owner:
                        pools[i].append(p)
                continue
            # Serbest kişi: eşiği geçen araçlar arasından en iyi skorlu TEK aracı seç.
            best_i: int | None = None
            best_s: float | None = None
            for i, (vid, vbbox) in enumerate(vehicles):
                s = self._assign_score(p.bbox, vbbox)
                if s is None:
                    continue
                if best_s is None or s > best_s:
                    best_s, best_i = s, i
            if best_i is not None:
                pools[best_i].append(p)

        return [
            self._lock_step(vid, vbbox, pools[i], persons)
            for i, (vid, vbbox) in enumerate(vehicles)
        ]

    # --- sorgu / bakım ----------------------------------------------------- #
    def driver_of(self, vehicle_id: int) -> int | None:
        """Araca kilitli sürücü ID'si (yoksa None)."""
        return self._locked.get(vehicle_id)

    def is_locked(self, vehicle_id: int) -> bool:
        return vehicle_id in self._locked

    def prune(self, frame_idx: int) -> None:
        """max_age'den uzun süredir görünmeyen araçların kilidini/adayını/sahipliğini unut."""
        dead = [vid for vid, seen in self._last_seen.items() if frame_idx - seen > self.max_age]
        for vid in dead:
            drv = self._locked.pop(vid, None)
            # Bu araca ait sürücü sahipliğini serbest bırak (kişi yeniden atanabilsin).
            if drv is not None and self._driver_vehicle.get(drv) == vid:
                self._driver_vehicle.pop(drv, None)
            self._cand.pop(vid, None)
            self._last_seen.pop(vid, None)
