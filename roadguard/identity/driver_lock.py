"""Sürücü/Yolcu atama (Driver Lock).

Kural (kullanıcı talebi — gözden geçirilmiş mekanik):
  1. Araç kabininde **en alttaki — berabere kalırsa en sağdaki — görünen kişi
     HER ZAMAN sürücüdür** (DİKEY öncelikli köşe kuralı). Sürücü kimliğe KİLİTLENMEZ;
     her kare konuma göre yeniden seçilir → track-ID titrese de görünen sürücü her
     zaman 'sürücü' etiketlenir (eski 'sürücüyü kilitle' mekaniğinin track kaybında
     gerçek sürücüyü 'yolcu' göstermesi böyle çözülür).
  2. Sürücü dışındaki herkes **yolcudur**. Bir kişi **confirm_frames (vars. 3)
     ardışık karede yolcu** kalırsa **YOLCU olarak kilitlenir**: artık o araçta
     sürücü adayı olamaz (anlık konum gürültüsüyle sürücü etiketini çalamaz).
  3. **Global dışlama:** kilitli bir yolcu yalnızca **TEK araca** (kilitlendiği)
     aittir; örtüşen araçlarda iki aracın havuzuna birden giremez. Serbest (kilitsiz)
     kişiler kare başına en iyi (örtüşme + merkez yakınlığı) araca atanır.

Tasarım: ID-merkezli değil KONUM-merkezli sürücü + ID-merkezli yolcu kilidi. Kişiler
Stage-1'de YOLO+ByteTrack ile tüm karede tespit edilip takip edilir; bu modül kişileri
araç kutusuna eşler, sağ-en-alt sürücüyü seçer ve yolcuları kilitler. Saf hesap.

Tek-araç (`update`) ve global (`assign_frame`) iki giriş noktası vardır; pipeline her
kare bir kez `assign_frame` çağırır. `update` geriye dönük uyumluluk için korunur.

"sağ-alt" yönü config ile değiştirilebilir (`driver_lock.corner`): Türkiye soldan
direksiyondur; kamera açısına göre sürücü görüntüde farklı köşeye düşebilir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from roadguard.detection.detector import Person
from roadguard.schema import BBox

log = logging.getLogger("roadguard.identity.driver_lock")

# corner adı → (hedef_nx, hedef_ny) normalize araç-içi köşe (0..1)
_CORNERS = {
    "bottom_right": (1.0, 1.0),
    "bottom_left": (0.0, 1.0),
    "top_right": (1.0, 0.0),
    "top_left": (0.0, 0.0),
}


@dataclass
class DriverAssignment:
    """Bir araç için sürücü/yolcu atamasının anlık durumu."""

    vehicle_id: int
    driver_id: int | None = None  # bu kareki sağ-en-alt sürücü (pozisyonel; kilitli DEĞİL)
    locked: bool = False  # sürücü KURULDU mu (≥confirm_frames ardışık sürücü-varlığı; yapışkan)
    candidate_id: int | None = None  # = driver_id (geriye dönük uyum)
    streak: int = 0  # sürücü-varlık streak'i (üst üste kaç karedir bir sürücü var)
    newly_locked: bool = False  # sürücü bu karede ilk kez mi kuruldu (DRIVER_LOCKED event)
    driver_bbox: BBox | None = None  # bu kareki sürücünün kutusu (Stage-2 ROI için)
    passenger_ids: list[int] = field(
        default_factory=list
    )  # sürücü-dışı herkes (kilitli+aday yolcu)
    locked_passenger_ids: list[int] = field(
        default_factory=list
    )  # YOLCU olarak kilitlenmiş olanlar


def _containment(person: BBox, vehicle: BBox) -> float:
    """Kişi kutusunun araç kutusuyla örtüşme oranı (kişi alanına göre, 0..1)."""
    ix1, iy1 = max(person.x1, vehicle.x1), max(person.y1, vehicle.y1)
    ix2, iy2 = min(person.x2, vehicle.x2), min(person.y2, vehicle.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    parea = max(person.width * person.height, 1e-6)
    return inter / parea


class DriverLock:
    """Araç başına KONUM-bazlı sürücü seçimi + ardışık-kare YOLCU kilidi."""

    def __init__(self, cfg):
        self.enabled = bool(cfg.get("driver_lock.enabled", True))
        self.confirm_frames = max(1, int(cfg.get("driver_lock.confirm_frames", 3)))
        corner = str(cfg.get("driver_lock.corner", "bottom_right")).lower()
        self.target = _CORNERS.get(corner, _CORNERS["bottom_right"])
        self.min_containment = float(cfg.get("driver_lock.min_containment", 0.5))
        self.max_age = int(cfg.get("driver_lock.max_age", 30))
        # Araç atamasında merkeze-yakınlığın ağırlığı: örtüşme (containment) eşit
        # olduğunda (kişi iki kutuya da tam giriyorsa) merkezi daha yakın aracı seçer.
        self.center_bias = float(cfg.get("driver_lock.center_bias", 0.25))

        # vehicle_id -> YOLCU olarak kilitlenmiş person_id'ler (sürücü adayı olamazlar)
        self._passenger_locked: dict[int, set[int]] = {}
        # vehicle_id -> {person_id: ardışık yolcu kare sayısı}
        self._passenger_streak: dict[int, dict[int, int]] = {}
        # person_id -> vehicle_id: kilitli yolcu yalnızca TEK araca aittir (global dışlama)
        self._passenger_owner: dict[int, int] = {}
        # vehicle_id -> ardışık 'sürücü var' kare sayısı (sürücü-kuruldu event'i için)
        self._driver_streak: dict[int, int] = {}
        # sürücüsü kurulmuş (DRIVER_LOCKED yayınlanmış) araçlar — yapışkan
        self._established: set[int] = set()
        # vehicle_id -> en son atanan sürücü person_id (driver_of sorgusu için)
        self._last_driver: dict[int, int] = {}
        # vehicle_id -> en son görüldüğü kare (prune için)
        self._last_seen: dict[int, int] = {}
        # person_id -> yolcu-havuzunda en son görüldüğü kare (KİLİTLENMEMİŞ streak budama).
        # Uzun yaşayan aracın bbox'ından geçen her benzersiz kişi pstreak'te kalıcı giriş
        # bırakıyordu (MEM sızıntısı); bu harita ile giden kişilerin sayacı düşürülür.
        self._person_seen: dict[int, int] = {}

    # --- yardımcılar ------------------------------------------------------- #
    def persons_in_vehicle(self, vehicle: BBox, persons: list[Person]) -> list[Person]:
        """Kutusu araç kutusuyla yeterince örtüşen (kabindeki) kişiler."""
        return [
            p
            for p in persons
            if p.track_id is not None and _containment(p.bbox, vehicle) >= self.min_containment
        ]

    def _pick_corner(self, vehicle: BBox, pool: list[Person]) -> Person | None:
        """Havuzdan sürücüyü seç: DİKEY öncelikli köşe kuralı (kullanıcı kararı).

        Kural: hedef köşenin (vars. sağ-alt) **dikey** hizasına en yakın kişi BİRİNCİL
        ('en alt' önce gelir); **yatay** hiza (sağ/sol) yalnızca dikeyde berabere
        kalanları ayırır ('sonra en sağ'). Tam eşitlikte küçük track_id.
        """
        if not pool:
            return None
        tx, ty = self.target
        vw, vh = max(vehicle.width, 1e-6), max(vehicle.height, 1e-6)

        def rank(p: Person) -> tuple[float, float, int]:
            cx, cy = p.bbox.center
            nx = (cx - vehicle.x1) / vw  # 0..1 (araç içinde)
            ny = (cy - vehicle.y1) / vh
            v_prox = -abs(ny - ty)  # hedef dikey hizaya yakınlık — BİRİNCİL
            h_prox = -abs(nx - tx)  # hedef yatay hizaya yakınlık — İKİNCİL
            return (v_prox, h_prox, -p.track_id)

        return max(pool, key=rank)

    def select_candidate(self, vehicle: BBox, persons: list[Person]) -> Person | None:
        """Araç içindeki kişilerden sürücüyü seç: en alttaki (berabere → en sağdaki)."""
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

    # --- atama çekirdeği (dışlamalı havuz üzerinde tek araç) --------------- #
    def _assign_step(
        self, vehicle_id: int, vehicle: BBox, pool: list[Person], frame_idx: int
    ) -> DriverAssignment:
        """Bu araç için sürücüyü (pozisyonel) seç, yolcuları say/kilitle.

        `pool` → bu araca atanmış kişiler (başka araca kilitli yolcular hariç).
        Sürücü = kilitli-yolcu OLMAYAN kişiler arasında sağ-en-alt; her kare yeniden.
        """
        plock = self._passenger_locked.setdefault(vehicle_id, set())
        pstreak = self._passenger_streak.setdefault(vehicle_id, {})
        for _p in pool:  # bu kişiler bu karede görüldü → streak budama için işaretle
            self._person_seen[_p.track_id] = frame_idx

        # 1) Sürücü = kilitli-yolcu OLMAYAN kişiler arasında sağ-en-alt (pozisyonel).
        eligible = [p for p in pool if p.track_id not in plock]
        driver = self._pick_corner(vehicle, eligible)
        driver_id = driver.track_id if driver is not None else None
        driver_bbox = driver.bbox if driver is not None else None

        # 2) Sürücü dışındaki herkes yolcu; yolcu kaldıkça streak artar, eşikte kilitlenir.
        passengers: list[int] = []
        for p in pool:
            if p.track_id == driver_id:
                pstreak[p.track_id] = 0  # sürücü → yolcu sayacı sıfırlanır
                continue
            passengers.append(p.track_id)
            if p.track_id not in plock:
                pstreak[p.track_id] = pstreak.get(p.track_id, 0) + 1
                if pstreak[p.track_id] >= self.confirm_frames:
                    plock.add(p.track_id)
                    self._passenger_owner[p.track_id] = vehicle_id
                    log.info(
                        "Yolcu kilitlendi: araç=%s yolcu=%s (%d kare)",
                        vehicle_id,
                        p.track_id,
                        self.confirm_frames,
                    )

        # 3) Sürücü-varlık streak'i → 'sürücü kuruldu' tek-seferlik event + yapışkan locked.
        if driver is not None:
            self._driver_streak[vehicle_id] = self._driver_streak.get(vehicle_id, 0) + 1
            self._last_driver[vehicle_id] = driver_id
        else:
            self._driver_streak[vehicle_id] = 0
        streak = self._driver_streak[vehicle_id]
        newly = False
        if streak >= self.confirm_frames and vehicle_id not in self._established:
            self._established.add(vehicle_id)
            newly = True
            log.info("Sürücü kuruldu: araç=%s sürücü=%s (%d kare)", vehicle_id, driver_id, streak)

        return DriverAssignment(
            vehicle_id=vehicle_id,
            driver_id=driver_id,
            locked=vehicle_id in self._established,
            candidate_id=driver_id,
            streak=streak,
            newly_locked=newly,
            driver_bbox=driver_bbox,
            passenger_ids=passengers,
            locked_passenger_ids=[pid for pid in passengers if pid in plock],
        )

    # --- giriş noktası: tek araç (geriye dönük uyumluluk) ------------------ #
    def update(
        self, vehicle_id: int, vehicle: BBox, persons: list[Person], frame_idx: int
    ) -> DriverAssignment:
        """Tek bir araç için atamayı güncelle ve döndür.

        Havuz: araç kabinindeki kişiler eksi BAŞKA araca kilitli YOLCULAR.
        (Çok-araçlı kareler için `assign_frame` tercih edilir.)
        """
        self._last_seen[vehicle_id] = frame_idx
        if not self.enabled:
            return DriverAssignment(vehicle_id=vehicle_id)
        pool = [
            p
            for p in self.persons_in_vehicle(vehicle, persons)
            if self._passenger_owner.get(p.track_id, vehicle_id) == vehicle_id
        ]
        return self._assign_step(vehicle_id, vehicle, pool, frame_idx)

    # --- giriş noktası: global / dışlamalı (pipeline bunu kullanır) -------- #
    def assign_frame(
        self, vehicles: list[tuple[int, BBox]], persons: list[Person], frame_idx: int
    ) -> list[DriverAssignment]:
        """Kare içindeki TÜM araç↔kişi eşleşmesini global ve dışlamalı çöz.

        `vehicles` : (vehicle_id, araç_bbox) listesi — pipeline'daki tespit sırası.
        Dönüş      : aynı sıradaki `DriverAssignment` listesi (vehicles[i] ↔ out[i]).

        Dışlama:
          • Kilitli yolcular yalnızca sahibi oldukları aracın havuzuna girer.
          • Serbest her kişi, eşiği geçen araçlar arasından TEK bir araca (en iyi skor:
            örtüşme + merkez yakınlığı) atanır — iki aracın havuzunda birden bulunmaz.
        """
        for vid, _ in vehicles:
            self._last_seen[vid] = frame_idx
        if not self.enabled:
            return [DriverAssignment(vehicle_id=vid) for vid, _ in vehicles]

        valid = [p for p in persons if p.track_id is not None]
        # Pozisyona göre havuzlar (vehicle_id çakışabilir, ör. takipsiz araçlar -1).
        pools: list[list[Person]] = [[] for _ in vehicles]

        for p in valid:
            owner = self._passenger_owner.get(p.track_id)
            if owner is not None:
                # Kilitli yolcu: SADECE sahibi olan aracın havuzuna (başka araçlardan dışlanır).
                for i, (vid, _) in enumerate(vehicles):
                    if vid == owner:
                        pools[i].append(p)
                continue
            # Serbest kişi: eşiği geçen araçlar arasından en iyi skorlu TEK aracı seç.
            best_i: int | None = None
            best_s: float | None = None
            for i, (_vid, vbbox) in enumerate(vehicles):
                s = self._assign_score(p.bbox, vbbox)
                if s is None:
                    continue
                if best_s is None or s > best_s:
                    best_s, best_i = s, i
            if best_i is not None:
                pools[best_i].append(p)

        return [
            self._assign_step(vid, vbbox, pools[i], frame_idx)
            for i, (vid, vbbox) in enumerate(vehicles)
        ]

    # --- sorgu / bakım ----------------------------------------------------- #
    def driver_of(self, vehicle_id: int) -> int | None:
        """Araca en son atanan sürücü ID'si (pozisyonel; yoksa None)."""
        return self._last_driver.get(vehicle_id)

    def is_locked(self, vehicle_id: int) -> bool:
        """Sürücü KURULDU mu (≥confirm_frames ardışık sürücü-varlığı)."""
        return vehicle_id in self._established

    def passengers_of(self, vehicle_id: int) -> set[int]:
        """Araçta YOLCU olarak kilitlenmiş kişilerin ID'leri."""
        return set(self._passenger_locked.get(vehicle_id, set()))

    def prune(self, frame_idx: int) -> None:
        """max_age'den uzun süredir görünmeyen araçların tüm durumunu unut."""
        dead = [vid for vid, seen in self._last_seen.items() if frame_idx - seen > self.max_age]
        for vid in dead:
            for pid in self._passenger_locked.pop(vid, set()):
                # Bu araca ait yolcu sahipliğini serbest bırak (kişi yeniden atanabilsin).
                if self._passenger_owner.get(pid) == vid:
                    self._passenger_owner.pop(pid, None)
            self._passenger_streak.pop(vid, None)
            self._driver_streak.pop(vid, None)
            self._established.discard(vid)
            self._last_driver.pop(vid, None)
            self._last_seen.pop(vid, None)
        # Yolcu-streak sızıntısı (MEM): YAŞAYAN aracın bbox'ından geçip giden KİLİTLENMEMİŞ
        # kişilerin sayaçlarını düş — kilitli yolcu/owner 'yapışkan' semantiği KORUNUR
        # (locked pid'lere ve _passenger_owner'a dokunulmaz; davranış değişmez).
        for vid, pst in self._passenger_streak.items():
            locked = self._passenger_locked.get(vid, set())
            stale = [
                pid
                for pid in pst
                if pid not in locked
                and frame_idx - self._person_seen.get(pid, frame_idx) > self.max_age
            ]
            for pid in stale:
                pst.pop(pid, None)
        # _person_seen yalnız son-görülme cache'i — eski girişleri düş (sınırlı tut).
        for pid in [p for p, s in self._person_seen.items() if frame_idx - s > self.max_age]:
            self._person_seen.pop(pid, None)
