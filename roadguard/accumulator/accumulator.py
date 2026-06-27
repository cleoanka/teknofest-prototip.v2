"""ID-merkezli karar accumulator.

Tüm modül çıktılarını `TrackRecord`'a yazar, durum değişimlerinde `RoadGuardEvent` üretir
ve config'ten gelen risk kurallarını uygular. Kare-merkezli değil, ID-merkezli çalışır.
"""

from __future__ import annotations

import threading

from roadguard.schema import (
    BBox,
    DriverState,
    PlateState,
    SceneContext,
    SpeedState,
    TrackRecord,
    make_event,
)


class Accumulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks: dict[int, TrackRecord] = {}
        # CA-002: pipeline iş parçacığı tracks'i YAZAR (update_track ekler/mutasyon,
        # prune siler) iken bir tüketici (dashboard/REST okuyucu) active_tracks()/get()
        # ile OKUR. dict boyut-mutasyonu (prune sırasında del) eş-zamanlı list()/get
        # ile "dict changed size during iteration" / yarım görüntü üretebilir. Bu
        # re-entrant kilit yaz ile okuları serileştirir (RLock: update_track içinden
        # ileride iç çağrı yapılsa da deadlock olmaz; davranış aynı).
        self._lock = threading.RLock()
        self.rules = cfg.get("risk.rules", []) or []
        self.high_speed = float(cfg.get("risk.high_speed_kmh", 90))
        self.long_lived = int(cfg.get("risk.long_lived_frames", 90))
        # Sahne-seviyesi aktif hız limiti (SignTracker → set_scene). risk eval'de
        # 'speed.over_limit' koşulu bunu araç hızıyla karşılaştırır. None → kural pasif.
        self.active_speed_limit: int | None = None
        # Frame-tabanlı saat (pipeline idx/fps ile besler). Üretilen event'lerin ts'i
        # buradan alınır → QoDController ile AYNI zaman ekseni (offline eval'de
        # tekrar-üretilebilirlik: ts wall-clock time.time() yerine deterministik).
        # None → make_event varsayılanına (wall-clock) düşer (geriye dönük uyum).
        self._now: float | None = None
        # risk kurallarını __init__'te ön-derle (token bazlı string-parse'ı her frame
        # tekrarlama; davranış aynı, yalnız bir kez ayrıştırılır). Her kural:
        # (name, [token,...]). Boş kural listesinde _evaluate_risk erken döner.
        self._compiled_rules: list[tuple[str, list[str]]] = [
            (r.get("name", "risk"), r.get("all_of", []) or []) for r in self.rules
        ]

    def set_now(self, now: float) -> None:
        """Frame-tabanlı zaman ekseni (idx/fps). Pipeline her frame çağırır."""
        self._now = now

    def set_scene(self, scene: SceneContext) -> None:
        """Kare-başı sahne bağlamını güncelle — track risk eval'inden ÖNCE çağrılır."""
        self.active_speed_limit = scene.active_speed_limit_kmh

    def _event(self, track_id: int, type_: str, payload: dict):
        """make_event sarmalayıcı — frame-saatini (varsa) ts olarak geçirir."""
        return make_event(track_id, type_, payload, ts=self._now)

    # --- ana giriş noktası ------------------------------------------------- #
    def update_track(
        self,
        track_id: int,
        *,
        frame_idx: int,
        bbox: BBox,
        vehicle_class: str = "",
        plate: PlateState | None = None,
        driver: DriverState | None = None,
        speed: SpeedState | None = None,
        qod_active: bool = False,
        qod_profile: str | None = None,
    ):
        events = []
        with self._lock:  # CA-002: lookup+insert prune/okuyucularla serileşsin
            rec = self.tracks.get(track_id)
            is_new = rec is None
            if is_new:
                rec = TrackRecord(
                    track_id=track_id,
                    vehicle_class=vehicle_class,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    bbox=bbox,
                )
                self.tracks[track_id] = rec
        if is_new:
            events.append(
                self._event(
                    track_id,
                    "DETECTION_UPDATE",
                    {
                        "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                        "cls": vehicle_class,
                        "conf": bbox.conf,
                        "new": True,
                    },
                )
            )

        rec.bbox = bbox
        rec.last_frame = frame_idx
        if vehicle_class:
            rec.vehicle_class = vehicle_class

        if driver is not None:
            # active_flags() bir list comprehension — değişim dalında iki kez (karşılaştırma
            # + payload) çağrılıyordu; tek seferde hesapla ve yeniden kullan (per-frame
            # allocation azalt; davranış aynı).
            new_flags = driver.active_flags()
            if new_flags != rec.driver.active_flags():
                events.append(
                    self._event(
                        track_id,
                        "DRIVER_STATE",
                        {
                            "flags": new_flags,
                            "confidence": driver.confidence,
                        },
                    )
                )
            rec.driver = driver

        if plate is not None:
            prev_status = rec.plate.status
            # PlateReader aynı nesneyi yerinde mutasyona uğratır → snapshot al,
            # yoksa prev_status zaten güncellenmiş olur ve geçiş event'i kaçar.
            rec.plate = plate.model_copy(deep=True)
            if plate.status == "confirmed" and prev_status != "confirmed":
                events.append(
                    self._event(
                        track_id,
                        "PLATE_CONFIRMED",
                        {
                            "value": plate.value,
                            "confidence": plate.confidence,
                        },
                    )
                )
            elif plate.status == "rejected" and prev_status != "rejected":
                events.append(
                    self._event(
                        track_id,
                        "PLATE_REJECTED",
                        {
                            "reason": "consensus_fail",
                            "votes": plate.votes,
                        },
                    )
                )

        if speed is not None:
            prev = rec.speed
            rec.speed = speed
            if (
                speed.value_kmh != prev.value_kmh
                or speed.relative_velocity_flag != prev.relative_velocity_flag
                or speed.swerving != prev.swerving
            ):
                events.append(
                    self._event(
                        track_id,
                        "SPEED",
                        {
                            "value_kmh": speed.value_kmh,
                            "mode": speed.mode,
                            "relative_velocity_flag": speed.relative_velocity_flag,
                            "swerving": speed.swerving,
                            # km/h değerinin gerçek (kalibre) mi sezgisel mi olduğu —
                            # downstream tüketici güvenilmez değeri ayırt edebilsin.
                            "calibrated": speed.is_calibrated,
                        },
                    )
                )

        rec.qod_active = qod_active
        rec.qod_profile = qod_profile

        # risk değerlendirmesi — yeni tetiklenen kurallar için RISK_ALERT
        fired = self._evaluate_risk(rec)
        newly = [f for f in fired if f not in rec.risk_flags]
        rec.risk_flags = fired
        for rule_name in newly:
            if rule_name == "speed_limit_violation" and self.active_speed_limit is not None:
                # Hız-limiti ihlali zengin payload'lı kendi event'iyle çıkar (jüri demosu için).
                over = (
                    round(rec.speed.value_kmh - self.active_speed_limit, 1)
                    if rec.speed.value_kmh is not None
                    else None
                )
                events.append(
                    self._event(
                        track_id,
                        "SPEED_LIMIT_VIOLATION",
                        {
                            "speed_kmh": rec.speed.value_kmh,
                            "limit_kmh": self.active_speed_limit,
                            "over_by_kmh": over,
                            "plate": rec.plate.value,
                        },
                    )
                )
            else:
                events.append(self._event(track_id, "RISK_ALERT", {"rule": rule_name}))

        return rec, events

    # --- risk kuralları ---------------------------------------------------- #
    def _evaluate_risk(self, rec: TrackRecord) -> list[str]:
        if not self._compiled_rules:
            return []  # kural yoksa erken çık (boş-yapılandırma ucuz yolu)
        fired = []
        for name, conds in self._compiled_rules:
            if conds and all(self._cond(rec, c) for c in conds):
                fired.append(name)
        return fired

    def _cond(self, rec: TrackRecord, token: str) -> bool:
        if token == "speed.high":
            # MUTLAK yüksek hız (tabeladan bağımsız). Tek başına bir kural değil; bir
            # yapıtaşı. Artık varsayılan kurallarda kullanılmıyor (yerine speed.speeding).
            return rec.speed.value_kmh is not None and rec.speed.value_kmh >= self.high_speed
        if token == "speed.over_limit":
            # SAF tabela ihlali: yalnızca tabela limiti aktifken (None değilse) anlamlı.
            # Tabela yoksa pasif kalır → yanlış ihlal üretmez (SPEED_LIMIT_VIOLATION için).
            return (
                self.active_speed_limit is not None
                and rec.speed.value_kmh is not None
                and rec.speed.value_kmh > self.active_speed_limit
            )
        if token == "speed.speeding":
            # "Uygulanabilir limiti aşıyor mu?" → tabela varsa onun limiti, yoksa mutlak
            # high_speed tabanı. over_limit'ten farkı: tabela YOKKEN pasif kalmaz, tabana
            # düşer; böylece 30 bölgesinde 40 (limit aşımı) da, tabelasız yolda 150 (mutlak
            # tehlike) de yakalanır. 'distracted_speeding = phone + speeding' bunu kullanır.
            if rec.speed.value_kmh is None:
                return False
            if self.active_speed_limit is not None:
                return rec.speed.value_kmh > self.active_speed_limit
            return rec.speed.value_kmh >= self.high_speed
        if token == "speed.swerving":
            # Dikkatsiz sürüş: yanal yörünge zigzag/ani kayma bayrağı (SpeedEstimator
            # üretir, 16/8 kararlılık süzgecinden geçmiş halde gelir).
            return bool(rec.speed.swerving)
        if token == "track.long_lived":
            return (rec.last_frame - rec.first_frame) >= self.long_lived
        if token.startswith("driver."):
            return bool(getattr(rec.driver, token.split(".", 1)[1], False))
        return False

    # --- yardımcılar ------------------------------------------------------- #
    def active_tracks(self) -> list[TrackRecord]:
        with self._lock:  # CA-002: kilit altında DERİN-KOPYA snapshot
            # Pipeline iş parçacığı update_track'te kayıt alanlarını (plate/driver/speed)
            # KİLİT DIŞINDA mutasyona uğratır; okuyucu (REST/dashboard) referansı alıp
            # model_dump ederken yırtık okuma görürdü. Donmuş kopya bunu kapatır.
            return [r.model_copy(deep=True) for r in self.tracks.values()]

    def get(self, track_id: int) -> TrackRecord | None:
        with self._lock:
            r = self.tracks.get(track_id)
            return r.model_copy(deep=True) if r is not None else None

    def prune(self, frame_idx: int, max_age: int = 30) -> None:
        with self._lock:  # CA-002: silme okuyucularla serileşsin (size-change yarışı yok)
            dead = [tid for tid, r in self.tracks.items() if frame_idx - r.last_frame > max_age]
            for tid in dead:
                del self.tracks[tid]
