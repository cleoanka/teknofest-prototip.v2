"""ID-merkezli karar accumulator.

Tüm modül çıktılarını `TrackRecord`'a yazar, durum değişimlerinde `AuraEvent` üretir
ve config'ten gelen risk kurallarını uygular. Kare-merkezli değil, ID-merkezli çalışır.
"""
from __future__ import annotations

from aura.schema import BBox, DriverState, PlateState, SpeedState, TrackRecord, make_event


class Accumulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks: dict[int, TrackRecord] = {}
        self.rules = cfg.get("risk.rules", []) or []
        self.high_speed = float(cfg.get("risk.high_speed_kmh", 90))
        self.long_lived = int(cfg.get("risk.long_lived_frames", 90))

    # --- ana giriş noktası ------------------------------------------------- #
    def update_track(self, track_id: int, *, frame_idx: int, bbox: BBox,
                     vehicle_class: str = "", plate: PlateState | None = None,
                     driver: DriverState | None = None, speed: SpeedState | None = None,
                     qod_active: bool = False, qod_profile: str | None = None):
        events = []
        rec = self.tracks.get(track_id)
        is_new = rec is None
        if is_new:
            rec = TrackRecord(track_id=track_id, vehicle_class=vehicle_class,
                              first_frame=frame_idx, last_frame=frame_idx, bbox=bbox)
            self.tracks[track_id] = rec
            events.append(make_event(track_id, "DETECTION_UPDATE", {
                "bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                "cls": vehicle_class, "conf": bbox.conf, "new": True,
            }))

        rec.bbox = bbox
        rec.last_frame = frame_idx
        if vehicle_class:
            rec.vehicle_class = vehicle_class

        if driver is not None:
            if driver.active_flags() != rec.driver.active_flags():
                events.append(make_event(track_id, "DRIVER_STATE", {
                    "flags": driver.active_flags(), "confidence": driver.confidence,
                }))
            rec.driver = driver

        if plate is not None:
            prev_status = rec.plate.status
            rec.plate = plate
            if plate.status == "confirmed" and prev_status != "confirmed":
                events.append(make_event(track_id, "PLATE_CONFIRMED", {
                    "value": plate.value, "confidence": plate.confidence,
                }))
            elif plate.status == "rejected" and prev_status != "rejected":
                events.append(make_event(track_id, "PLATE_REJECTED", {
                    "reason": "consensus_fail", "votes": plate.votes,
                }))

        if speed is not None:
            prev = rec.speed
            rec.speed = speed
            if (speed.value_kmh != prev.value_kmh
                    or speed.relative_velocity_flag != prev.relative_velocity_flag):
                events.append(make_event(track_id, "SPEED", {
                    "value_kmh": speed.value_kmh, "mode": speed.mode,
                    "relative_velocity_flag": speed.relative_velocity_flag,
                }))

        rec.qod_active = qod_active
        rec.qod_profile = qod_profile

        # risk değerlendirmesi — yeni tetiklenen kurallar için RISK_ALERT
        fired = self._evaluate_risk(rec)
        newly = [f for f in fired if f not in rec.risk_flags]
        rec.risk_flags = fired
        for rule_name in newly:
            events.append(make_event(track_id, "RISK_ALERT", {"rule": rule_name}))

        return rec, events

    # --- risk kuralları ---------------------------------------------------- #
    def _evaluate_risk(self, rec: TrackRecord) -> list[str]:
        fired = []
        for rule in self.rules:
            conds = rule.get("all_of", [])
            if conds and all(self._cond(rec, c) for c in conds):
                fired.append(rule.get("name", "risk"))
        return fired

    def _cond(self, rec: TrackRecord, token: str) -> bool:
        if token == "speed.high":
            return rec.speed.value_kmh is not None and rec.speed.value_kmh >= self.high_speed
        if token == "track.long_lived":
            return (rec.last_frame - rec.first_frame) >= self.long_lived
        if token.startswith("driver."):
            return bool(getattr(rec.driver, token.split(".", 1)[1], False))
        return False

    # --- yardımcılar ------------------------------------------------------- #
    def active_tracks(self) -> list[TrackRecord]:
        return list(self.tracks.values())

    def get(self, track_id: int) -> TrackRecord | None:
        return self.tracks.get(track_id)

    def prune(self, frame_idx: int, max_age: int = 30) -> None:
        dead = [tid for tid, r in self.tracks.items() if frame_idx - r.last_frame > max_age]
        for tid in dead:
            del self.tracks[tid]
