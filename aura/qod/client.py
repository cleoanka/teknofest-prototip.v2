"""CAMARA QoD istemcisi — dinamik kaynak yönetimi + histerezis.

Yalnızca gerektiğinde (anomali/tehlike veya yetersiz kalite) QoD profili talep eder.
Histerezis (min_active + cooldown) tetikle-bırak salınımını önler. Üretilen
QOD_TRIGGER / QOD_RELEASE event'leri pipeline tarafından drain edilip stream'e yayılır.

Tetikleyiciler:
- Kalite (HIGH_THROUGHPUT): voting buffer ret / yetersiz piksel
- Optimizasyon (LOW_LATENCY): hız/yörünge anomalisi (M6)

`backend: mock` → in-process session yönetimi (M7'de qod_mock HTTP servisine bağlanır).
`backend: camara` → gerçek Turkcell endpoint (final ortam).
"""
from __future__ import annotations

import logging

from aura.schema import AuraEvent, make_event

log = logging.getLogger("aura.qod")


class QoDController:
    def __init__(self, cfg):
        self.backend = cfg.get("qod.backend", "mock")
        self.endpoint = cfg.get("qod.endpoint")
        self.profiles = cfg.get("qod.profiles", {}) or {}
        self.min_active = float(cfg.get("qod.histeresis.min_active_seconds", 3))
        self.cooldown = float(cfg.get("qod.histeresis.cooldown_seconds", 5))
        self._now = 0.0
        self._sessions: dict[int, dict] = {}        # track_id -> {profile,kind,since,reason}
        self._last_release: dict[int, float] = {}   # track_id -> release zamanı
        self._pending: list[AuraEvent] = []

    # --- zaman ------------------------------------------------------------- #
    def set_now(self, now: float) -> None:
        self._now = now

    def _profile_for(self, kind: str) -> str:
        return self.profiles.get(kind, kind.upper())

    # --- talepler ---------------------------------------------------------- #
    def request(self, track_id: int, kind: str, reason: str) -> None:
        """kind: 'quality' | 'optimize'. Histerezis kurallarına uyar."""
        if track_id in self._sessions:
            return  # zaten aktif (salınımı önle)
        lr = self._last_release.get(track_id)
        if lr is not None and (self._now - lr) < self.cooldown:
            return  # cooldown içinde — yeniden tetikleme
        profile = self._profile_for(kind)
        self._sessions[track_id] = {
            "profile": profile, "kind": kind, "since": self._now, "reason": reason,
        }
        self._pending.append(make_event(track_id, "QOD_TRIGGER", {
            "profile": profile, "kind": kind, "reason": reason,
        }))
        log.debug("QoD TRIGGER track=%s %s (%s)", track_id, profile, reason)

    def request_quality(self, track_id: int, reason: str) -> None:
        self.request(track_id, "quality", reason)

    def request_optimize(self, track_id: int, reason: str) -> None:
        self.request(track_id, "optimize", reason)

    # --- otomatik bırakma (histerezis) ------------------------------------ #
    def tick(self) -> None:
        for tid, s in list(self._sessions.items()):
            if self._now - s["since"] >= self.min_active:
                del self._sessions[tid]
                self._last_release[tid] = self._now
                self._pending.append(make_event(tid, "QOD_RELEASE", {"profile": s["profile"]}))
                log.debug("QoD RELEASE track=%s %s", tid, s["profile"])

    def release(self, track_id: int) -> None:
        s = self._sessions.pop(track_id, None)
        if s:
            self._last_release[track_id] = self._now
            self._pending.append(make_event(track_id, "QOD_RELEASE", {"profile": s["profile"]}))

    # --- durum / drain ----------------------------------------------------- #
    def state(self, track_id: int) -> tuple[bool, str | None]:
        s = self._sessions.get(track_id)
        return (True, s["profile"]) if s else (False, None)

    def active_sessions(self) -> dict[int, dict]:
        return dict(self._sessions)

    def drain_events(self) -> list[AuraEvent]:
        ev = self._pending
        self._pending = []
        return ev
