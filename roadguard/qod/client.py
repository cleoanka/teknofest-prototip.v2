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

from roadguard.schema import RoadGuardEvent, make_event

log = logging.getLogger("roadguard.qod")


class QoDController:
    def __init__(self, cfg):
        self.backend = cfg.get("qod.backend", "mock")
        self.endpoint = cfg.get("qod.endpoint")
        self.profiles = cfg.get("qod.profiles", {}) or {}
        self.min_active = float(cfg.get("qod.histeresis.min_active_seconds", 3))
        self.cooldown = float(cfg.get("qod.histeresis.cooldown_seconds", 5))
        self._now = 0.0
        self._sessions: dict[int, dict] = {}  # track_id -> {profile,kind,since,reason}
        self._last_release: dict[int, float] = {}  # track_id -> release zamanı
        self._pending: list[RoadGuardEvent] = []

    # --- zaman ------------------------------------------------------------- #
    def set_now(self, now: float) -> None:
        self._now = now

    def _profile_for(self, kind: str) -> str:
        return self.profiles.get(kind, kind.upper())

    # --- talepler ---------------------------------------------------------- #
    def request(self, track_id: int, kind: str, reason: str) -> None:
        """kind: 'quality' | 'optimize'. Histerezis kurallarına uyar."""
        s = self._sessions.get(track_id)
        if s is not None:
            # Zaten aktif (salınımı önle): YENİ TRIGGER event'i üretme. Ancak tetikleyici
            # HÂLÂ sürüyorsa (her frame request_optimize çağrılıyor) min_active sayacını
            # tazele — yoksa tick() kritik an (swerving/yaklaşma) devam ederken oturumu
            # min_active sonrası düşürür ve cooldown boyunca kapalı kalır (docstring'in
            # 'LOW_LATENCY oturumu yaşamaya devam eder' niyetiyle çelişen bug).
            s["since"] = self._now
            return
        lr = self._last_release.get(track_id)
        if lr is not None and (self._now - lr) < self.cooldown:
            return  # cooldown içinde — yeniden tetikleme
        profile = self._profile_for(kind)
        self._sessions[track_id] = {
            "profile": profile,
            "kind": kind,
            "since": self._now,
            "reason": reason,
        }
        self._pending.append(
            make_event(
                track_id,
                "QOD_TRIGGER",
                {
                    "profile": profile,
                    "kind": kind,
                    "reason": reason,
                },
                # frame-saatini ts olarak geçir → accumulator event'leriyle AYNI zaman
                # ekseni (offline eval tekrar-üretilebilirlik; wall-clock kayması yok).
                ts=self._now,
            )
        )
        log.debug("QoD TRIGGER track=%s %s (%s)", track_id, profile, reason)

    def request_quality(self, track_id: int, reason: str) -> None:
        self.request(track_id, "quality", reason)

    def request_optimize(self, track_id: int, reason: str) -> None:
        self.request(track_id, "optimize", reason)

    # --- otomatik bırakma (histerezis) ------------------------------------ #
    def tick(self) -> None:
        if not self._sessions:
            return  # ucuz erken çıkış: çoğu frame'de oturum yok (gereksiz liste kopyası önlenir)
        for tid, s in list(self._sessions.items()):
            if self._now - s["since"] >= self.min_active:
                del self._sessions[tid]
                self._last_release[tid] = self._now
                self._pending.append(
                    make_event(tid, "QOD_RELEASE", {"profile": s["profile"]}, ts=self._now)
                )
                log.debug("QoD RELEASE track=%s %s", tid, s["profile"])

    def prune(self) -> None:
        """Bellek hijyeni (MEM-00x): cooldown'ı geçmiş `_last_release` girdilerini düşür.

        `_last_release[tid]`'in tek tüketicisi `request()`'teki cooldown guard'ıdır
        (`_now - lr < cooldown`). `_now - lr >= cooldown` olduğunda girdi semantik
        olarak ÖLÜdür (silmek davranış-eşdeğer: yok=tetikle, süresi-dolmuş=yine tetikle).
        Aksi halde QoD tetiği almış her benzersiz track_id kalıcı bir girdi bırakır →
        7/24 akışta monoton büyüme. Saniye-ekseni `self._now` ile çalışır (idx değil).
        """
        if not self._last_release:
            return
        dead = [tid for tid, lr in self._last_release.items() if self._now - lr >= self.cooldown]
        for tid in dead:
            del self._last_release[tid]

    def release(self, track_id: int) -> None:
        s = self._sessions.pop(track_id, None)
        if s:
            self._last_release[track_id] = self._now
            self._pending.append(
                make_event(track_id, "QOD_RELEASE", {"profile": s["profile"]}, ts=self._now)
            )

    def release_quality(self, track_id: int) -> None:
        """Yalnızca KALİTE oturumunu bırak (optimize oturumlarına dokunma).

        Plaka onaylandığı an HIGH_THROUGHPUT'u tutmaya devam etmek kaynak israfıdır
        (şartname: 'yalnızca kritik anlarda' + sorumlu bırakma kanıtı). Aynı track'te
        swerving/approach kaynaklı LOW_LATENCY oturumu varsa o yaşamaya devam eder.
        """
        s = self._sessions.get(track_id)
        if s and s.get("kind") == "quality":
            self.release(track_id)

    # --- durum / drain ----------------------------------------------------- #
    def state(self, track_id: int) -> tuple[bool, str | None]:
        s = self._sessions.get(track_id)
        return (True, s["profile"]) if s else (False, None)

    def active_sessions(self) -> dict[int, dict]:
        return dict(self._sessions)

    def drain_events(self) -> list[RoadGuardEvent]:
        ev = self._pending
        self._pending = []
        return ev
