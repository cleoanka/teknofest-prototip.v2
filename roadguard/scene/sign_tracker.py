"""Sahne-seviyesi trafik tabelası takibi — hız-limiti bağlamı.

ID-merkezli accumulator araç-başına karar üretir; trafik tabelası ise bir araca
değil **sahneye** aittir. Bu modül her kareki tabela tespitlerinden "aktif hız
limitini" çıkarır ve araç tabelayı geçtikten sonra da onu ``persistence_frames``
boyunca geçerli tutar (tabela sürekli görünmez ama kural sürer).

Akış:
  - En güvenilir hız-limiti tabelası seçilir (``sign.value_map`` ile km/h'ye çözülür).
  - Aktif limit DEĞİŞİNCE bir ``SPEED_LIMIT_DETECTED`` event'i üretilir (track_id=-1).
  - Limit, son görülmeden ``persistence_frames`` kare sonra sessizce düşer.

Üretilen ``SceneContext`` accumulator'a ``set_scene()`` ile verilir; oradaki
``speed.over_limit`` risk koşulu bu limiti araç hızıyla karşılaştırır.
"""

from __future__ import annotations

from roadguard.schema import RoadGuardEvent, SceneContext, make_event

# Sahne-seviyesi event'ler bir araca ait olmadığından RoadGuardEvent.track_id için sentinel.
# (-1, pipeline'da takip kurulmamış geçici tespitler için de kullanılan değerdir.)
SCENE_TRACK_ID = -1


class SignTracker:
    """Tabela tespitlerinden aktif hız limitini çıkaran sahne-seviyesi takipçi."""

    def __init__(self, cfg):
        self.enabled = bool(cfg.get("sign.enabled", True))
        vmap = cfg.get("sign.value_map", {}) or {}
        # sınıf adı → km/h (str anahtar, int değer); generic tabelalar haritada yer almaz
        self.value_map: dict[str, int] = {str(k): int(v) for k, v in vmap.items()}
        self.persistence = int(cfg.get("sign.persistence_frames", 150))
        self.min_conf = float(cfg.get("sign.min_conf", 0.40))
        self._limit: int | None = None
        self._src: str | None = None
        self._last_seen: int | None = None

    def limit_of(self, cls: str) -> int | None:
        """Tabela sınıfının hız limiti (km/h); hız-limiti tabelası değilse None."""
        return self.value_map.get(cls)

    @property
    def active_limit(self) -> int | None:
        return self._limit

    def update(
        self, signs, frame_idx: int, now: float | None = None
    ) -> tuple[SceneContext, list[RoadGuardEvent]]:
        """Bu kareki tabelaları işle → (güncel SceneContext, üretilen event'ler).

        `now` (frame-saati = idx/fps) verilirse SPEED_LIMIT_DETECTED ts'i deterministik
        olur (accumulator/qod ile AYNI eksen). Verilmezse make_event wall-clock'a düşer.
        """
        events: list[RoadGuardEvent] = []
        if not self.enabled:
            return SceneContext(sign_count=len(signs)), events

        # Bu karedeki en güvenilir hız-limiti tabelasını seç (generic tabelalar atlanır).
        best = None
        best_limit: int | None = None
        for s in signs:
            if s.bbox.conf < self.min_conf:
                continue
            lim = self.value_map.get(s.cls)
            if lim is None:
                continue
            if best is None or s.bbox.conf > best.bbox.conf:
                best, best_limit = s, lim

        if best is not None:
            self._last_seen = frame_idx
            if self._limit != best_limit:  # limit değişti → tek seferlik event
                self._limit, self._src = best_limit, best.cls
                events.append(
                    make_event(
                        SCENE_TRACK_ID,
                        "SPEED_LIMIT_DETECTED",
                        {
                            "speed_limit_kmh": best_limit,
                            "cls": best.cls,
                            "conf": best.bbox.conf,
                        },
                        ts=now,  # deterministik frame-saati (wall-clock kayması yok)
                    )
                )
        elif (
            self._limit is not None
            and self._last_seen is not None
            and frame_idx - self._last_seen > self.persistence
        ):
            # Tabela uzun süredir görülmedi → aktif limiti sessizce düşür.
            self._limit, self._src, self._last_seen = None, None, None

        return (
            SceneContext(
                active_speed_limit_kmh=self._limit,
                speed_limit_source_cls=self._src,
                sign_count=len(signs),
            ),
            events,
        )
