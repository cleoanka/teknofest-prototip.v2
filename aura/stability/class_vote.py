"""Track başına araç-sınıfı oylaması (çoğunluk + güven ağırlığı + hafif unutma).

Neden var?
    Fine-tune dedektör aynı fiziksel aracı kareler arasında farklı sınıflarla
    görebiliyor (gerçek video ölçümü: video_1/2'de araç İLK karede 0.79-0.84
    güvenle 'truck', sonra kalıcı 'car'; video_3'te yakın plandaki otomobil tek
    tek karelerde 'truck'a dönüyor). Son-tespit-kazanır yaklaşımı bu titremeyi
    dashboard'a, hız kalibrasyonuna (sınıf-bazlı genişlik önseli) ve event
    payload'larına taşıyordu.

Tasarım:
    Her track için sınıf→ağırlık sözlüğü tutulur; her karede o karenin tespit
    güveni sınıfın ağırlığına eklenir ve EN AĞIR sınıf döndürülür. Hafif üstel
    unutma (decay) erken yanlış oyların sonsuza dek baskın kalmasını önler —
    araç gerçekte sınıf değiştirmez ama İLK kareler en uzak/en bulanık
    karelerdir (en güvenilmez kanıt). Eşitlikte alfabetik küçük sınıf seçilir
    (deterministik çıktı). K-004: kural videoya değil takip istatistiğine bağlı.
"""

from __future__ import annotations


class TrackClassVoter:
    """Kümülatif, güven-ağırlıklı sınıf oyu; ``update`` çoğunluk sınıfını döndürür."""

    def __init__(self, cfg):
        cv = cfg.get("tracking.class_vote", {}) or {}
        self.enabled = bool(cv.get("enabled", True))
        # Kare başına unutma çarpanı: 0.98 ≈ ~34 karede eski oyların yarısı söner.
        # Gerçek ölçüm (video_1): araç uzaktayken ONLARCA kare üst üste 'truck'
        # görülebiliyor — unutma yavaşsa (0.995) yakın/net 'car' kanıtı baskınlığı
        # geç devralıyordu. 0.98 tek-kare flip'leri yine rahat bastırır (tek 0.9'luk
        # oy, son ~50 karenin birikimini geçemez). 1.0 = saf kümülatif.
        self.decay = float(cv.get("decay", 0.98))
        self._votes: dict[int, dict[str, float]] = {}

    def update(self, track_id: int | None, cls: str, conf: float = 1.0) -> str:
        """Bu karenin (sınıf, güven) oyunu işle ve track'in kararlı sınıfını döndür.

        Takipsiz tespitler (``track_id`` None/negatif) oylanmaz — kimliksiz kutuya
        geçmiş bağlanamaz; o karenin ham sınıfı aynen geri verilir.
        """
        if not self.enabled or not cls or track_id is None or track_id < 0:
            return cls
        votes = self._votes.setdefault(track_id, {})
        if self.decay < 1.0:
            for k in votes:
                votes[k] *= self.decay
        votes[cls] = votes.get(cls, 0.0) + max(float(conf), 1e-3)
        # Eşitlikte deterministik: önce ağırlık, sonra alfabetik küçük ad.
        return max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def stable_class(self, track_id: int) -> str | None:
        """Oy birikmiş track'in güncel çoğunluk sınıfı (telemetri/teşhis için)."""
        votes = self._votes.get(track_id)
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def prune(self, live_track_ids: set[int]) -> None:
        """Artık yaşamayan track'lerin oylarını bırak (uzun koşumda bellek hijyeni)."""
        for tid in [t for t in self._votes if t not in live_track_ids]:
            self._votes.pop(tid, None)
