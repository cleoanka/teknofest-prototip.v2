"""Track başına araç-sınıfı oylaması (çoğunluk + güven & alan ağırlığı + unutma).

Neden var?
    Fine-tune dedektör aynı fiziksel aracı kareler arasında farklı sınıflarla
    görebiliyor. Gerçek video ÖLÇÜMÜ (13 Haz): video_2'de ana araç İLK 53 kare
    ham tespitte 'truck' (uzakta/arkadan car silüeti truck'a benziyor, conf 0.84),
    sonra yakınlaşınca kalıcı 'car'. Son-tespit-kazanır VEYA saf çoğunluk bu
    titremeyi dashboard'a, hız kalibrasyonuna (sınıf-bazlı genişlik önseli) ve
    event payload'larına taşıyordu.

Tasarım — ALAN-AĞIRLIKLI oy (kilit içgörü):
    Her track için sınıf→ağırlık sözlüğü tutulur. Her kareni oyu
    ``conf × alan_faktörü`` ile ağırlıklanır: YAKIN/BÜYÜK araç daha net görünür,
    sınıfı daha güvenilirdir; UZAK/KÜÇÜK araç sınıfı belirsizdir (en güvenilmez
    kanıt). Böylece birkaç yakın 'car' karesi, onlarca uzak 'truck' karesini
    devralır — plaka 'boyut-farkında kanıt'ıyla aynı felsefe. Hafif üstel unutma
    (decay) eski oyların baskınlığını ayrıca zayıflatır. Eşitlikte alfabetik küçük
    sınıf (deterministik). K-004: kural videoya değil takip+ölçek istatistiğine bağlı.
"""

from __future__ import annotations


class TrackClassVoter:
    """Kümülatif, güven & alan ağırlıklı sınıf oyu; ``update`` çoğunluk sınıfını döndürür."""

    def __init__(self, cfg):
        cv = cfg.get("tracking.class_vote", {}) or {}
        self.enabled = bool(cv.get("enabled", True))
        # Kare başına unutma çarpanı. VARSAYILAN 1.0 = saf kümülatif: araç sınıfı
        # fiziksel olarak değişmez, tüm ömür-boyu kanıt birikir. Gerçek video dersi:
        # decay<1 + alan-ağırlığı, GEÇ gelen büyük-alan (yakın) yanlış-sınıf
        # tespitlerine aşırı tepki verip salınıma yol açıyordu (video_3: car→truck→car).
        # Yanlış-ERKEN kanıt zaten alan-ağırlığıyla (uzak=küçük=düşük ağırlık) zayıf;
        # decay'e gerek yok. (<1 yapılırsa unutma açılır — özel durumlar için.)
        self.decay = float(cv.get("decay", 1.0))
        # Alan faktörü tabanı: bbox alanı / kare alanı bunun altındaysa bile bu kadar
        # ağırlık verilir (çok uzak araç oyu sıfırlanmasın, sadece çok zayıf kalsın).
        self.area_floor = float(cv.get("area_floor", 0.0008))
        self._votes: dict[int, dict[str, float]] = {}
        # Bellek hijyeni (MEM-004): per-track oy sözlüğü giden track'ler için kalıcı
        # birikiyordu. _last_seen son-görülme karesini tutar; prune(frame_idx) max_age
        # grace'li temizler — kısa oklüzyonda (recycled id) kümülatif oy KORUNUR
        # (davranış-koruyan; class-vote ömür-boyu birikime dayanır).
        self.max_age = int(cv.get("max_age", 30))
        self._last_seen: dict[int, int] = {}

    def update(
        self,
        track_id: int | None,
        cls: str,
        conf: float = 1.0,
        area_norm: float | None = None,
        frame_idx: int | None = None,
    ) -> str:
        """Bu karenin oyunu işle ve track'in kararlı sınıfını döndür.

        ``conf``: tespit güveni. ``area_norm``: bbox alanı / kare alanı (0..1) —
        verilirse oy ``conf × max(area_norm, area_floor)`` ile ağırlıklanır (yakın/net
        araca daha çok güven). Verilmezse yalnız ``conf`` kullanılır (geriye uyum).
        Takipsiz tespitler (``track_id`` None/negatif) oylanmaz — o karenin ham
        sınıfı aynen geri verilir.
        """
        if not self.enabled or not cls or track_id is None or track_id < 0:
            return cls
        if frame_idx is not None:
            self._last_seen[track_id] = frame_idx  # prune grace için son-görülme
        votes = self._votes.setdefault(track_id, {})
        if self.decay < 1.0:
            for k in votes:
                votes[k] *= self.decay
        w = max(float(conf), 1e-3)
        if area_norm is not None:
            w *= max(float(area_norm), self.area_floor)
        votes[cls] = votes.get(cls, 0.0) + w
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
            self._last_seen.pop(tid, None)

    def prune_aged(self, frame_idx: int) -> None:
        """max_age grace'li bellek hijyeni (MEM-004) — pipeline kare-başı çağırır.

        Yalnız ``max_age``'den UZUN süredir görünmeyen track'in oy sözlüğü düşer; kısa
        oklüzyon/recycled-id grace içinde kümülatif oyu KORUR (davranış-koruyan).
        update(..., frame_idx=...) ile beslenen _last_seen'e dayanır; frame_idx hiç
        geçilmemişse (boş _last_seen) hiçbir şey düşmez (geriye uyum)."""
        dead = [tid for tid, seen in self._last_seen.items() if frame_idx - seen > self.max_age]
        for tid in dead:
            self._votes.pop(tid, None)
            self._last_seen.pop(tid, None)
