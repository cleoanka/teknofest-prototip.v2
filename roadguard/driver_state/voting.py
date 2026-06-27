"""ID-merkezli zaman tamponu (temporal voting) — Stage-2 Katman B çekirdeği.

Her HAM tespit bayrağı için son ``window`` karelik model çıktısını tutar. Bir bayrak
yalnızca pencerede en az ``min_votes`` kez görülürse "kararlı aktif" sayılır; böylece
tek-karelik yanlış pozitifler (flicker) elenir.

Ham bayraklar (modelin doğrudan tespit ettikleri):
    phone, smoking, seatbelt (kemer ŞERİDİ görüldü = kemer VAR), fatigue

Dikkat: ``no_seatbelt`` (ihlal) burada bir HAM bayrak DEĞİLDİR — o, kemerin YOKLUĞUNDAN
``DriverStateEngine`` tarafından türetilir. Bu yüzden voter ek olarak ``seen`` (gözlenen
kare sayısı) ve ``votes()`` sayaçlarını dışarı verir; engine türetmeyi bunlarla yapar.

Tasarım: bu katman MODELDEN BAĞIMSIZDIR. Ham bayrak nereden gelirse gelsin (placeholder
ya da eğitilmiş YOLO26l) mantık aynı çalışır.
"""

from __future__ import annotations

from collections import deque

from roadguard.schema import DriverState

#: Modelin doğrudan ürettiği HAM tespit bayrakları (seatbelt = kemer takılı gözlemi).
RAW_FLAGS = ("phone", "smoking", "seatbelt", "fatigue")


class TrackVoter:
    """Tek bir araç ID'si için ham bayrakların kayan-pencere oy tamponu."""

    def __init__(self, window: int, min_votes: int):
        self.window = window
        self.min_votes = min_votes
        self._hist: dict[str, deque[bool]] = {f: deque(maxlen=window) for f in RAW_FLAGS}
        self._conf: dict[str, deque[float]] = {f: deque(maxlen=window) for f in RAW_FLAGS}
        self.seen: int = 0  # bu ID için gözlenen kare sayısı (pencere boyutunda doyar)
        self.last_frame: int = 0

    def update(self, raw: DriverState, frame_idx: int) -> None:
        """Bu karenin HAM tahminini pencereye ekle (henüz karar verme)."""
        self.last_frame = frame_idx
        self.seen = min(self.seen + 1, self.window)
        for f in RAW_FLAGS:
            on = bool(getattr(raw, f))
            self._hist[f].append(on)
            self._conf[f].append(raw.confidence.get(f, 0.0) if on else 0.0)

    def votes(self, flag: str) -> int:
        """`flag` bayrağının penceredeki True oy sayısı."""
        return sum(self._hist[flag])

    def mean_conf(self, flag: str) -> float:
        """`flag` aktifken görülen güven skorlarının ortalaması (yoksa 0)."""
        positives = [c for c in self._conf[flag] if c > 0.0]
        return round(sum(positives) / len(positives), 3) if positives else 0.0

    def stable_raw(self) -> DriverState:
        """Pencereye bakıp KARARLI HAM durumu üret (henüz no_seatbelt TÜRETİLMEDEN).

        Her ham bayrak, son `window` karenin en az `min_votes` tanesinde True ise aktif
        kabul edilir. no_seatbelt türetmesini engine yapar (bkz. DriverStateEngine).
        """
        ds = DriverState()
        for f in RAW_FLAGS:
            if self.votes(f) >= self.min_votes:
                setattr(ds, f, True)
                ds.confidence[f] = self.mean_conf(f)
        return ds
