"""ID-merkezli zaman tamponu (temporal voting) — Stage-2 Katman B çekirdeği.

Her sürücü-durum bayrağı (phone/smoking/no_seatbelt/fatigue) için son ``window``
karelik HAM model çıktısını tutar. Bir bayrak yalnızca pencerede en az ``min_votes``
kez görülürse "kararlı aktif" sayılır; aksi halde pasif kalır. Böylece tek-karelik
yanlış pozitifler (titreşim/flicker) event'e dönüşmeden elenir.

Tasarım: bu katman MODELDEN BAĞIMSIZDIR. Ham bayrak nereden gelirse gelsin
(şimdilik deterministik placeholder, sonra eğitilmiş YOLO26l) mantık aynı çalışır.
Yapay zeka eklendiğinde bu dosyaya dokunmaya gerek yoktur.
"""

from __future__ import annotations

from collections import deque

from aura.schema import DriverState

#: İzlenen sürücü-durum bayrakları (DriverState alanlarıyla birebir aynı isimler).
FLAGS = ("phone", "smoking", "no_seatbelt", "fatigue")


class TrackVoter:
    """Tek bir araç ID'si için tüm bayrakların kayan-pencere oy tamponu.

    Her ``update`` çağrısı o karenin ham tahminini pencereye ekler; ``stable``
    pencereye bakıp kararlı (oy eşiğini geçen) DriverState üretir.
    """

    def __init__(self, window: int, min_votes: int):
        self.window = window
        self.min_votes = min_votes
        # Her bayrak için son `window` karenin True/False geçmişi (maxlen ile otomatik kayar).
        self._hist: dict[str, deque[bool]] = {f: deque(maxlen=window) for f in FLAGS}
        # Aynı pencerede, bayrak aktifken görülen güven skorları (kararlı conf raporu için).
        self._conf: dict[str, deque[float]] = {f: deque(maxlen=window) for f in FLAGS}
        self.last_frame: int = 0

    def update(self, raw: DriverState, frame_idx: int) -> None:
        """Bu karenin HAM tahminini pencereye ekle (henüz karar verme)."""
        self.last_frame = frame_idx
        for f in FLAGS:
            on = bool(getattr(raw, f))
            self._hist[f].append(on)
            # Bayrak kapalıysa 0 yaz: kararlı conf yalnızca "açık" oylardan hesaplanır.
            self._conf[f].append(raw.confidence.get(f, 0.0) if on else 0.0)

    def stable(self) -> DriverState:
        """Pencereye bakıp KARARLI sürücü durumunu üret.

        Bir bayrak, son `window` karenin en az `min_votes` tanesinde True ise aktif
        kabul edilir. Aktif bayrağın confidence'ı, pencerede onu True yapan karelerin
        ortalama güveni olarak raporlanır.
        """
        ds = DriverState()
        for f in FLAGS:
            votes = sum(self._hist[f])
            if votes >= self.min_votes:
                setattr(ds, f, True)
                positives = [c for c in self._conf[f] if c > 0.0]
                if positives:
                    ds.confidence[f] = round(sum(positives) / len(positives), 3)
        return ds
