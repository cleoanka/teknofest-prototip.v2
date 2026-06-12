"""Stage-2 sürücü-durum motoru — Katman B (ID-merkezli işleme) orkestratörü.

Akış:
    arkadaş (Stage-1) → (track_id, kabin ROI)
        → [Katman A] DriverClassifier.infer(roi)  → HAM bayraklar (tek kare)
        → [Katman B] TrackVoter (her ID için zaman tamponu) → KARARLI bayraklar
        → DriverState (accumulator'a gider, DRIVER_STATE event'i üretir)

Bu motor, her ``track_id`` için ayrı bir ``TrackVoter`` tutar — yani sistem
kare-merkezli değil, ID-merkezli çalışır. Bir aracın sürücü-durumu zaman içinde
o ID'nin tamponunda birikir; araç sahneden çıkınca tampon ``prune`` ile düşer.

Model katmanı (A) takılabilir: şimdilik deterministik placeholder, eğitilmiş
YOLO26l geldiğinde fabrika onu döndürür — bu dosya değişmez.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aura.driver_state.classifier import build_driver_classifier
from aura.driver_state.voting import TrackVoter
from aura.schema import DriverState

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("aura.driver_state.engine")


class DriverStateEngine:
    """ID-merkezli sürücü-durum motoru (Katman A modelini + Katman B oylamasını birleştirir)."""

    def __init__(self, cfg):
        self.cfg = cfg
        # Katman A — ham, durumsuz (stateless) model. Placeholder ya da YOLO26l.
        self.model = build_driver_classifier(cfg)
        # Katman B — oylama parametreleri (config'ten; yoksa güvenli varsayılan = 16/8).
        self.window = int(cfg.get("driver_state.voting.window", 16))
        self.min_votes = int(cfg.get("driver_state.voting.min_votes", 8))
        self.max_age = int(cfg.get("driver_state.voting.max_age", 30))
        # track_id → o ID'nin zaman tamponu (ID-merkezli durum burada yaşar).
        self.voters: dict[int, TrackVoter] = {}
        log.info(
            "DriverStateEngine: window=%d min_votes=%d max_age=%d",
            self.window,
            self.min_votes,
            self.max_age,
        )

    def process(
        self, track_id: int, cabin_roi: np.ndarray | None, frame_idx: int = 0
    ) -> DriverState:
        """Bir aracın ID'si + kabin ROI'sini al → o ID için KARARLI DriverState üret.

        Arkadaşının Stage-1'inin verdiği ``track_id`` burada ana anahtardır: aynı ID
        her karede aynı tampona yazar, böylece sürücü-durumu zaman içinde birikir.
        """
        # Katman A: bu karenin ham tahmini (henüz oylanmamış).
        raw = self.model.infer(cabin_roi)
        # Katman B: bu ID'nin tamponunu bul/oluştur, ham tahmini ekle, kararlısını döndür.
        voter = self.voters.get(track_id)
        if voter is None:
            voter = TrackVoter(self.window, self.min_votes)
            self.voters[track_id] = voter
        voter.update(raw, frame_idx)
        return voter.stable()

    def prune(self, frame_idx: int) -> None:
        """Uzun süredir görülmeyen ID'lerin tamponunu düşür (bellek sızıntısını önler)."""
        dead = [tid for tid, v in self.voters.items() if frame_idx - v.last_frame > self.max_age]
        for tid in dead:
            del self.voters[tid]

    def forget(self, track_id: int) -> None:
        """Tek bir ID'nin tamponunu unut (araç kesin sahneden çıktıysa)."""
        self.voters.pop(track_id, None)


def build_driver_engine(cfg) -> DriverStateEngine:
    """Config'e göre Stage-2 motorunu kur (model katmanını fabrika seçer)."""
    return DriverStateEngine(cfg)
