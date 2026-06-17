"""Stage-2 sürücü-durum motoru — Katman B (ID-merkezli işleme) orkestratörü.

Akış:
    Stage-1 → (track_id, sürücü ROI [+ Stage-1 nesne kanıtı])
        → [Katman A] DriverClassifier.infer(roi, track_id)  → HAM bayraklar (tek kare)
        → [aux füzyonu] Stage-1 phone/smoking nesnesi OR'lanır (varsa)
        → [Katman B] TrackVoter (her ID için zaman tamponu) → KARARLI bayraklar
        → DriverState (accumulator'a gider, DRIVER_STATE event'i üretir)

Bu motor, her ``track_id`` için ayrı bir ``TrackVoter`` tutar — sistem kare-merkezli
değil, ID-merkezli çalışır. Bir aracın sürücü-durumu zaman içinde o ID'nin tamponunda
birikir; araç sahneden çıkınca tampon ``prune`` ile düşer.

Katman A (model) takılabilir: pose-hibrit geometrisi, fine-tune YOLO26l detection veya
mock — fabrika (``build_driver_classifier``) hangisini döndürürse Katman B değişmez.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aura.driver_state.classifier import build_driver_classifier
from aura.driver_state.voting import TrackVoter

if TYPE_CHECKING:
    import numpy as np

    from aura.schema import DriverState

log = logging.getLogger("aura.driver_state.engine")


class DriverStateEngine:
    """ID-merkezli sürücü-durum motoru (Katman A modelini + Katman B oylamasını birleştirir)."""

    def __init__(self, cfg):
        self.cfg = cfg
        # Katman A — ham, durumsuz(a yakın) model. Pose-hibrit / YOLO26l / mock.
        self.model = build_driver_classifier(cfg)
        # Katman B — oylama parametreleri (config'ten; yoksa güvenli varsayılan = 16/8).
        # Anahtarlar models.driver_state.voting altında (tek config sözleşmesi).
        self.window = int(cfg.get("models.driver_state.voting.window", 16))
        self.min_votes = int(cfg.get("models.driver_state.voting.min_votes", 8))
        self.max_age = int(cfg.get("models.driver_state.voting.max_age", 30))
        # track_id → o ID'nin zaman tamponu (ID-merkezli durum burada yaşar).
        self.voters: dict[int, TrackVoter] = {}
        log.info(
            "DriverStateEngine: window=%d min_votes=%d max_age=%d",
            self.window,
            self.min_votes,
            self.max_age,
        )

    def _infer(self, roi, track_id: int):
        """Katman A modelini çağır. Model track_id kabul ediyorsa geçir (pose hibrit
        telefon-nesnesi latch belleği ID'ye bağlıdır); etmiyorsa (eski/stub) düş."""
        try:
            return self.model.infer(roi, track_id=track_id)
        except TypeError:
            return self.model.infer(roi)

    def process(
        self,
        track_id: int,
        cabin_roi: np.ndarray | None,
        frame_idx: int = 0,
        aux_flags: dict[str, float] | None = None,
    ) -> DriverState:
        """Bir aracın ID'si + sürücü ROI'sini al → o ID için KARARLI DriverState üret.

        ``aux_flags``: Stage-1 dedektörünün tam karede görüp BU aracın kutusuna düşen
        phone/smoking nesnelerinin {alan: conf} kanıtı; ham tahmine OR'lanır (kanıt da
        Katman B oylamasından geçer → tek-kare nesne FP'si event olamaz).
        """
        raw = self._infer(cabin_roi, track_id)
        if aux_flags:
            for field, conf in aux_flags.items():
                setattr(raw, field, True)
                raw.confidence[field] = max(raw.confidence.get(field, 0.0), float(conf))
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
