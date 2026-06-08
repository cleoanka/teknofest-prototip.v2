"""Voting buffer — oy birliği havuzu.

Sweet spot içindeyken yapılan ardışık OCR okumaları toplanır; tampon dolunca en sık
okuma `consensus_ratio` eşiğini geçerse konsensüs sağlanır.
"""

from __future__ import annotations

from collections import Counter


class VotingBuffer:
    def __init__(self, size: int, consensus_ratio: float):
        self.size = int(size)
        self.ratio = float(consensus_ratio)
        self.reads: list[str] = []

    def add(self, text: str | None) -> None:
        if text:
            self.reads.append(text)

    @property
    def full(self) -> bool:
        return len(self.reads) >= self.size

    def counts(self) -> dict[str, int]:
        return dict(Counter(self.reads))

    def consensus(self) -> tuple[str | None, float]:
        """(konsensüs_değeri|None, en_sık_okuma_oranı)."""
        if not self.reads:
            return None, 0.0
        value, count = Counter(self.reads).most_common(1)[0]
        frac = count / len(self.reads)
        return (value, frac) if frac >= self.ratio else (None, frac)

    def clear(self) -> None:
        self.reads.clear()
