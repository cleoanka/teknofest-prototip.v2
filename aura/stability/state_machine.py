"""16/8 kararlılık state machine.

M2: pass-through arayüz (önerilen değeri olduğu gibi döndürür).
M4: kayar pencere (16 kare) — yeni durum ancak son 16 karenin ≥8'inde tutarlıysa
yazılır; aksi halde yüksek güvenli önceki değer korunur (flicker koruması).
"""
from __future__ import annotations


class StabilityTracker:
    def __init__(self, cfg):
        self.window = int(cfg.get("stability.window", 16))
        self.min_consistent = int(cfg.get("stability.min_consistent", 8))

    def update(self, key: str, value, conf: float = 1.0):
        """`key` = f"{track_id}:{alan}" için önerilen değeri kararlılık süzgecinden geçir.

        M2: pass-through. M4'te gerçek 16/8 oylaması gelir.
        """
        # TODO(M4): per-key kayar pencere + 16/8 konsensüs.
        return value
