"""16/8 kararlılık state machine (plan.md §6.3).

Her `track_id × durum alanı` için bağımsız kayar pencere (16 kare). Yeni bir durum
ancak son `window` karenin en az `min_consistent`'ında tutarlı tespit edilirse
"commit" edilir; aksi halde yüksek güvenli önceki değer korunur (override yok).
Bu, flickering / geçici gürültünün sistemi yanlış alarma sürüklemesini engeller.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class StabilityTracker:
    def __init__(self, cfg):
        self.window = int(cfg.get("stability.window", 16))
        self.min_consistent = int(cfg.get("stability.min_consistent", 8))
        self._windows: dict[str, deque] = {}
        self._committed: dict[str, Any] = {}

    @staticmethod
    def _default(value: Any) -> Any:
        # Bool alanlar için "kanıtlanana kadar durum yok" → False.
        return False if isinstance(value, bool) else value

    def update(self, key: str, value: Any, conf: float = 1.0) -> Any:
        """`key` için önerilen `value`'yu 16/8 süzgecinden geçir, kararlı değeri döndür."""
        w = self._windows.get(key)
        if w is None or w.maxlen != self.window:
            w = deque(w or (), maxlen=self.window)
            self._windows[key] = w
        w.append(value)

        # Mevcut önerinin (value) pencerede kaç kez göründüğü
        count = sum(1 for v in w if v == value)
        if count >= self.min_consistent:
            self._committed[key] = value

        return self._committed.get(key, self._default(value))

    # --- introspeksiyon (test/debug) -------------------------------------- #
    def support(self, key: str, value: Any) -> int:
        w = self._windows.get(key)
        return sum(1 for v in w if v == value) if w else 0

    def committed(self, key: str, default: Any = None) -> Any:
        return self._committed.get(key, default)

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._windows.clear()
            self._committed.clear()
        else:
            self._windows.pop(key, None)
            self._committed.pop(key, None)
