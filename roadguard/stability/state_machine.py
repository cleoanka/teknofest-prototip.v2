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
        # Bellek hijyeni (MEM-003): anahtarlar "{track_id}:{alan}" — uzun akışta her
        # benzersiz track kalıcı pencere/commit biriktirir. prune_aged(frame_idx)
        # max_age grace'li temizler (speed/driver_lock deseni). update(..., frame_idx)
        # _track_last_seen'i besler; beslenmezse prune hiçbir şey düşürmez (geriye uyum).
        self.max_age = int(cfg.get("stability.max_age", 30))
        self._track_last_seen: dict[int, int] = {}

    @staticmethod
    def _default(value: Any) -> Any:
        # Bool alanlar için "kanıtlanana kadar durum yok" → False.
        return False if isinstance(value, bool) else value

    def update(self, key: str, value: Any, conf: float = 1.0, frame_idx: int | None = None) -> Any:
        """`key` için önerilen `value`'yu 16/8 süzgecinden geçir, kararlı değeri döndür."""
        if frame_idx is not None:
            prefix, sep, _ = key.partition(":")
            if sep:
                try:
                    self._track_last_seen[int(prefix)] = frame_idx  # prune grace
                except ValueError:
                    pass  # sayısal-olmayan prefix — track-bağlı değil
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

    # --- bellek temizliği (uzun-süreli akış) ------------------------------- #
    def prune_aged(self, frame_idx: int) -> None:
        """max_age grace'li bellek hijyeni (MEM-003) — pipeline kare-başı çağırır.

        Anahtarlar ``"{track_id}:{alan}"`` (örn. ``"7:speed.rel"``). Yalnız
        ``max_age``'den UZUN süredir görünmeyen track'in TÜM anahtarları (pencere+
        commit) düşer; kısa oklüzyon/recycled-id grace içinde kayan pencereler KORUNUR
        (davranış-koruyan; speed/driver_lock deseni). update(..., frame_idx=...) ile
        beslenen _track_last_seen'e dayanır; beslenmezse hiçbir şey düşmez (geriye uyum).
        Track-bağlı olmayan (":" içermeyen) anahtarlar her zaman korunur."""
        dead_tracks = {
            tid for tid, seen in self._track_last_seen.items() if frame_idx - seen > self.max_age
        }
        if not dead_tracks:
            return
        dead_keys = []
        for key in self._windows:
            prefix, sep, _ = key.partition(":")
            if not sep:
                continue
            try:
                tid = int(prefix)
            except ValueError:
                continue
            if tid in dead_tracks:
                dead_keys.append(key)
        for key in dead_keys:
            self._windows.pop(key, None)
            self._committed.pop(key, None)
        for tid in dead_tracks:
            self._track_last_seen.pop(tid, None)
