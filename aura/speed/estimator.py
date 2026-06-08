"""Kalibrasyon-bağımlı hız tahmini (plan.md §6.6, §7).

Modlar:
- `tripwire`  : iki sanal çizgi (line_a_y, line_b_y) arası frame-delta × gerçek mesafe.
- `ipm`       : homography (opsiyonel `homography_ipm` modülü — M12). Modül kapalıysa
                disabled davranışına güvenli düşüş.
- `disabled`  : hız iddiası yok; yalnızca `relative_velocity_flag` (anormal göreli hız).

Sistem kendi sınırlarını tanır: kalibrasyon yoksa hız uydurmaz.
"""
from __future__ import annotations

import logging
from collections import deque

from aura.schema import BBox, SpeedState

log = logging.getLogger("aura.speed")


class SpeedEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.get("speed.mode", "disabled")
        self.calibration_file = cfg.get("speed.calibration_file")
        self.fps = 30.0
        tw = cfg.get("speed.tripwire", {}) or {}
        self.line_a = float(tw.get("line_a_y", 0.40))
        self.line_b = float(tw.get("line_b_y", 0.70))
        self.real_distance = float(tw.get("real_distance_m", 20.0))
        self.rel_threshold = float(cfg.get("speed.relative_threshold", 0.012))
        self.window = int(cfg.get("speed.window", 8))
        self._hist: dict[int, deque] = {}            # track_id -> (frame_idx, cy_norm)
        self._tw: dict[int, dict] = {}               # tripwire durum makinesi
        self._ipm_warned = False

    # --- ana giriş --------------------------------------------------------- #
    def update(self, track_id: int, bbox: BBox, frame_idx: int,
               frame_shape: tuple[int, ...] | None = None) -> SpeedState:
        h = frame_shape[0] if frame_shape else 1.0
        cy_norm = bbox.center[1] / h if h else 0.0
        hist = self._hist.setdefault(track_id, deque(maxlen=self.window))
        hist.append((frame_idx, cy_norm))

        rel_flag = self._relative_flag(hist)

        if self.mode == "tripwire":
            value = self._tripwire(track_id, cy_norm, frame_idx)
            return SpeedState(mode="tripwire", value_kmh=value, relative_velocity_flag=rel_flag)
        if self.mode == "ipm":
            return self._ipm(track_id, bbox, frame_idx, rel_flag)
        return SpeedState(mode="disabled", value_kmh=None, relative_velocity_flag=rel_flag)

    # --- göreli hız anomalisi (kalibrasyon gerektirmez) ------------------- #
    def _relative_flag(self, hist: deque) -> bool:
        if len(hist) < 2:
            return False
        (f0, y0), (f1, y1) = hist[0], hist[-1]
        df = f1 - f0
        if df <= 0:
            return False
        v_norm = abs(y1 - y0) / df                  # normalize dikey hız (ekran/kare)
        return v_norm > self.rel_threshold

    # --- tripwire ---------------------------------------------------------- #
    def _tripwire(self, track_id: int, cy_norm: float, frame_idx: int) -> float | None:
        s = self._tw.setdefault(track_id, {"prev": None, "a": None, "b": None, "kmh": None})
        if s["kmh"] is not None:
            return s["kmh"]
        prev = s["prev"]
        if prev is not None:
            if s["a"] is None and prev < self.line_a <= cy_norm:
                s["a"] = frame_idx
            elif s["a"] is not None and s["b"] is None and prev < self.line_b <= cy_norm:
                s["b"] = frame_idx
                dframes = s["b"] - s["a"]
                if dframes > 0:
                    seconds = dframes / max(self.fps, 1e-6)
                    s["kmh"] = round((self.real_distance / seconds) * 3.6, 1)
                    log.debug("tripwire track=%s %.1f km/h", track_id, s["kmh"])
        s["prev"] = cy_norm
        return s["kmh"]

    # --- ipm (opsiyonel modül) -------------------------------------------- #
    def _ipm(self, track_id: int, bbox: BBox, frame_idx: int, rel_flag: bool) -> SpeedState:
        enabled = self.cfg.get("optional_modules.homography_ipm", False)
        if enabled:
            try:
                from aura.optional.homography_ipm import ipm_speed  # M12

                value = ipm_speed(self.cfg, track_id, bbox, frame_idx, self.fps)
                return SpeedState(mode="ipm", value_kmh=value, relative_velocity_flag=rel_flag)
            except Exception as e:  # noqa: BLE001
                if not self._ipm_warned:
                    log.warning("IPM modülü kullanılamadı (%s) → disabled davranışı", e)
                    self._ipm_warned = True
        elif not self._ipm_warned:
            log.warning("speed.mode=ipm ama optional_modules.homography_ipm kapalı → disabled")
            self._ipm_warned = True
        return SpeedState(mode="disabled", value_kmh=None, relative_velocity_flag=rel_flag)
