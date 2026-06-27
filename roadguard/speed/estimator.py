"""Kalibrasyon-bağımlı hız tahmini (plan.md §6.6, §7).

Modlar:
- `metric`    : kalibrasyonsuz **oto-kalibrasyon** (eski prototip `ai/calibration.py`):
                araç-genişliği (varsa plaka 520 mm) → ppm(y) ölçek-alanı → yer düzlemi
                metrik yer değiştirmesi → gerçek km/h. Isınma bitene dek km/h yok
                (`is_calibrated=False`), yalnız göreli bayrak. Detay: roadguard/speed/calibration.py.
- `tripwire`  : iki sanal çizgi (line_a_y, line_b_y) arası frame-delta × gerçek mesafe.
- `ipm`       : homography (opsiyonel `homography_ipm` modülü — M12). Modül kapalıysa
                disabled davranışına güvenli düşüş.
- `disabled`  : hız iddiası yok; yalnızca `relative_velocity_flag` (anormal göreli hız).

Sistem kendi sınırlarını tanır: kalibrasyon yoksa hız uydurmaz.
`relative_velocity_flag` her modda üretilir (QoD optimize tetiği + accumulator risk için).
"""

from __future__ import annotations

import logging
from collections import deque
from itertools import islice
from types import SimpleNamespace

from roadguard.schema import BBox, SpeedState

log = logging.getLogger("roadguard.speed")


class SpeedEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.get("speed.mode", "disabled")
        self.calibration_file = cfg.get("speed.calibration_file")
        self.fps = 30.0
        # ipm modülü bayrağını __init__'te cache'le (hot _ipm yolunda her kare cfg.get
        # okumasını önle — diğer eşikler gibi, K-004 davranış aynı kalır).
        self._ipm_enabled = bool(cfg.get("optional_modules.homography_ipm", False))
        # Uzun-süreli akışta bellek sızıntısını önlemek için track son-görülme kaydı:
        # prune() bu haritaya göre giden track'lerin tüm durumunu (hist/lat/tw/track/
        # last_speed + metric Kalman/EMA) düşürür. (driver_lock.prune ile aynı desen.)
        self.max_age = int(cfg.get("speed.max_age", 30))
        self._last_seen: dict[int, int] = {}
        tw = cfg.get("speed.tripwire", {}) or {}
        self.line_a = float(tw.get("line_a_y", 0.40))
        self.line_b = float(tw.get("line_b_y", 0.70))
        self.real_distance = float(tw.get("real_distance_m", 20.0))
        self.rel_threshold = float(cfg.get("speed.relative_threshold", 0.012))
        self.window = int(cfg.get("speed.window", 8))
        self._hist: dict[int, deque] = {}  # track_id -> (frame_idx, cy_norm)
        # --- swerving (dikkatsiz sürüş) — yanal yörünge analizi ---------------- #
        # Genlik eşiği ARAÇ GENİŞLİĞİ birimindedir (ölçek/çözünürlük bağımsız, K-004),
        # pencere SANİYE cinsindendir (fps-bağımsız — 50fps'te de 25fps'te de aynı
        # salınım periyodunu görür). Algoritma: ZigZag ekstremum sayacı — cx serisi
        # mevcut uç noktadan amp kadar GERİ dönünce bir yön-değişimi (yalpalama)
        # sayılır. Monoton hareket (yaklaşma kayması, tek şerit değişimi) yapısal
        # olarak 0 üretir; trend çıkarmaya gerek kalmaz.
        sw = cfg.get("speed.swerving", {}) or {}
        self.swerve_enabled = bool(sw.get("enabled", True))
        self.swerve_window_s = float(sw.get("window_s", 3.0))
        # min_flips=2: gerçek yalpalama (sol→sağ→sol) ≥2 dönüş üretir; tek şerit
        # değişimi 0, oturma payı (overshoot) en çok 1 dönüş üretir → elenir.
        self.swerve_min_flips = int(sw.get("min_flips", 2))
        # amp O ANKİ araç genişliğiyle ölçeklenir (pencere medyanı değil): yaklaşan
        # araçta genişlik 5 kat büyür; medyan kullanmak uzak evredeki yalpalamayı
        # görünmez kılıyordu (gerçek video_3 ölçümü: salınım ≈ 0.11×genişlik;
        # 0.04-0.08 bandı yakalar, 3 videoda çapraz-FP=0 doğrulandı).
        self.swerve_amp_ratio = float(sw.get("amp_ratio", 0.06))
        self._lat_hist: dict[int, deque] = {}  # track_id -> (cx_px, bbox_w_px)
        self._tw: dict[int, dict] = {}  # tripwire durum makinesi
        self._ipm_warned = False
        # Ölü bölge (margin): araç kadraj kenarına bu kadar piksel yaklaşınca hız
        # hesaplama dondurulur (kadrajdan çıkarken bbox kırpılıp hız aniden düşmesin).
        self.frame_margin_px = int(cfg.get("speed.frame_margin_px", 50))
        # metric mod: oto-kalibrasyon estimator'ı + track-başı foot/ts geçmişi (lazy)
        self._metric = None
        self._tracks: dict[int, object] = {}
        self._last_speed: dict[int, float] = {}  # track_id -> son geçerli km/h (ölü bölge tutma)
        if self.mode == "metric":
            self._init_metric()

    def _init_metric(self) -> None:
        from roadguard.speed.calibration import MetricSpeedEstimator

        s = SimpleNamespace(
            plate_width_m=float(self.cfg.get("speed.plate_width_m", 0.520)),
            plate_aspect_tolerance=float(self.cfg.get("speed.plate_aspect_tolerance", 0.35)),
            vehicle_width_m=dict(
                self.cfg.get("speed.vehicle_width_m", {})
                or {"car": 1.80, "minibus": 2.00, "truck": 2.50, "bus": 2.50, "motorcycle": 0.80}
            ),
            vehicle_ppm_weight=float(self.cfg.get("speed.vehicle_ppm_weight", 0.25)),
            calib_min_samples=int(self.cfg.get("speed.calib_min_samples", 6)),
            speed_window_frames=int(self.cfg.get("speed.speed_window_frames", 6)),
            speed_max_accel_mps2=float(self.cfg.get("speed.speed_max_accel_mps2", 8.0)),
            speed_kalman_q=float(self.cfg.get("speed.speed_kalman_q", 3.0)),
            speed_kalman_r=float(self.cfg.get("speed.speed_kalman_r", 8.0)),
            speed_ema_alpha=float(self.cfg.get("speed.speed_ema_alpha", 0.0)),
            speed_metric_max_kmh=float(self.cfg.get("speed.speed_metric_max_kmh", 200.0)),
        )
        self._metric = MetricSpeedEstimator(s)

    # --- ana giriş --------------------------------------------------------- #
    def update(
        self,
        track_id: int,
        bbox: BBox,
        frame_idx: int,
        frame_shape: tuple[int, ...] | None = None,
        plate_bbox: BBox | None = None,
    ) -> SpeedState:
        h = frame_shape[0] if frame_shape else 1.0
        cy_norm = bbox.center[1] / h if h else 0.0
        self._last_seen[track_id] = frame_idx  # bellek temizliği (prune) için son-görülme
        hist = self._hist.setdefault(track_id, deque(maxlen=self.window))
        hist.append((frame_idx, cy_norm))

        rel_flag = self._relative_flag(hist)
        swerving = self._swerving_flag(track_id, bbox, frame_idx)

        if self.mode == "metric":
            st = self._metric_update(track_id, bbox, frame_idx, rel_flag, frame_shape, plate_bbox)
        elif self.mode == "tripwire":
            value = self._tripwire(track_id, cy_norm, frame_idx)
            st = SpeedState(mode="tripwire", value_kmh=value, relative_velocity_flag=rel_flag)
        elif self.mode == "ipm":
            st = self._ipm(track_id, bbox, frame_idx, rel_flag, frame_shape)
        else:
            st = SpeedState(mode="disabled", value_kmh=None, relative_velocity_flag=rel_flag)
        st.swerving = swerving
        return st

    # --- swerving (yanal yörünge) ------------------------------------------ #
    def _swerving_flag(self, track_id: int, bbox: BBox, frame_idx: int) -> bool:
        """Yanal yörüngeden dikkatsiz-sürüş (swerving = yalpalama) bayrağı üret.

        Yöntem — ZigZag ekstremum sayacı (v1 ``Track.is_swerving`` fikrinin fps- ve
        ölçek-bağımsız hali): son ``window_s`` saniyenin cx serisinde, mevcut uç
        noktadan ``amp_ratio×medyan-genişlik``ten fazla GERİ dönüş bir yön-değişimi
        sayılır; sayı ``min_flips``'e ulaşırsa swerving.

        Neden bu tasarım: monoton hareketler (kameraya yaklaşma perspektif kayması,
        tek şerit değişimi S-eğrisi) hiç geri dönmediği için yapısal olarak 0 üretir
        — trend modeli/çıkarması gerekmez. Bbox titremesi amp kapısının altında
        kalır. Sentetik doğrulama: S-eğrisi=0, overshoot'lu şerit değişimi=1,
        1.5 periyot yalpalama=2, 3 periyot=4-6.
        """
        if not self.swerve_enabled:
            return False
        maxlen = max(8, int(self.swerve_window_s * max(self.fps, 1.0)))
        lat = self._lat_hist.get(track_id)
        if lat is None or lat.maxlen != maxlen:
            lat = deque(lat or (), maxlen=maxlen)
            self._lat_hist[track_id] = lat
        lat.append((bbox.center[0], max(bbox.width, 1.0)))
        if len(lat) < maxlen // 3:
            return False
        up_ext = down_ext = lat[0][0]
        direction, reversals = 0, 0
        # islice(lat, 1, None): deque'in tamamını her kare list()'e kopyalamadan
        # (track başına ~90 öğe) ilk öğeyi atlayarak gez — kare-başı tahsisi azaltır.
        for x, w in islice(lat, 1, None):
            amp = self.swerve_amp_ratio * w  # o anki genişliğe göre eşik
            if direction >= 0:
                up_ext = max(up_ext, x)
                if up_ext - x > amp:  # tepe onaylandı → aşağı dönüş
                    if direction == 1:
                        reversals += 1
                    direction, down_ext = -1, x
                    continue
            if direction <= 0:
                down_ext = min(down_ext, x)
                if x - down_ext > amp:  # dip onaylandı → yukarı dönüş
                    if direction == -1:
                        reversals += 1
                    direction, up_ext = 1, x
        return reversals >= self.swerve_min_flips

    # --- ölü bölge (margin) ------------------------------------------------ #
    def _in_frame_border(self, bbox: BBox, frame_shape: tuple[int, ...] | None) -> bool:
        """Araç bbox'ı kadraj kenarındaki `frame_margin_px` şeridine değiyor mu?

        Değiyorsa araç kadrajdan çıkmaya başlamıştır → bbox kırpılır, foot noktası ve
        genişlik bozulur → hız hesabı güvenilmez. O kare için yeni hız üretilmez.
        """
        m = self.frame_margin_px
        if not frame_shape or m <= 0:
            return False
        h, w = frame_shape[0], frame_shape[1]
        return bbox.x1 <= m or bbox.y1 <= m or bbox.x2 >= w - m or bbox.y2 >= h - m

    # --- metric (oto-kalibrasyon) ----------------------------------------- #
    def _metric_update(
        self,
        track_id: int,
        bbox: BBox,
        frame_idx: int,
        rel_flag: bool,
        frame_shape: tuple[int, ...] | None,
        plate_bbox: BBox | None = None,
    ) -> SpeedState:
        from roadguard.speed.calibration import SpeedTrack

        if self._metric is None:
            self._init_metric()

        # Ölü bölge: araç kadraj kenarındaysa yeni hız HESAPLAMA; sınıra girmeden
        # önceki son geçerli hızı tut ve araç tamamen çıkana kadar onu yazdır.
        if self._in_frame_border(bbox, frame_shape):
            held = self._last_speed.get(track_id)
            return SpeedState(
                mode="metric",
                value_kmh=held,
                relative_velocity_flag=rel_flag,
                is_calibrated=held is not None,
            )

        ts = frame_idx / self.fps if self.fps > 0 else float(frame_idx)
        track = self._tracks.get(track_id)
        if track is None:
            track = SpeedTrack(track_id)
            self._tracks[track_id] = track
        track.update(((bbox.x1 + bbox.x2) / 2.0, bbox.y2), ts)

        # Ölçek-alanını besle: plaka (520 mm referans, ağırlık 1.0) varsa en kesin ppm
        # kaynağıdır; LP dedektörü plakayı bulduğunda gelir. Araç genişliği (ağırlık
        # 0.25) her zaman yedek olarak eklenir → ısınma hızlanır, km/h daha doğru oturur.
        if plate_bbox is not None:
            self._metric.observe_plate(plate_bbox)
        self._metric.observe_vehicle(bbox, bbox.cls)
        self._metric.maybe_fit()
        value, is_cal = self._metric.estimate(track)
        if value is not None:
            self._last_speed[track_id] = value  # son geçerli hız (ölü bölge tutma için)
        return SpeedState(
            mode="metric",
            value_kmh=value,
            relative_velocity_flag=rel_flag,
            is_calibrated=is_cal,
        )

    # --- göreli hız anomalisi (kalibrasyon gerektirmez) ------------------- #
    def _relative_flag(self, hist: deque) -> bool:
        if len(hist) < 2:
            return False
        (f0, y0), (f1, y1) = hist[0], hist[-1]
        df = f1 - f0
        if df <= 0:
            return False
        v_norm = abs(y1 - y0) / df  # normalize dikey hız (ekran/kare)
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
    def _ipm(
        self, track_id: int, bbox: BBox, frame_idx: int, rel_flag: bool, frame_shape=None
    ) -> SpeedState:
        enabled = self._ipm_enabled  # __init__'te cache'lendi (hot yol — her kare cfg.get yok)
        if enabled:
            try:
                from roadguard.optional.homography_ipm import ipm_speed  # M12

                value = ipm_speed(self.cfg, track_id, bbox, frame_idx, self.fps, frame_shape)
                return SpeedState(mode="ipm", value_kmh=value, relative_velocity_flag=rel_flag)
            except Exception as e:  # noqa: BLE001
                if not self._ipm_warned:
                    log.warning("IPM modülü kullanılamadı (%s) → disabled davranışı", e)
                    self._ipm_warned = True
        elif not self._ipm_warned:
            log.warning("speed.mode=ipm ama optional_modules.homography_ipm kapalı → disabled")
            self._ipm_warned = True
        return SpeedState(mode="disabled", value_kmh=None, relative_velocity_flag=rel_flag)

    # --- bellek temizliği (uzun-süreli akış) ------------------------------- #
    def prune(self, frame_idx: int) -> None:
        """`max_age`'den uzun süredir görünmeyen track'lerin TÜM durumunu düşür.

        Uzun-süreli akışta her benzersiz track_id kalıcı olarak deque/dict/Kalman
        biriktirir (sızıntı). Pipeline._prune'dan kare-başı çağrılır (driver_lock/
        driver ile aynı desen). Ayrıca eski tripwire durum-makinesini temizleyerek
        recycled track_id'nin bayat `s['a']` ile taze `line_b` geçişini eşleyip
        absürt km/h üretmesini engeller (stale-state tehlikesi).
        """
        dead = [tid for tid, seen in self._last_seen.items() if frame_idx - seen > self.max_age]
        for tid in dead:
            self._hist.pop(tid, None)
            self._lat_hist.pop(tid, None)
            self._tw.pop(tid, None)
            self._tracks.pop(tid, None)
            self._last_speed.pop(tid, None)
            self._last_seen.pop(tid, None)
        if dead and self._metric is not None:
            # metric mod: Kalman/EMA durumunu da temizle (artık aktif track'lere göre).
            self._metric.prune(set(self._last_seen))
