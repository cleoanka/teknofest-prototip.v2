"""Oto-kalibrasyon: piksel→metre ölçeği (ppm) sahnenin kendisinden türetilir.

Kaynak: eski prototip `ai/calibration.py` (gercek_hiz_plani.md §1, §4, §7).
Sabit yol-kenarı kamerasında kamera yüksekliği/açısı/odak verilmez; metrik km/h
için 1 pikselin yerde kaç metreye karşılık geldiğini (ppm) bilmek şarttır. Dışarıdan
alınamaz → sahneden öğrenilir:

  • plate_ppm()  — TR plakası 520 mm referansından yerel ppm (varsa en kesin ölçü).
  • ScaleField   — onlarca aracın ölçümünü görüntü-y'sine (derinlik vekili) göre
                   toplayıp ppm(y) doğrusunu (aykırı-dayanıklı) uydurur.
  • MetricSpeedEstimator — ppm(y) ile aracın yer düzlemindeki metrik yer değiştirmesini
                   hesaplayıp v = Δs/Δt · 3.6 ile km/h üretir; pencere-medyan + ivme
                   aykırı reddi + track-başı Kalman ile titremeyi bastırır.

Bağımlılık: yalnız numpy (cv2 gerekmez). Homografi opsiyoneldir (set_homography);
verilmezse ppm(y) ölçek-alanı kullanılır. Isınma bitene dek estimate() (None, False)
döndürür → çağıran metrik iddiada bulunmaz (is_calibrated=False).
"""

from __future__ import annotations

from collections import deque
from itertools import islice

import numpy as np

from roadguard.schema import BBox

# TR Tip-1 plaka en/boy oranı: 520/120 ≈ 4.33. Plaka cepheden görünürken bu orana
# yakındır; açıyla (foreshortening) daralınca oran sapar → o ölçümü güvenilmez sayarız.
_PLATE_ASPECT = 520.0 / 120.0


def plate_ppm(
    plate_bbox: BBox | None, plate_width_m: float = 0.520, aspect_tolerance: float = 0.35
) -> float | None:
    """Plaka piksel genişliğinden yerel ölçeği (piksel/metre) döndür.

    Foreshortening koruması: en/boy oranı 4.33'ten `aspect_tolerance` bağıl payından
    fazla saparsa plaka eğik görünüyordur → None (ölçümü düşür).
    """
    if plate_bbox is None or plate_width_m <= 0:
        return None
    w = plate_bbox.x2 - plate_bbox.x1
    h = plate_bbox.y2 - plate_bbox.y1
    if w <= 1 or h <= 1:
        return None
    aspect = w / h
    if abs(aspect - _PLATE_ASPECT) / _PLATE_ASPECT > aspect_tolerance:
        return None
    return w / plate_width_m


class KalmanSpeed1D:
    """1-D sabit-hız Kalman filtresi — kare-başı hız tahminindeki titremeyi bastırır.

    Durum x = düzgünleştirilmiş hız (m/s). Filtre lineer, kazanç dizisi yalnız
    (P0, Q, R)'ye bağlı → düzgünleştirme oranı ölçekten bağımsızdır. Q: süreç gürültüsü
    (hızın ne kadar değişebildiği), R: ölçüm gürültüsü (anlık tahminin güvenilmezliği).
    """

    __slots__ = ("x", "P", "Q", "R", "_init")

    def __init__(self, Q: float = 3.0, R: float = 8.0):
        self.Q = Q
        self.R = R
        self.x = 0.0
        self.P = 200.0
        self._init = False

    def update(self, z: float) -> float:
        if not self._init:
            self.x = z
            self._init = True
            return z
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= 1.0 - K
        return self.x


class SpeedTrack:
    """Metrik hız için tek araç geçmişi: yer-temas noktası (foot) + zaman damgası.

    foot = bbox alt-orta piksel (aracın yere değdiği nokta); ppm(y)/homografi yer
    düzlemi hesabı bunu kullanır. ts = video-zaman çizgisi damgası (s). İkisi paralel
    (aynı maxlen) tutulur; "gerçek Δt" buradan ölçülür (düşürülen kare/VFR doğru ele alınır).
    """

    def __init__(self, track_id: int, maxlen: int = 16):
        self.track_id = track_id
        self.foot_history: deque = deque(maxlen=maxlen)
        self.ts_history: deque = deque(maxlen=maxlen)

    def update(self, foot: tuple[float, float], ts: float | None) -> None:
        self.foot_history.append((float(foot[0]), float(foot[1])))
        self.ts_history.append(ts)


class ScaleField:
    """Görüntü dikey konumuna bağlı yerel ölçek alanı: ppm(y) = slope·y + intercept.

    Sabit kamerada derinlik ≈ y'nin tek-yönlü fonksiyonudur; ppm de y ile (yaklaşık
    doğrusal) değişir. Onlarca aracın plaka/araç ölçümü birikir, aykırı-dayanıklı bir
    doğru uydurulur. Yeterli y-yayılımı yoksa sabit ppm = medyan(ppm) kullanılır.
    """

    def __init__(self, min_samples: int = 6, maxlen: int = 4000):
        self.min_samples = max(2, min_samples)
        self._ys: deque = deque(maxlen=maxlen)
        self._ppms: deque = deque(maxlen=maxlen)
        self._ws: deque = deque(maxlen=maxlen)
        self._slope: float = 0.0
        self._intercept: float = 0.0
        self._median_ppm: float = 0.0
        self._fitted: bool = False

    def add(self, y: float, ppm: float | None, weight: float = 1.0) -> None:
        """Ölçüm ekle. weight = güven (1/sigma): plaka yüksek (1.0), araç-genişliği
        düşük (~0.25) — gürültülü kaynak az, kesin kaynak çok ağırlık alır."""
        if ppm is None or ppm <= 0 or not np.isfinite(ppm) or weight <= 0:
            return
        self._ys.append(float(y))
        self._ppms.append(float(ppm))
        self._ws.append(float(weight))

    @property
    def n_samples(self) -> int:
        return len(self._ppms)

    @property
    def is_ready(self) -> bool:
        return self._fitted

    def fit(self) -> bool:
        """Birikmiş ölçümlerden ppm(y)'yi uydur. Başarılıysa True.

        Tek tur artık-bazlı aykırı reddi (|resid| > 2.5·MAD atılır) ile sağlamlaştırılır.
        """
        n = len(self._ppms)
        if n < self.min_samples:
            return False
        ys = np.asarray(self._ys, dtype=float)
        ppms = np.asarray(self._ppms, dtype=float)
        ws = np.asarray(self._ws, dtype=float)
        self._median_ppm = float(np.median(ppms))

        # y yeterince yayılmadıysa eğim güvenilmez → (ağırlıklı) sabit ppm
        if float(np.std(ys)) < 1e-3:
            self._slope = 0.0
            self._intercept = float(np.average(ppms, weights=ws))
            self._fitted = True
            return True

        slope, intercept = np.polyfit(ys, ppms, 1, w=ws)
        resid = ppms - (slope * ys + intercept)
        mad = float(np.median(np.abs(resid - np.median(resid)))) or 1e-9
        keep = np.abs(resid - np.median(resid)) <= 2.5 * mad
        if keep.sum() >= self.min_samples and keep.sum() < n:
            slope, intercept = np.polyfit(ys[keep], ppms[keep], 1, w=ws[keep])
        self._slope, self._intercept = float(slope), float(intercept)
        self._fitted = True
        return True

    def ppm_at(self, y: float) -> float | None:
        """Verilen görüntü-y'sinde ppm. Uydurma yoksa veya tahmin fiziksel değilse
        (≤0) medyan ppm'e düşer."""
        if not self._fitted:
            return None
        val = self._slope * float(y) + self._intercept
        if not np.isfinite(val) or val <= 1e-6:
            return self._median_ppm if self._median_ppm > 0 else None
        return val


class MetricSpeedEstimator:
    """Sahneden öğrenilen ppm(y) ile metrik km/h üretir (gercek_hiz_plani.md Aşama 1-3).

    Pipeline örneği başına bir adet; kareler boyunca araç/plaka ölçümlerini biriktirir
    (ısınma), yeterince ölçüm olunca ppm(y)'yi uydurur ve track'in yer-temas noktasının
    iki kare arası metrik yer değiştirmesinden hız hesaplar. Isınma bitene dek estimate()
    (None, False) döndürür → çağıran metrik iddiada bulunmaz.
    """

    def __init__(self, settings):
        self.s = settings
        self.scale = ScaleField(min_samples=getattr(settings, "calib_min_samples", 6))
        # CL: maybe_fit() araç-başı/kare-başı çağrılıyor; n_samples>=min iken HER çağrıda
        # tüm birikmiş örnekler (≤4000) üzerinde np.polyfit koşturmak kare-başı O(N)
        # gereksiz iş demekti (aynı doğru tekrar tekrar uydurulur). Yeniden-uydurma artık
        # yalnız EN AZ `refit_every` yeni örnek eklendiğinde yapılır; ilk min_samples
        # eşiğini geçişte daima uydurulur (is_ready davranışı korunur). DAVRANIŞ-KORUYAN:
        # ppm(y) ölçek-alanı düzgün biriken örneklerle yumuşak değişir → seyrek refit
        # kalibrasyon sonucunu pratikte değiştirmez, yalnız maliyeti düşürür.
        self._refit_every = max(1, int(getattr(settings, "calib_refit_every", 25)))
        self._last_fit_n = 0
        self._kalman: dict = {}
        self._ema: dict = {}  # track_id -> EMA durumu (Kalman sonrası ek yumuşatma)
        # Aşama 4 — şerit homografisi (kurulursa ppm(y)'ye göre öncelikli ölçek kaynağı).
        # roadguard'da henüz üretici yok; verilirse (to_ground + is_valid) kullanılır.
        self.homography = None

    def set_homography(self, homography) -> None:
        """Yer düzlemi homografisini ata (§7.1: B kaynağı A'dan önceliklidir)."""
        if homography is not None and not getattr(homography, "is_valid", False):
            return
        self.homography = homography

    def observe_plate(self, plate_bbox: BBox | None) -> None:
        """Bir karede görülen plakadan yerel ppm örneği topla (varsa en kesin kaynak)."""
        ppm = plate_ppm(
            plate_bbox,
            getattr(self.s, "plate_width_m", 0.520),
            getattr(self.s, "plate_aspect_tolerance", 0.35),
        )
        if ppm is not None and plate_bbox is not None:
            self.scale.add(plate_bbox.y2, ppm, weight=1.0)

    def observe_vehicle(self, vehicle_bbox: BBox | None, vtype: str | None) -> None:
        """Aşama 2 — araç bbox genişliğinden sınıf-bazlı ppm yedeği (§4.2).

        Tekil araç genişliği ±%15-20 oynar ⇒ DÜŞÜK ağırlık. Çok araçtan istatistiksel
        olarak (ScaleField'in ağırlıklı + aykırı-dayanıklı regresyonu) sağlamlaşır.
        """
        if vehicle_bbox is None:
            return
        widths = getattr(self.s, "vehicle_width_m", {}) or {}
        typ_w = widths.get(vtype) or widths.get("car") or 1.80
        px_w = vehicle_bbox.x2 - vehicle_bbox.x1
        if px_w <= 1 or typ_w <= 0:
            return
        self.scale.add(
            vehicle_bbox.y2, px_w / typ_w, weight=getattr(self.s, "vehicle_ppm_weight", 0.25)
        )

    def maybe_fit(self) -> None:
        """Yeterli ölçüm biriktiyse ppm(y)'yi (yeniden) uydur.

        CL: kare-başı O(N) polyfit yerine seyrek refit. İlk kez min_samples eşiği
        aşılınca daima uydurulur (is_ready True olur); sonrasında yalnız son
        uydurmadan bu yana `refit_every` yeni örnek eklendiyse yeniden uydurulur.
        """
        n = self.scale.n_samples
        if n < self.scale.min_samples:
            return
        if self.scale.is_ready and (n - self._last_fit_n) < self._refit_every:
            return
        if self.scale.fit():
            self._last_fit_n = n

    def _step_meters(self, f0, f1) -> float | None:
        """İki yer-temas noktası arası metrik yer değiştirme (m).

        Füzyon önceliği (§7.1): homografi varsa noktalar metrik yer düzlemine
        izdüşürülüp Öklid mesafe alınır (RİGOROUS, boylamsal-doğru); yoksa yerel
        ppm(y) ortalamasıyla çevrilir.

        DÜRÜSTLÜK NOTU (perspektif foreshortening). ppm(y) YATAY genişliklerden
        türetilmiş px/m'dir; fallback yol (homografi YOK) dikey/derinlik ekseni
        piksel hareketini de bu yatay ölçekle metreye çevirir → kameraya doğru/
        uzaklaşan (boylamsal) harekette SİSTEMATİK yanlılık (kamera eğimine bağlı).
        Boylamsal ölçek yalnız homografi/IPM ile TANIMLIDIR — tek başına yatay
        genişliklerden geri kazanılamaz (bilinmeyen kamera yüksekliği/odak sabiti).
        Rigorous yol: optional/homography_ipm.py + speed.mode=ipm. Bu yüzden
        fallback mutlak hız bir TAHMİNDİR; sistem mutlak hızı homografi olmadan
        kanıt olarak sunmaz (FTR §4.7: mutlak hız nicel sınanmamıştır; nicel
        sınanan yetenek kalibrasyonsuz swerving'dir).
        """
        (x0, y0), (x1, y1) = f0, f1
        if self.homography is not None:
            g0 = self.homography.to_ground(x0, y0)
            g1 = self.homography.to_ground(x1, y1)
            if g0 is None or g1 is None:
                return None
            return ((g1[0] - g0[0]) ** 2 + (g1[1] - g0[1]) ** 2) ** 0.5
        ppm0 = self.scale.ppm_at(y0)
        ppm1 = self.scale.ppm_at(y1)
        if not ppm0 or not ppm1:
            return None
        ppm = 0.5 * (ppm0 + ppm1)
        return (((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5) / ppm

    def _window_steps(self, track: SpeedTrack):
        """Pencere içindeki ardışık kare çiftleri için (dt, anlık_hız_mps) listesi."""
        n = len(track.foot_history)
        window = max(1, getattr(self.s, "speed_window_frames", 6))
        start = max(1, n - window)
        # Yalnız gereken kuyruğu (start-1 .. n) islice ile al — 16-derin geçmişi her
        # kare iki kez list()'e kopyalama; kare-başı tahsisi azaltır (K-004 davranış aynı).
        foots = list(islice(track.foot_history, start - 1, n))
        tss = list(islice(track.ts_history, start - 1, n))
        steps = []
        for j in range(1, len(foots)):
            t0, t1 = tss[j - 1], tss[j]
            if t0 is None or t1 is None:
                continue
            dt = t1 - t0
            if dt <= 0:
                continue
            meters = self._step_meters(foots[j - 1], foots[j])
            if meters is None:
                continue
            steps.append((dt, meters / dt))
        return steps

    def estimate(self, track: SpeedTrack | None) -> tuple[float | None, bool]:
        """(km/h, is_calibrated) döndür. Ölçek hazır değilse (None, False).

        Aşama 3: pencere üzerinden medyan hız + fiziksel-olmayan ivme reddi (medyandan
        `speed_max_accel_mps2·Δt`'den fazla sapan adımlar atılır) + track-başı Kalman.
        """
        if track is None:
            return None, False
        if self.homography is None and not self.scale.is_ready:
            return None, False
        steps = self._window_steps(track)
        if not steps:
            return None, False

        vs = np.array([v for _, v in steps], dtype=float)
        med = float(np.median(vs))
        accel = getattr(self.s, "speed_max_accel_mps2", 8.0)
        kept = [v for (dt, v) in steps if abs(v - med) <= accel * max(dt, 1e-3)]
        v_robust = float(np.median(kept)) if kept else med

        q = getattr(self.s, "speed_kalman_q", 3.0)
        r = getattr(self.s, "speed_kalman_r", 8.0)
        tid = track.track_id
        if tid not in self._kalman:
            self._kalman[tid] = KalmanSpeed1D(Q=q, R=r)
        v_smooth = self._kalman[tid].update(v_robust)

        # Kalman üstü EMA (ek yumuşatma): alpha küçük → daha düz (geçmişe ağırlık).
        # Kalman ani sıçramaları, EMA kalan kare-kare titremeyi bastırır.
        alpha = float(getattr(self.s, "speed_ema_alpha", 0.0) or 0.0)
        if 0.0 < alpha < 1.0:
            prev = self._ema.get(tid)
            v_smooth = v_smooth if prev is None else alpha * v_smooth + (1.0 - alpha) * prev
            self._ema[tid] = v_smooth

        max_kmh = getattr(self.s, "speed_metric_max_kmh", 200.0)
        return round(max(0.0, min(v_smooth * 3.6, max_kmh)), 1), True

    def prune(self, active_ids) -> None:
        """Tracker'da artık olmayan track'lerin Kalman/EMA durumunu temizle (bellek)."""
        for tid in [t for t in self._kalman if t not in active_ids]:
            del self._kalman[tid]
        for tid in [t for t in self._ema if t not in active_ids]:
            del self._ema[tid]
