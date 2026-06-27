"""Pose-tabanlı sürücü davranış sınıflandırıcı — YOLO26-pose keypoint geometrisi.

Neden var?
    Telefon/sigara gibi davranışlar için fine-tune edilmiş bir detection ağırlığı
    yokken stok COCO modeli bu sınıfları ÜRETEMEZ (sessiz sıfır). v1 prototip aynı
    problemi MediaPipe el/yüz geometrisiyle çözmüş ve gerçek videolarda ölçmüştü
    (sigara recall %59, telefon %61, FP %0). MediaPipe hem Python 3.13'te yok hem
    de RoadGuard mimari kararı landmark kütüphanelerini yasaklıyor — bu modül aynı
    KANITLANMIŞ geometriyi ultralytics YOLO26-pose (COCO 17 keypoint) ile uygular:
    mimari karar korunur (saf YOLO26), bağımlılık eklenmez.

Geometri (v1 K-012 dersi — mutlak eşik değil GÖRELİ yakınlık):
    - bilek↔kulak mesafesi < phone_ear_ratio × yüz-genişliği VE bilek kulağa
      ağızdan daha yakın → TELEFON adayı
    - bilek↔ağız mesafesi < smoke_mouth_ratio × yüz-genişliği VE bilek ağıza
      kulaktan daha yakın → SİGARA adayı
    Tüm eşikler yüz genişliği biriminde (ölçek/çözünürlük bağımsız, K-004).

Zamansal teyit bu modülde DEĞİL: pipeline'daki 16/8 kararlılık süzgeci tek-kare
yanlış pozitifleri zaten eler (v1 sustain penceresinin v2 karşılığı).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from roadguard.config import resolve_repo_path
from roadguard.device import resolve_device
from roadguard.driver_state.classifier import DriverClassifier
from roadguard.schema import DriverState

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("roadguard.driver_state.pose")

# COCO-17 keypoint indeksleri (ultralytics pose çıktı sırası)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PoseDriverClassifier(DriverClassifier):
    """Sürücü ROI'sinde YOLO26-pose koşar, el-yüz geometrisinden bayrak üretir."""

    def __init__(self, cfg):
        from ultralytics import YOLO

        # Varsayılan yolo26l-pose (kullanıcı kararı: minimum alana büyük model);
        # yapılandırılan ağırlık diskte yoksa stok s-pose'a LOGLU fallback.
        path = resolve_repo_path(
            cfg.get("models.driver_state.pose_path", "weights/yolo26l-pose.pt")
        )
        if not path.exists():
            fallback = resolve_repo_path("weights/yolo26s-pose.pt")
            if fallback.exists():
                log.warning("Pose ağırlığı yok (%s) → stok %s kullanılıyor", path, fallback)
                path = fallback
        self.model = YOLO(str(path))
        self.conf = float(cfg.get("models.driver_state.pose_conf", 0.25))
        self.kp_conf = float(cfg.get("models.driver_state.pose_kp_conf", 0.30))
        self.imgsz = int(cfg.get("models.driver_state.pose_imgsz", 640))
        # v1 K-012 ölçümü: gerçek telefonda el kulağa ÇOK yakındır (d_ear %99 < 0.40×fw);
        # 0.55 gibi gevşek eşik sigara jestini telefon sanıyordu (video_1 FP'si).
        self.phone_ear_ratio = float(cfg.get("models.driver_state.phone_ear_ratio", 0.40))
        self.smoke_mouth_ratio = float(cfg.get("models.driver_state.smoke_mouth_ratio", 0.60))
        # ROI ön-işleme (v1 dersi: cam arkası sürücü küçük/karanlık — büyüt + parlat
        # → keypoint bulunabilirliği %2-5'ten %27-55'e çıkmıştı)
        self.roi_min_side = int(cfg.get("models.driver_state.roi_min_side", 320))
        self.roi_max_upscale = float(cfg.get("models.driver_state.roi_max_upscale", 4.0))
        self.roi_enhance = bool(cfg.get("models.driver_state.roi_enhance", True))
        self.device = resolve_device(cfg.get("runtime.device", "auto"))
        # ROI parlatma yardımcıları SABİT — kare başına yeniden kurmak boşuna alloc'tu.
        # CLAHE nesnesi + gamma LUT tablosu ilk kullanımda bir kez kurulup saklanır
        # (cv2/np runtime-import olduğu için __init__'te değil _ensure_enhance'te lazy).
        self._clahe = None  # cv2.CLAHE — ilk _prep_roi_scaled'da kurulur
        self._gamma_lut = None  # np.ndarray(256,uint8) — sabit gamma=1.6 tablosu
        # --- sürücü-içi sıkı kırpma ------------------------------------------ #
        # Modele giden alan MINIMUM olmalı (kullanıcı kararı): gelen ROI (kabin
        # fallback'inde araç kutusunun üst %55'i — ön cam + yolcu yansımaları)
        # önce SÜRÜCÜNÜN kişi kutusuna (+pad_ratio) daraltılır; pose ve v4 nesne
        # kanıtı yalnız bu dar kırpıkta koşar. Kutu track başına önbelleğe alınır
        # (normalize koordinat; sürücü araç içinde sabit oturur) ve redetect_every
        # karede bir tazelenir → kare başına TEK pose geçişi korunur.
        dc = cfg.get("models.driver_state.driver_crop", {}) or {}
        self.crop_enabled = bool(dc.get("enabled", True))
        self.crop_pad = float(dc.get("pad_ratio", 0.10))
        self.crop_redetect = int(dc.get("redetect_every", 15))
        # ROI zaten dar ise (ör. DriverLock kişi kutusundan kesilmiş) kırpma katma
        # değersizdir: alan kazancı bu çarpanın altındaysa ROI olduğu gibi kullanılır.
        self.crop_min_gain = float(dc.get("min_gain", 1.25))
        # Sürücü tarafı seçimi DriverLock ile aynı sözleşmeyi kullanır (vars. sağ-alt).
        corner = str(cfg.get("driver_lock.corner", "bottom_right")).lower()
        self.corner_target = {
            "bottom_right": (1.0, 1.0),
            "bottom_left": (0.0, 1.0),
            "top_right": (1.0, 0.0),
            "top_left": (0.0, 0.0),
        }.get(corner, (1.0, 1.0))
        self._crop_cache: dict[int, list] = {}  # track_id -> [normalize kutu, yaş]
        self.last_crop_box: tuple[int, int, int, int] | None = None  # teşhis/görselleştirici
        self._last_person_seen = False  # geometri geçişinde kişi kutusu görüldü mü
        # --- ROI nesne kanıtı (hibrit) -------------------------------------- #
        # Geometri tek başına yetmez: telefon kulağa değil AĞZIN ÖNÜNE tutulursa
        # (hoparlör) el-ağız yakınlığı sigara gibi görünür; telefon tutan bilek
        # keypoint'i de sıkça düşük güvenli kalır (gerçek video_2 ölçümü). Çözüm:
        # fine-tune dedektör (v4, 'phone' sınıfı) sürücü ROI'sinde ayrıca koşulur;
        # NESNE kanıtı geometrik çıkarımdan üstündür (phone nesnesi varken
        # geometrik 'smoking' bastırılır).
        ro = cfg.get("models.driver_state.roi_objects", {}) or {}
        self.obj_enabled = bool(ro.get("enabled", True))
        self.obj_conf = float(ro.get("conf", 0.25))
        self.obj_imgsz = int(ro.get("imgsz", 640))
        # BASTIRMA latch'i (assert ETMEYEN): telefon nesnesi yakın geçmişte
        # görüldüyse geometrik 'sigara' çıkarımı bu süre boyunca bastırılır —
        # ama telefon BAYRAĞI yalnızca gerçek kanıt karelerinde (nesne o karede ||
        # geometri) üretilir; 16/8 süzgeci sıklık ayrımını doğal yapar.
        # Gerçek-video dersi (iki regresyon turu): latch telefon İDDİASI taşırsa
        # seyrek/orta sıklıktaki nesne FP'leri (v4, sigara tutan eli telefon
        # sanabiliyor) amplifiye olup gerçek sigarayı eziyor; yalnız BASTIRMA
        # taşırsa video_2'de (gerçek arama, sık isabet) sigara FP'si sıfırlanırken
        # video_1'de (sigara, seyrek FP) tespit korunur.
        self.obj_suppress_frames = int(ro.get("suppress_frames", 25))
        # Bastırma için AYRI (daha yüksek) güven eşiği: bayraklama duyarlı kalır
        # (obj_conf), sigara-bastırma yalnız güçlü telefon kanıtıyla tetiklenir.
        # Gerçek ölçüm: FP'ler ~0.22-0.25 bandında, gerçek telefon >= 0.34.
        self.obj_suppress_conf = float(ro.get("suppress_conf", 0.30))
        self._smoke_suppress: dict[int, int] = {}  # track_id -> kalan bastırma karesi
        self.obj_model = None
        if self.obj_enabled:
            obj_path = resolve_repo_path(
                ro.get("path") or cfg.get("models.detector.path", "weights/yolo26s.pt")
            )
            if obj_path.exists():
                self.obj_model = YOLO(str(obj_path))
                log.info("Pose hibrit ROI nesne kanıtı: %s (conf=%.2f)", obj_path, self.obj_conf)
            else:
                log.warning("ROI nesne modeli yok (%s) — yalnız geometri kullanılacak", obj_path)
        # --- OPSİYONEL İKİNCİ MODEL: özel eğitimli sigara dedektörü ----------- #
        # roi_objects (stok phone + sigara-bastırma latch'i) AYNEN korunur; bu model
        # onun YANINDA (replace DEĞİL) sürücü ROI'sinde yalnız 'smoking' NESNESİ arar.
        # Bulgusu mevcut 'smoking' kanıtına OR'lanır (16/8 oylamasından geçer) → sigara
        # GÜÇLENİR, phone yolu (nesne-kanıtı + bastırma) hiç değişmez. K-004 A/B dersi:
        # custom_smoking'i roi_objects'e DROP-IN koymak phone kanıtını siliyordu;
        # ayrı ikinci-model kanalı regresyonu önler.
        # GRACEFUL-ABSENT: ağırlık diskte yoksa (CI/takım — gitignore) yüklenmez,
        # davranış DEĞİŞMEZ (no-op). Loglanır.
        sm = cfg.get("models.driver_state.smoking_model", {}) or {}
        self.smoking_enabled = bool(sm.get("enabled", False))
        self.smoking_conf = float(sm.get("conf", 0.25))
        self.smoking_imgsz = int(sm.get("imgsz", 640))
        self.smoking_model = None
        if self.smoking_enabled:
            sm_path = resolve_repo_path(sm.get("path", "weights/custom_smoking.pt"))
            if sm_path.exists():
                self.smoking_model = YOLO(str(sm_path))
                log.info(
                    "Pose ikinci-model sigara kanıtı: %s (conf=%.2f) — phone yolu korunur",
                    sm_path,
                    self.smoking_conf,
                )
            else:
                log.warning("Sigara modeli yok (%s) — graceful no-op, davranış değişmez", sm_path)
        log.info(
            "Pose driver-state yüklendi: %s (imgsz=%d, device=%s, kulak=%.2f ağız=%.2f)",
            path,
            self.imgsz,
            self.device,
            self.phone_ear_ratio,
            self.smoke_mouth_ratio,
        )

    # --- ROI ön-işleme ------------------------------------------------------ #
    def _prep_roi(self, roi: np.ndarray) -> np.ndarray:
        return self._prep_roi_scaled(roi)[0]

    def _prep_roi_scaled(self, roi: np.ndarray) -> tuple[np.ndarray, float]:
        """ROI'yi büyüt + parlat; uygulanan ölçeği de döndür (koordinat geri-eşleme)."""
        import cv2
        import numpy as np  # noqa: F811 - runtime import (TYPE_CHECKING bloğu lazy)

        h, w = roi.shape[:2]
        short = min(h, w)
        if short <= 0:
            return roi, 1.0
        scale = min(self.roi_max_upscale, max(1.0, self.roi_min_side / short))
        if scale > 1.01:
            roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        if self.roi_enhance:
            # LAB-L kanalında CLAHE + hafif gamma: cam yansıması/karanlık kabini açar.
            # CLAHE nesnesi ve gamma LUT SABİT → ilk çağrıda kur, sonra yeniden kullan
            # (davranış birebir aynı; yalnız kare başına alloc/pow zinciri elenir).
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            if self._gamma_lut is None:
                inv_gamma = 1.0 / 1.6
                self._gamma_lut = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype("uint8")
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            lab_l, lab_a, lab_b = cv2.split(lab)
            lab_l = self._clahe.apply(lab_l)
            roi = cv2.cvtColor(cv2.merge((lab_l, lab_a, lab_b)), cv2.COLOR_LAB2BGR)
            roi = cv2.LUT(roi, self._gamma_lut)
        return roi, scale

    # --- sürücü-içi sıkı kırpma ---------------------------------------------- #
    def _locate_driver(self, roi: np.ndarray) -> tuple[int, int, int, int] | None:
        """ROI'de sürücünün kişi kutusunu bul (ham ROI koordinatlarında).

        Pose modeli ön-işlenmiş ROI'de koşulur; birden çok kişi varsa (yolcu,
        cam yansıması) DriverLock ile aynı sözleşmeyle SÜRÜCÜ KÖŞESİNE en yakın
        kutu seçilir — en yüksek conf değil (yansıma/yolcu daha 'net' olabilir).
        """
        prepped, scale = self._prep_roi_scaled(roi)
        results = self.model.predict(
            prepped, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )
        if not results:
            return None
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        ph, pw = prepped.shape[:2]
        tx, ty = self.corner_target

        def corner_score(b) -> tuple[float, float]:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            nx = (x1 + x2) / 2.0 / max(pw, 1)
            ny = (y1 + y2) / 2.0 / max(ph, 1)
            return (-((nx - tx) ** 2 + (ny - ty) ** 2), float(b.conf.item()))

        best = max(boxes, key=corner_score)
        x1, y1, x2, y2 = (v / scale for v in best.xyxy[0].tolist())
        return int(x1), int(y1), int(x2), int(y2)

    def _driver_crop(
        self, roi: np.ndarray, key: int
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """ROI'yi sürücü kutusuna (+crop_pad) daralt; (kırpık, kutu|None) döndür.

        Önbellek normalize koordinat tutar: araç kutusu kareler arasında
        büyüyüp küçülse de sürücünün araç-içi GÖRELİ konumu sabittir.
        """
        h, w = roi.shape[:2]
        if h <= 0 or w <= 0:
            return roi, None
        cached = self._crop_cache.get(key)
        if cached is not None and cached[1] < self.crop_redetect:
            cached[1] += 1
            nx1, ny1, nx2, ny2 = cached[0]
            x1, y1 = max(0, int(nx1 * w)), max(0, int(ny1 * h))
            x2, y2 = min(w, int(nx2 * w)), min(h, int(ny2 * h))
            crop = roi[y1:y2, x1:x2]
            if crop.size:
                return crop, (x1, y1, x2, y2)
        self._crop_cache.pop(key, None)

        box = self._locate_driver(roi)
        if box is None:
            return roi, None  # kişi yok → dürüstçe tüm ROI (geometri zaten çekimser kalır)
        bx1, by1, bx2, by2 = box
        pad_x = (bx2 - bx1) * self.crop_pad
        pad_y = (by2 - by1) * self.crop_pad
        x1, y1 = max(0, int(bx1 - pad_x)), max(0, int(by1 - pad_y))
        x2, y2 = min(w, int(bx2 + pad_x)), min(h, int(by2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            return roi, None
        # Kazanç kontrolü: ROI zaten dar (DriverLock kişi kutusu) ise kırpma anlamsız.
        if (w * h) / max((x2 - x1) * (y2 - y1), 1) < self.crop_min_gain:
            return roi, None
        crop = roi[y1:y2, x1:x2]
        if not crop.size:
            return roi, None
        self._crop_cache[key] = [(x1 / w, y1 / h, x2 / w, y2 / h), 0]
        return crop, (x1, y1, x2, y2)

    # --- ROI nesne kanıtı ----------------------------------------------------- #
    def _object_evidence(self, roi: np.ndarray, ds: DriverState) -> None:
        """ROI'de phone/smoking NESNESİ ara; bulursa bayrağı doğrudan set et."""
        from roadguard.taxonomy import canonical

        results = self.obj_model.predict(
            roi, conf=self.obj_conf, imgsz=self.obj_imgsz, device=self.device, verbose=False
        )
        if not results:
            return
        r = results[0]
        names = getattr(r, "names", None) or self.obj_model.names
        for b in getattr(r, "boxes", None) or []:
            idx = int(b.cls.item())
            name = canonical(
                names[idx] if isinstance(names, (list, tuple)) else names.get(idx, str(idx))
            )
            if name in ("phone", "smoking") and hasattr(ds, name):
                setattr(ds, name, True)
                ds.confidence[name] = max(ds.confidence.get(name, 0.0), float(b.conf.item()))

    def _smoking_object_evidence(self, roi: np.ndarray, ds: DriverState) -> None:
        """Özel eğitimli ikinci modelle ROI'de yalnız 'smoking' NESNESİ ara.

        roi_objects'ten BAĞIMSIZ ek kanal: bulgu mevcut 'smoking' bayrağına OR'lanır,
        conf max-birleştirilir. 'phone' veya başka sınıf ÜRETMEZ (phone yolu yalnız
        roi_objects'e ait kalır → bastırma latch'i karışmaz). Tek-sınıf eğitilmemiş
        bir model 'smoking' dışında sınıf verse bile yok sayılır.
        """
        from roadguard.taxonomy import canonical

        results = self.smoking_model.predict(
            roi, conf=self.smoking_conf, imgsz=self.smoking_imgsz, device=self.device, verbose=False
        )
        if not results:
            return
        r = results[0]
        names = getattr(r, "names", None) or self.smoking_model.names
        for b in getattr(r, "boxes", None) or []:
            idx = int(b.cls.item())
            name = canonical(
                names[idx] if isinstance(names, (list, tuple)) else names.get(idx, str(idx))
            )
            if name == "smoking":
                ds.smoking = True
                ds.confidence["smoking"] = max(
                    ds.confidence.get("smoking", 0.0), float(b.conf.item())
                )

    def forget(self, track_id: int | None) -> None:
        """ID'ye bağlı pose durumunu (sürücü-kırpık önbelleği + sigara-bastırma latch'i)
        temizle. Track ID recycle olunca yeni araç eski kırpık/latch ile işlenmesin ve
        uzun akışta bu sözlükler sınırsız büyümesin (engine.prune/forget buradan çağırır)."""
        key = -1 if track_id is None else track_id
        self._crop_cache.pop(key, None)
        self._smoke_suppress.pop(key, None)

    # --- ana giriş ----------------------------------------------------------- #
    def infer(self, cabin_roi: np.ndarray | None, track_id: int | None = None) -> DriverState:
        ds = DriverState()
        if cabin_roi is None or cabin_roi.size == 0:
            return ds
        key = -1 if track_id is None else track_id
        # 1) Sürücü-içi sıkı kırpma: pose + nesne kanıtı yalnız sürücü kutusunda koşar
        #    (tüm kabin gereksiz — ön cam/yolcu yansımaları FP kaynağıydı).
        if self.crop_enabled:
            roi, crop_box = self._driver_crop(cabin_roi, key)
        else:
            roi, crop_box = cabin_roi, None
        self.last_crop_box = crop_box
        roi = self._prep_roi(roi)
        geo = self._geometry(roi)
        # Önbellek hijyeni: kırpıkta artık kişi görünmüyorsa kutu bayatlamıştır
        # (sürücü kaydı/araç döndü) → düşür, sonraki karede yeniden tespit edilir.
        if crop_box is not None and not self._last_person_seen:
            self._crop_cache.pop(key, None)
        if self.obj_model is not None:
            self._object_evidence(roi, ds)
        # İkinci-model sigara kanıtı: roi_objects'in YANINDA koşar, 'smoking'
        # bayrağına OR'lanır. phone yolunu (yukarıdaki nesne-kanıtı + aşağıdaki
        # bastırma latch'i) HİÇ etkilemez. Ağırlık yoksa smoking_model None → no-op.
        if self.smoking_model is not None:
            self._smoking_object_evidence(roi, ds)
        # Bastırma latch'i: telefon nesnesi BU karede görüldüyse zamanlayıcıyı doldur;
        # zamanlayıcı aktifken geometrik 'sigara' bastırılır (ağız önündeki el
        # telefondur) — ama telefon bayrağı İLERİ TAŞINMAZ (FP amplifikasyonu yok).
        if ds.phone and ds.confidence.get("phone", 0.0) >= self.obj_suppress_conf:
            self._smoke_suppress[key] = self.obj_suppress_frames
        sup = self._smoke_suppress.get(key, 0)
        if sup > 0:
            self._smoke_suppress[key] = sup - 1
            geo.smoking = False
        ds.phone = ds.phone or geo.phone
        ds.smoking = ds.smoking or geo.smoking
        for k, v in geo.confidence.items():
            ds.confidence[k] = max(ds.confidence.get(k, 0.0), v)
        return ds

    def _geometry(self, roi: np.ndarray) -> DriverState:
        """Pose keypoint geometrisinden telefon/sigara çıkarımı (v1 K-012 portu)."""
        ds = DriverState()
        self._last_person_seen = False
        results = self.model.predict(
            roi, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )
        if not results:
            return ds
        r = results[0]
        kps = getattr(r, "keypoints", None)
        boxes = getattr(r, "boxes", None)
        if kps is None or boxes is None or len(boxes) == 0:
            return ds
        self._last_person_seen = True

        # ROI'deki en belirgin kişi = sürücü adayı (ROI zaten sürücü kutusundan kesik).
        # conf'ları tek seferde listeye çek (her eleman için tekrarlı tensor.item()
        # köprüsünden kaçın — davranış aynı, yalnız tensor erişimi azalır).
        confs = [float(b.conf.item()) for b in boxes]
        best_i = max(range(len(confs)), key=confs.__getitem__)
        person_conf = confs[best_i]
        xy = kps.xy[best_i].tolist()
        kc = kps.conf[best_i].tolist() if kps.conf is not None else [1.0] * len(xy)

        def pt(i: int) -> tuple[float, float] | None:
            # Sınır kontrolü HEM xy HEM kc uzunluğuna bakar: bozuk/kısmi pose
            # çıktısında kps.conf, xy'den KISA dönebilir → kc[i] IndexError olurdu.
            if i < len(xy) and i < len(kc) and kc[i] >= self.kp_conf:
                return (float(xy[i][0]), float(xy[i][1]))
            return None

        nose = pt(NOSE)
        ears = [p for p in (pt(L_EAR), pt(R_EAR)) if p is not None]
        wrists = [p for p in (pt(L_WRIST), pt(R_WRIST)) if p is not None]

        # KARAR İÇİN KULAK ŞART: telefon/sigara ayrımı "bilek kulağa mı ağza mı
        # daha yakın" GÖRELİ kıyasına dayanır; kulak görünmüyorsa d_ear sonsuz olur
        # ve elinde telefon olan sürücü bile 'sigara' sayılırdı (video_2 FP dersi).
        # Kulak yoksa dürüst çekimserlik: iddia üretme (16/8 süzgeci boşluğu tolere eder).
        if not ears or nose is None or not wrists:
            return ds
        # Yüz genişliği (ölçek birimi): iki kulak arası; tek kulaksa 2×(kulak-burun)
        fw = _dist(ears[0], ears[1]) if len(ears) == 2 else 2.0 * _dist(nose, ears[0])
        if fw < 2.0:
            return ds  # geometri kurulamıyor → iddia yok (uydurma yok)

        # Ağız vekili: burnun fw×0.30 altı (profilde de makul kalır)
        mouth = (nose[0], nose[1] + 0.30 * fw)

        for wrist in wrists:
            d_mouth = _dist(wrist, mouth)
            d_ear = min(_dist(wrist, e) for e in ears)
            # TELEFON: el kulağa ÇOK yakın VE kulağa ağızdan daha yakın (göreli kıyas)
            if d_ear < self.phone_ear_ratio * fw and d_ear < d_mouth:
                ds.phone = True
                score = person_conf * max(0.0, 1.0 - d_ear / (self.phone_ear_ratio * fw))
                ds.confidence["phone"] = max(ds.confidence.get("phone", 0.0), round(score, 3))
            # SİGARA: el ağza yakın VE ağza kulaktan daha yakın (v1: kıyas FP'yi keser)
            elif d_mouth < self.smoke_mouth_ratio * fw and d_mouth < d_ear:
                ds.smoking = True
                score = person_conf * max(0.0, 1.0 - d_mouth / (self.smoke_mouth_ratio * fw))
                ds.confidence["smoking"] = max(ds.confidence.get("smoking", 0.0), round(score, 3))
        return ds
