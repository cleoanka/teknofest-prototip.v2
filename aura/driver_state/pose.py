"""Pose-tabanlı sürücü davranış sınıflandırıcı — YOLO26-pose keypoint geometrisi.

Neden var?
    Telefon/sigara gibi davranışlar için fine-tune edilmiş bir detection ağırlığı
    yokken stok COCO modeli bu sınıfları ÜRETEMEZ (sessiz sıfır). v1 prototip aynı
    problemi MediaPipe el/yüz geometrisiyle çözmüş ve gerçek videolarda ölçmüştü
    (sigara recall %59, telefon %61, FP %0). MediaPipe hem Python 3.13'te yok hem
    de AURA mimari kararı landmark kütüphanelerini yasaklıyor — bu modül aynı
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

from aura.config import resolve_repo_path
from aura.device import resolve_device
from aura.driver_state.classifier import DriverClassifier
from aura.schema import DriverState

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("aura.driver_state.pose")

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

        path = resolve_repo_path(
            cfg.get("models.driver_state.pose_path", "weights/yolo26s-pose.pt")
        )
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
        import cv2
        import numpy as np  # noqa: F811 - runtime import (TYPE_CHECKING bloğu lazy)

        h, w = roi.shape[:2]
        short = min(h, w)
        if short <= 0:
            return roi
        scale = min(self.roi_max_upscale, max(1.0, self.roi_min_side / short))
        if scale > 1.01:
            roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        if self.roi_enhance:
            # LAB-L kanalında CLAHE + hafif gamma: cam yansıması/karanlık kabini açar
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            lab_l, lab_a, lab_b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            lab_l = clahe.apply(lab_l)
            roi = cv2.cvtColor(cv2.merge((lab_l, lab_a, lab_b)), cv2.COLOR_LAB2BGR)
            inv_gamma = 1.0 / 1.6
            table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype("uint8")
            roi = cv2.LUT(roi, table)
        return roi

    # --- ROI nesne kanıtı ----------------------------------------------------- #
    def _object_evidence(self, roi: np.ndarray, ds: DriverState) -> None:
        """ROI'de phone/smoking NESNESİ ara; bulursa bayrağı doğrudan set et."""
        from aura.taxonomy import canonical

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

    # --- ana giriş ----------------------------------------------------------- #
    def infer(self, cabin_roi: np.ndarray | None, track_id: int | None = None) -> DriverState:
        ds = DriverState()
        if cabin_roi is None or cabin_roi.size == 0:
            return ds
        roi = self._prep_roi(cabin_roi)
        geo = self._geometry(roi)
        if self.obj_model is not None:
            self._object_evidence(roi, ds)
        # Bastırma latch'i: telefon nesnesi BU karede görüldüyse zamanlayıcıyı doldur;
        # zamanlayıcı aktifken geometrik 'sigara' bastırılır (ağız önündeki el
        # telefondur) — ama telefon bayrağı İLERİ TAŞINMAZ (FP amplifikasyonu yok).
        key = -1 if track_id is None else track_id
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

        # ROI'deki en belirgin kişi = sürücü adayı (ROI zaten sürücü kutusundan kesik)
        best_i = max(range(len(boxes)), key=lambda i: float(boxes[i].conf.item()))
        person_conf = float(boxes[best_i].conf.item())
        xy = kps.xy[best_i].tolist()
        kc = kps.conf[best_i].tolist() if kps.conf is not None else [1.0] * len(xy)

        def pt(i: int) -> tuple[float, float] | None:
            if i < len(xy) and kc[i] >= self.kp_conf:
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
