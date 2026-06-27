"""Gerçek Stage-1 dedektör — ultralytics YOLO26 + ByteTrack.

`ai_mode=real` (veya `auto` + ağırlık mevcut) iken kullanılır. ByteTrack tracking
mode ultralytics'e dahildir (`tracker="bytetrack.yaml"`). Yalnızca config'teki
araç sınıfları geçirilir; her tespit için ROI crop'lar üretilir.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from roadguard.config import resolve_repo_path
from roadguard.detection.detector import Detection, Detector, Person, Sign, crop_rois
from roadguard.device import resolve_device
from roadguard.schema import BBox
from roadguard.taxonomy import canonical

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("roadguard.detection.yolo")


class YOLO26Detector(Detector):
    def __init__(self, cfg):
        super().__init__()  # last_persons/last_signs/last_aux'u örnek-seviyesinde kur
        from ultralytics import YOLO

        # Yol repo köküne göre çözülür (CWD-bağımsız); yapılandırılan ağırlık yoksa
        # stok yolo26s.pt'ye LOGLU fallback (sessiz mock düşüşü yerine).
        configured = resolve_repo_path(cfg.get("models.detector.path", "weights/yolo26s.pt"))
        if not configured.exists():
            fallback = resolve_repo_path("weights/yolo26s.pt")
            if fallback.exists():
                log.warning(
                    "Detector ağırlığı yok (%s) → stok %s kullanılıyor", configured, fallback
                )
                configured = fallback
        self.path = str(configured)
        self.model = YOLO(self.path)
        self.conf = float(cfg.get("models.detector.conf", 0.35))
        self.iou = float(cfg.get("models.detector.iou", 0.45))
        self.imgsz = int(cfg.get("models.detector.imgsz", 640))
        self.tracker = str(cfg.get("tracking.tracker", "bytetrack"))
        # ultralytics'in beklediği yaml adını bir kez kur (her karede string kurma yok).
        self.tracker_yaml = f"{self.tracker}.yaml"
        vc = cfg.get("models.detector.vehicle_classes", [])
        # FAIL-CLOSED (Codex): boş liste 'süzgeç yok = her şey araç' anlamına gelip
        # bilinmeyen sınıfların OCR/hız hattına sızmasına yol açıyordu. Boşsa COCO araç
        # varsayılanlarına düş (fail-open değil), bilinmeyen sınıfı araç sayma.
        _COCO_VEHICLES = {"car", "bus", "truck", "motorcycle", "minibus"}
        self.vehicle_classes = set(vc) if vc else set(_COCO_VEHICLES)
        if not vc:
            log.warning(
                "models.detector.vehicle_classes boş → COCO araç varsayılanları kullanılıyor"
            )
        # Sürücü kilidi için kişi sınıfları (aynı ByteTrack geçişinde toplanır)
        pc = cfg.get("driver_lock.person_classes", ["person"])
        self.person_classes = set(pc) if pc else set()
        self.last_persons: list[Person] = []
        # Trafik tabelası sınıfları (araç/kişi DIŞI; SignTracker tüketir). value_map
        # anahtarları da tabela sayılır → config'te classes eksik kalsa bile yakalanır.
        self.sign_enabled = bool(cfg.get("sign.enabled", True))
        sc = cfg.get("sign.classes", []) or []
        vmap = cfg.get("sign.value_map", {}) or {}
        self.sign_classes = set(sc) | {str(k) for k in vmap}
        self.last_signs: list[Sign] = []
        # Yardımcı kanıt sınıfları (kanonik adlar): fine-tune dedektör (ör. v4)
        # 'phone' nesnesini tam karede görürse pipeline sürücü durumuna füzyon eder.
        ax = cfg.get("models.driver_state.aux_classes", ["phone", "smoking"]) or []
        self.aux_classes = set(ax)
        self.last_aux: list[BBox] = []
        # Araç-içi kopya kutu bastırma: NMS-free YOLO26/fine-tune modeller aynı araca
        # hafif kaymış İKİNCİ bir kutu üretebiliyor; kopya her karede yeni ByteTrack
        # ID'si doğurup (hayalet track) OCR/driver maliyeti yaratıyor. Aynı sınıftan,
        # IoU > dedup_iou örtüşen kutulardan yalnız en yüksek conf'lu tutulur.
        self.dedup_iou = float(cfg.get("models.detector.dedup_iou", 0.80))
        self.device = resolve_device(cfg.get("runtime.device", "auto"))
        log.info(
            "YOLO26 yüklendi: %s (imgsz=%d, tracker=%s, device=%s)",
            self.path,
            self.imgsz,
            self.tracker,
            self.device,
        )

    def _track(self, frame: np.ndarray):
        """model.track sarmalayıcı: seçili cihaz (MPS/CUDA) çalışma-zamanında
        çökerse (ör. 'no kernel image', MPS backend hatası) bir KEZ CPU'ya düş ve
        cihazı kalıcı CPU yap — tek kare hatası tüm akışı durdurmasın (HW-002).
        Zaten CPU'daysa hata yeniden fırlatılır (gizlenecek yedek yok)."""
        try:
            return self.model.track(
                frame,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                tracker=self.tracker_yaml,
                device=self.device,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001 - GPU/MPS çalışma-zamanı hatası → CPU fallback
            if self.device == "cpu":
                raise
            log.warning(
                "Dedektör cihazı '%s' çalışma-zamanında başarısız (%s) → CPU'ya düşülüyor",
                self.device,
                e,
            )
            self.device = "cpu"
            return self.model.track(
                frame,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                tracker=self.tracker_yaml,
                device=self.device,
                verbose=False,
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._track(frame)
        dets: list[Detection] = []
        self.last_persons = []
        self.last_signs = []
        self.last_aux = []
        if not results:
            return dets
        r = results[0]
        names = getattr(r, "names", None) or self.model.names
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return dets
        for b in boxes:
            cls_idx = int(b.cls.item())
            cls_name = (
                names[cls_idx]
                if isinstance(names, (list, tuple))
                else names.get(cls_idx, str(cls_idx))
            )
            # Model-uzayı adını kanonik ada çevir ('cell phone'→'phone' vb.) —
            # stok COCO ve fine-tune ağırlıkları aynı config sözleşmesiyle çalışır.
            cls_name = canonical(cls_name)
            is_person = cls_name in self.person_classes
            is_sign = self.sign_enabled and cls_name in self.sign_classes
            is_aux = cls_name in self.aux_classes
            # vehicle_classes boşsa "süzgeç yok" demektir → kişi/tabela/kanıt dışı her şey araç.
            is_vehicle = (
                cls_name in self.vehicle_classes
                if self.vehicle_classes
                else not (is_person or is_sign or is_aux)
            )
            if not (is_vehicle or is_person or is_sign or is_aux):
                continue
            xyxy = b.xyxy[0].tolist()
            tid = int(b.id.item()) if getattr(b, "id", None) is not None else None
            bbox = BBox(
                x1=xyxy[0],
                y1=xyxy[1],
                x2=xyxy[2],
                y2=xyxy[3],
                conf=float(b.conf.item()),
                cls=cls_name,
            )
            # Tabela → SignTracker'a (sahne-seviyesi; ID-merkezli akışa girmez)
            if is_sign:
                self.last_signs.append(Sign(bbox=bbox, cls=cls_name, track_id=tid))
                continue
            # Yardımcı kanıt nesnesi (phone/smoking) → pipeline füzyonuna
            if is_aux:
                self.last_aux.append(bbox)
                continue
            # Kişi sınıfı → sürücü kilidine; (araç sınıfıyla çakışmaz, COCO'da ayrık)
            if is_person:
                self.last_persons.append(Person(bbox=bbox, track_id=tid))
                continue
            d = Detection(bbox=bbox, track_id=tid)
            dets.append(d)
        dets = self._dedup(dets)
        # ROI crop'lar yalnız dedup'tan SAĞ ÇIKAN tespitler için üretilir (maliyet).
        for d in dets:
            d.cabin_roi, d.plate_roi = crop_rois(frame, d.bbox)
        return dets

    @staticmethod
    def _iou(a: BBox, b: BBox, area_a: float | None = None, area_b: float | None = None) -> float:
        ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
        ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        # Alanlar verilmezse property'den hesapla (geriye uyum); _dedup önceden hesaplar.
        if area_a is None:
            area_a = a.width * a.height
        if area_b is None:
            area_b = b.width * b.height
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _dedup(self, dets: list[Detection]) -> list[Detection]:
        """Yüksek-IoU'lu kopya araç kutularını bastır (en iyi conf kalır).

        SINIFTAN BAĞIMSIZ: fine-tune araç-tipi modelleri aynı fiziksel araca hem
        'car' hem 'truck' kutusu üretebiliyor (NMS sınıf-bazlı çalıştığından ikisi de
        sağ kalır). Liste zaten yalnız araçları içerir ve iki gerçek araç fiziksel
        olarak %80+ örtüşemez → sınıf kontrolü yapılmaz.
        """
        if self.dedup_iou >= 1.0 or len(dets) < 2:
            return dets
        ordered = sorted(dets, key=lambda d: d.bbox.conf, reverse=True)
        # Kutu alanlarını bir kez önceden hesapla (O(n^2) döngüde property çağrı yükü yok).
        areas = [d.bbox.width * d.bbox.height for d in ordered]
        kept: list[tuple[Detection, float]] = []
        for d, area_d in zip(ordered, areas, strict=True):
            if not any(
                self._iou(k.bbox, d.bbox, area_k, area_d) >= self.dedup_iou for k, area_k in kept
            ):
                kept.append((d, area_d))
        return [k for k, _ in kept]
