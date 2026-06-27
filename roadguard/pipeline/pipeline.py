"""Pipeline orkestratörü.

Akış (plan.md §6.9):
  preprocessing → detection+track → ROI → stability ⊗ (driver_state ∥ plate)
                → speed → accumulator → events + annotations

İki-kanal çıktı: `AnnotationFrame` (kare başına bbox, dashboard canvas için) ve
`RoadGuardEvent` (durum değişimleri). Pipeline upstream/downstream'i bilmez.
"""

# `from __future__ import annotations`: tip ipuçlarını metin (lazy) olarak değerlendir;
# böylece `np.ndarray` gibi ağır importları runtime'da yapmadan imza yazabiliriz.
from __future__ import annotations

import inspect  # crop_rois imza varsayılanından cabin_ratio'yu türetmek için (desync önleme)
import logging  # pipeline olaylarını "roadguard.pipeline" kanalına loglamak için
from collections.abc import Iterator  # frames() jeneratörünün dönüş tipini belirtmek için
from typing import TYPE_CHECKING  # sadece tip-denetiminde çalışan, runtime'da atlanan import bloğu

# --- Pipeline'ın orkestra ettiği alt modüller (her biri akışın bir aşaması) --- #
# Aşağıdaki her import bir boru-hattı aşamasını getirir; ne işe yaradıkları:
#   Accumulator           → track durumunu biriktirir, durum değişiminde event üretir
#   build_detector/crop_* → araç (ve kişi) tespiti + ROI kırpma yardımcıları
#   build_driver_engine   → Stage-2 sürücü-durum motoru (ID-merkezli; Katman A model + Katman B oylama)
#   EventEmitter          → event ve annotation'ları downstream'e (dashboard) yayınlar
#   DriverLock            → sürücüyü araca kilitleyen kimlik takipçisi
#   get_optional          → §8 opsiyonel modülleri tembel (lazy) yükler
#   PlateReader           → plaka OCR okuyucu
#   Preprocessor          → kare ön-işleme (yeniden boyut/normalizasyon vb.)
#   QoDController         → Quality-on-Demand: anomalide ilgili track'in kalitesini yükseltir
#   schema.*              → ortak veri sözleşmeleri (TrackRecord, event tipleri, annotation karesi)
#   SpeedEstimator        → bbox hareketinden hız/göreli hız tahmini
#   StabilityTracker      → 16/8 kararlılık süzgeci (titreşimli bayrakları yumuşatır)
from roadguard.accumulator.accumulator import Accumulator
from roadguard.detection.detector import (
    build_detector,
    cap_roi_to_area,
    crop_person_roi,
    crop_rois,
)
from roadguard.driver_state.engine import build_driver_engine
from roadguard.events.emitter import EventEmitter
from roadguard.identity.driver_lock import DriverLock
from roadguard.optional.loader import get_optional
from roadguard.plate.reader import PlateReader
from roadguard.preprocessing.preprocess import Preprocessor
from roadguard.qod.client import QoDController
from roadguard.scene.sign_tracker import SignTracker
from roadguard.schema import (
    AnnotationFrame,
    RoadGuardEvent,
    TrackRecord,
    make_event,
)
from roadguard.speed.estimator import SpeedEstimator
from roadguard.stability.class_vote import TrackClassVoter
from roadguard.stability.state_machine import StabilityTracker

# Bu blok yalnızca tip-denetleyiciler (mypy vb.) için çalışır, çalışma anında atlanır;
# numpy'yi runtime'da import etmeden `np.ndarray` imzası yazmamızı sağlar.
if TYPE_CHECKING:
    import numpy as np

# Modül seviyesinde tek logger: tüm pipeline mesajları "roadguard.pipeline" altında toplanır.
log = logging.getLogger("roadguard.pipeline")

# Kararlılık süzgecinden geçirilen 4 sürücü-durumu bayrağı (her biri ayrı izlenir).
# frozenset: aux füzyon döngüsünde `aux.cls in _DRIVER_FIELDS` üyelik kontrolü O(1)
# (tuple'da O(n) idi; etki küçük ama her aux nesnesi için her frame çalışıyor).
_DRIVER_FIELDS = frozenset(("phone", "smoking", "no_seatbelt", "fatigue"))


def record_to_annotation(rec: TrackRecord) -> dict:
    """TrackRecord → dashboard canvas için annotation sözlüğü.

    İç veri modelini (TrackRecord) dashboard'un beklediği düz JSON-uyumlu sözlüğe
    çevirir; böylece çizim katmanı pipeline iç tiplerini bilmek zorunda kalmaz.
    """
    return {
        "track_id": rec.track_id,  # aracın takip kimliği (kareler arası aynı kalır)
        "bbox": [rec.bbox.x1, rec.bbox.y1, rec.bbox.x2, rec.bbox.y2],  # çizilecek kutu köşeleri
        "cls": rec.vehicle_class,  # araç sınıfı (car/truck/bus...)
        "conf": rec.bbox.conf,  # tespit güven skoru
        "plate": rec.plate.value,  # okunan plaka metni (yoksa None)
        "plate_status": rec.plate.status,  # plaka okuma durumu (ör. okundu/bekliyor)
        "driver": rec.driver.active_flags(),  # aktif sürücü ihlalleri listesi
        "driver_id": rec.driver_id,  # kilitlenmiş sürücünün kimliği
        "driver_locked": rec.driver_locked,  # sürücü araca kilitlendi mi
        "speed_kmh": rec.speed.value_kmh,  # tahmini hız (km/s)
        "speed_calibrated": rec.speed.is_calibrated,  # km/h gerçek (kalibre) mi
        "relative_velocity_flag": rec.speed.relative_velocity_flag,  # ego'ya göre hızlı yaklaşıyor mu
        "swerving": rec.speed.swerving,  # dikkatsiz sürüş (yanal zigzag/ani kayma)
        "plate_partial": rec.plate.partial,  # tam doğrulanamadıysa en güçlü plaka adayı
        "risk_flags": rec.risk_flags,  # birleşik risk bayrakları
        "qod_active": rec.qod_active,  # bu track için yüksek-kalite modu açık mı
    }


class Pipeline:
    """Tüm alt modülleri tek akışta birleştiren orkestratör.

    __init__ tüm aşama nesnelerini bir kez kurar; sonra her kare process_frame()
    içinde bu hazır nesnelerden geçirilir (kare başına yeniden kurulum yok).
    """

    def __init__(self, cfg):
        self.cfg = cfg  # tüm modüllerin ayarlarını çektiği merkezi yapılandırma
        self.pre = Preprocessor(cfg)  # 1. aşama: ham kareyi modele uygun hale getirir
        self.detector = build_detector(cfg)  # 2. aşama: araçları (ve kişileri) tespit + takip eder
        # bayrak titreşimini bastıran 16/8 kararlılık süzgeci
        self.stability = StabilityTracker(cfg)
        # Track başına sınıf oylaması: tek-kare 'car↔truck' titremesini çoğunlukla düzeltir
        # (gerçek video ölçümü: araç ilk karede 0.8 güvenle 'truck', sonra kalıcı 'car').
        self.cls_voter = TrackClassVoter(cfg)
        # Stage-2 sürücü-durum motoru (ID-merkezli): Katman A model (pose-hibrit/YOLO26l/mock)
        # + Katman B per-track zaman-oylaması (eski per-alan 16/8 stability'nin yerine).
        self.driver = build_driver_engine(cfg)
        self.driver_lock = DriverLock(cfg)  # sürücüyü araca kilitleyen kimlik takipçisi
        # Sürücü kutusunu kırparken etrafa eklenen oran (varsayılan %15 dolgu).
        self.driver_roi_pad = float(cfg.get("driver_lock.roi_pad", 0.15))
        # Kabin (sürücü bölgesi) oranı: araç bbox yüksekliğinin üst payı kabin sayılır.
        # TEK doğruluk kaynağı crop_rois imza varsayılanı (desync önlenir: aynı değer
        # hem ROI kırpmada hem aşağıdaki devasa-ROI cap geometri hesabında kullanılır).
        # Config'le opsiyonel override (plate.cabin_ratio); verilmezse imza varsayılanı.
        _cabin_default = inspect.signature(crop_rois).parameters["cabin_ratio"].default
        self.cabin_ratio = float(cfg.get("plate.cabin_ratio", _cabin_default))
        # Devasa sürücü ROI sınırı (yalnız geometrik kabin FALLBACK'inde, kişi-kutusu yok):
        # ROI kare alanının max_roi_area_ratio'sunu aşarsa sürücü köşesine doğru o orana
        # KIRPILIR; skip_if_oversized=True ise (alternatif) o kare sürücü-durum çıkarımı
        # ATLANIR (devasa/güvenilmez ROI'de FP üretme). Kişi-kutusu varken ROI zaten
        # dardır → bu sınır DEVREYE GİRMEZ (davranış değişmez). 0/None → kapalı.
        dc = cfg.get("models.driver_state.driver_crop", {}) or {}
        self.driver_max_roi_area_ratio = float(dc.get("max_roi_area_ratio", 0.0) or 0.0)
        self.driver_skip_if_oversized = bool(dc.get("skip_if_oversized", False))
        corner = str(cfg.get("driver_lock.corner", "bottom_right")).lower()
        self._driver_corner = {
            "bottom_right": (1.0, 1.0),
            "bottom_left": (0.0, 1.0),
            "top_right": (1.0, 0.0),
            "top_left": (0.0, 0.0),
        }.get(corner, (1.0, 1.0))
        self.qod = QoDController(cfg)  # Quality-on-Demand kontrolcüsü (anomalide kalite yükseltir)
        self.plate = PlateReader(cfg, qod=self.qod)  # plaka okuyucu; QoD ile koordineli çalışır
        self.speed = SpeedEstimator(cfg)  # hız/göreli hız tahmincisi
        # MUTLAK yüksek-hız tabanı (km/s): yalnızca tabela YOKKEN devreye girer.
        # QoD tetiği (aşağıda) tabela varsa doğrudan onun limitini kullanır, yoksa bu tabana düşer.
        # (Aynı mantık accumulator'daki 'speed.speeding' dikkatsiz-sürüş kuralında da geçerli.)
        self.high_speed = float(cfg.get("risk.high_speed_kmh", 90))
        self.acc = Accumulator(cfg)  # track durumunu biriktirir ve durum değişiminde event üretir
        # Stage-1 yardımcı kanıt füzyonu: dedektör tam karede phone/smoking nesnesi
        # görmüşse ve kutu araca düşüyorsa sürücü durumuna OR'lanır (16/8'den geçer).
        self.fuse_aux = bool(cfg.get("models.driver_state.fuse_detections", True))
        # QoD yaklaşma tetiği (şartname: "TOGG aracının yaklaştığını algıladığında"):
        # bbox alanı pencere içinde `growth` katına çıktıysa ve araç yeterince
        # büyükse (min_area_ratio) → kritik an, optimize tetikle.
        ap = cfg.get("qod.approach", {}) or {}
        self.approach_enabled = bool(ap.get("enabled", True))
        self.approach_window = int(ap.get("window", 20))
        self.approach_growth = float(ap.get("growth", 1.35))
        self.approach_min_area = float(ap.get("min_area_ratio", 0.02))
        from collections import deque as _deque

        self._area_hist: dict[int, _deque] = {}
        self._deque = _deque
        # Ağır aşama kapısı: bir track bu kadar karede bir görülmeden driver/plaka
        # aşamalarına girmez (hidden_prototip min-track-hits dersi: tek-kare hayalet
        # tespitler OCR/pose maliyeti yaratmasın, çıktıyı kirletmesin).
        self.min_track_frames = int(cfg.get("tracking.min_track_frames", 3))
        # ÇIKTI kapısı (annotation/event): heavy-stage kapısından AYRI ayarlanabilir.
        # Bir track çıktı üretmek için en az bu kadar kare görülmüş OLMALI (kümülatif
        # görünürlük). Vars. min_track_frames (geriye dönük uyum); orkestratör çıktı
        # bastırmayı heavy-stage'den bağımsız sıkılaştırabilir. Çıktı eşiği heavy
        # kapısından küçük OLAMAZ (heavy'den geçmeyen track zaten çıktı üretemez).
        self.min_output_frames = max(
            self.min_track_frames,
            int(cfg.get("tracking.min_output_frames", self.min_track_frames)),
        )
        self._track_age: dict[int, int] = {}
        # MEM-005 bellek hijyeni: pipeline'ın per-track sözlükleri (_track_age/
        # _area_hist) için son-görülme karesi. prune() max_age'den eski (artık
        # görünmeyen) track'leri düşürür — kısa kesinti/oklüzyonda (recycled id,
        # max_age grace içinde) kümülatif age KORUNUR (davranış-koruyan; immediate
        # set-fark çıktı kapısını bozardı). speed/driver_lock ile aynı grace deseni.
        self._track_max_age = int(cfg.get("tracking.max_age", 30))
        self._track_last_seen: dict[int, int] = {}
        # Sahne-seviyesi tabela takibi: aktif hız limitini çıkarır (ID-merkezli akışın yanında)
        self.sign_tracker = SignTracker(cfg)
        self.emitter = EventEmitter()  # event/annotation'ları downstream'e yayınlar
        self.frame_idx = 0  # dışarıdan idx verilmezse kullanılan kare sayacı
        self.fps = 30.0  # zaman tabanlı hesaplar için kare hızı (frames() içinde güncellenir)
        # §8 opsiyonel: kapalıysa None döner, import bile yapılmaz (lazy)
        self.zwp = get_optional(cfg, "zero_waste_payload")  # sıfır-atık payload üreteci (varsa)

    # --- tek kare ---------------------------------------------------------- #
    def process_frame(
        self, frame: np.ndarray, frame_idx: int | None = None
    ) -> tuple[AnnotationFrame, list[RoadGuardEvent]]:
        # Tek bir kareyi baştan sona işler ve (annotation, events) ikilisi döndürür.
        # Kare no verilmezse iç sayacı kullan (canlı kamera/akış senaryosu).
        idx = self.frame_idx if frame_idx is None else frame_idx
        # QoD'a "şu anki zaman" bilgisini ver (kare no / fps = saniye); 1e-6 ile sıfıra bölme koruması.
        now = idx / max(self.fps, 1e-6)
        self.qod.set_now(now)
        # Accumulator event'leri de AYNI frame-saatini kullansın (QoD ile tutarlı ts ekseni;
        # offline eval tekrar-üretilebilirliği — wall-clock time.time() kayması giderildi).
        self.acc.set_now(now)
        frame = self.pre.process(frame)  # 1) ön-işleme: kareyi standart hale getir
        detections = self.detector.detect(frame)  # 2) tespit+takip: araç kutuları + track_id'ler
        # Sürücü kilidi için aynı karede tespit edilen kişiler (YOLO; mock'ta boş olabilir)
        persons = getattr(self.detector, "last_persons", [])
        # Sahne tabelaları (YOLO; mock'ta sign.mock_synthetic açıksa sentetik)
        signs = getattr(self.detector, "last_signs", [])

        events: list[RoadGuardEvent] = []  # bu karede üretilen tüm event'ler burada toplanır
        track_dicts: list[dict] = []  # bu karedeki her aracın annotation sözlüğü
        # ÇIKTIYA GİREN araçların track_id'leri: yalnız ÇIKTI kapısını (min_output_frames)
        # ve takip-guard'ını (track_id>=0) geçen, track_dicts'e yazılan araçlar buraya
        # girer. persons (sürücü/yolcu) çıktısı YALNIZ bu araçlara bağlanır → bastırılmış/
        # genç araç ve takipsiz (vehicle_id=-1) araç ASLA orphan sürücü kutusu üretmez
        # (track_dicts ile persons tam tutarlı; K-004: kurala bağlı, videoya değil).
        emitted_vehicle_ids: set[int] = set()

        # Sahne-seviyesi tabela bağlamı: aktif hız limitini çıkar ve accumulator'a ver.
        # Araç döngüsünden ÖNCE yapılır — çünkü 'speed.over_limit' risk koşulu bunu kullanır.
        scene, scene_events = self.sign_tracker.update(signs, idx, now=now)
        events.extend(scene_events)
        self.acc.set_scene(scene)

        # Sürücü kilidi GLOBAL çözülür: kare içindeki TÜM araç↔kişi eşleşmesi tek seferde
        # yapılır ki bir kişi yalnızca GERÇEKTEN içinde olduğu TEK araca kilitlensin
        # (örtüşen araçlarda çift-sahiplenme olmasın). Araç döngüsünden ÖNCE, çünkü
        # karar tüm araçları aynı anda görmeyi gerektirir. Sıra: vehicles[i] ↔ assign[i].
        vehicles = [
            (det.track_id if det.track_id is not None else -1, det.bbox) for det in detections
        ]
        driver_assignments = self.driver_lock.assign_frame(vehicles, persons, idx)

        # Her tespit edilen araç için aşamaları sırayla uygula:
        for det, assign in zip(detections, driver_assignments, strict=True):
            # GUARD: takip KURULMAMIŞ tespit (track_id is None). Tüm böyle tespitler
            # tek bir yapay '-1' kimliğine çökerdi → per-track durum (sınıf oyu, yaş
            # sayacı, hız geçmişi) farklı araçlar arasında KİRLENİR ve '-1' sayacı hızla
            # min_track_frames'i aşıp annotation/event'e SIZARDI (gerçek ölçüm: summary'lerde
            # track_id=-1 çıktısı). Takipsiz tespit ağır aşamalara da çıktıya da girmez —
            # net erken çıkış (K-004: kurala, videoya değil, takip-durumuna bağlı).
            if det.track_id is None:
                continue
            tid = det.track_id
            # Sınıf oyu: tek-kare 'car↔truck' titremesi track çoğunluğuyla düzeltilir.
            # Oy ALAN-AĞIRLIKLI: yakın/büyük araç sınıfı daha güvenilir (uzak araç
            # truck görünebiliyor — gerçek ölçüm). det.bbox.cls YERİNDE güncellenir →
            # hız genişlik-önseli, accumulator, annotation, event'ler aynı kararlı sınıfı görür.
            _h, _w = frame.shape[0], frame.shape[1]
            _area_norm = (det.bbox.width * det.bbox.height) / max(1.0, float(_h * _w))
            det.bbox.cls = self.cls_voter.update(
                tid, det.bbox.cls, det.bbox.conf, _area_norm, frame_idx=idx
            )

            # Ağır aşama kapısı: yeni doğan track min_track_frames kare görülmeden
            # ağır aşamalara (driver_state/plaka OCR) girmez (maliyet koruması).
            # (Gerçek video_3 dersi: ByteTrack parçalanması 2-karelik hayalet 'truck'
            # track'leri doğuruyor — bunlar kanıt videosuna/dashboard'a sızmamalı.)
            age = self._track_age[tid] = self._track_age.get(tid, 0) + 1
            self._track_last_seen[tid] = idx  # MEM-005: prune grace için son-görülme
            if age < self.min_track_frames:
                # Hız geçmişi yine de biriksin: gerçek track ise km/h erken otursun.
                self.speed.update(tid, det.bbox, idx, frame.shape)
                continue

            # Araç kutusundan iki ROI kes: kabin (sürücü bölgesi) ve plaka bölgesi.
            cabin, plate_roi = crop_rois(frame, det.bbox, cabin_ratio=self.cabin_ratio)

            # Sürücü ROI: kilitli/aday sürücünün kutusundan kes (kesin, DAR);
            # kişi yoksa geometrik kabin crop'una düş (geriye dönük uyumluluk).
            # Fallback devasa ROI (ön cam + yolcu) FP kaynağıdır → boyut sınırı uygulanır.
            using_fallback = True
            if assign.driver_bbox is not None:
                driver_roi = crop_person_roi(frame, assign.driver_bbox, self.driver_roi_pad)
                if driver_roi is None:
                    driver_roi = cabin
                else:
                    using_fallback = False  # kişi kutusundan kesik DAR ROI: sınır uygulanmaz
            else:
                driver_roi = cabin
            # Devasa kabin-fallback ROI sınırı: yalnız fallback'te ve config açıkken.
            # (Kişi-kutusu ROI'si zaten dar — using_fallback=False → değişmez.)
            if (
                using_fallback
                and driver_roi is not None
                and getattr(driver_roi, "size", 0)
                and self.driver_max_roi_area_ratio > 0
            ):
                h_, w_ = frame.shape[0], frame.shape[1]
                # cabin kutusunun frame koordinatları (crop_rois ile AYNI cabin_ratio —
                # tek kaynak self.cabin_ratio; hardcode 0.55 desync riski giderildi)
                cx1 = max(0, int(det.bbox.x1))
                cy1 = max(0, int(det.bbox.y1))
                cx2 = min(w_, int(det.bbox.x2))
                cy2 = min(h_, int(det.bbox.y1 + (det.bbox.y2 - det.bbox.y1) * self.cabin_ratio))
                capped = cap_roi_to_area(
                    frame,
                    (cx1, cy1, cx2, cy2),
                    self.driver_max_roi_area_ratio,
                    self._driver_corner,
                )
                if capped is not None:
                    if self.driver_skip_if_oversized:
                        # Alternatif politika: devasa/güvenilmez ROI'de çıkarımı ATLA
                        # (engine None ROI'de boş DriverState üretir → negatif oy, FP yok).
                        driver_roi = None
                    else:
                        nx1, ny1, nx2, ny2 = capped
                        cropped = frame[ny1:ny2, nx1:nx2]
                        driver_roi = cropped.copy() if cropped.size else driver_roi

            # Stage-2 sürücü durumu — ID-merkezli iki katman (DriverStateEngine):
            #   Katman A: pose-hibrit / YOLO26l ham tahmin (track_id → latch belleği)
            #   Katman B: track_id başına zaman-oylaması (eski per-alan 16/8'in yerine)
            # Stage-1 yardımcı kanıt füzyonu: dedektörün tam karede gördüğü phone/smoking
            # NESNESİ BU aracın kutusuna düşüyorsa ham tahmine OR'lanır (kanıt da Katman B
            # oylamasından geçer — tek-kare nesne FP'si event olamaz).
            aux_flags: dict[str, float] = {}
            if self.fuse_aux:
                for aux in getattr(self.detector, "last_aux", []):
                    ax, ay = aux.center
                    if (
                        aux.cls in _DRIVER_FIELDS
                        and det.bbox.x1 <= ax <= det.bbox.x2
                        and det.bbox.y1 <= ay <= det.bbox.y2
                    ):
                        aux_flags[aux.cls] = max(aux_flags.get(aux.cls, 0.0), float(aux.conf))
            driver = self.driver.process(tid, driver_roi, idx, aux_flags=aux_flags)

            # Plaka OCR: ilgili ROI'den oku; track'e göre sonucu biriktirir/günceller.
            plate = self.plate.update(
                tid, plate_roi, det.bbox, frame.shape, frame=frame, frame_idx=idx
            )
            # Hız tahmini: bbox'ın kareler arası hareketinden km/s ve göreli hız bayrağı.
            # LP dedektörü bu karede plakayı bulduysa (last_plate_bbox), 520 mm referanslı
            # en kesin ppm örneğini oto-kalibrasyona besle → daha hızlı/doğru km/h.
            speed = self.speed.update(
                tid, det.bbox, idx, frame.shape, plate_bbox=self.plate.last_plate_bbox
            )
            # göreli hız bayrağını da 16/8 süzgecinden geçir (eşik civarı salınımı önle)
            speed.relative_velocity_flag = bool(
                self.stability.update(
                    f"{tid}:speed.rel", speed.relative_velocity_flag, frame_idx=idx
                )
            )
            # swerving bayrağı da 16/8 süzgecinden geçer (tek pencerelik zigzag FP'si elenir)
            speed.swerving = bool(
                self.stability.update(f"{tid}:speed.swerve", speed.swerving, frame_idx=idx)
            )
            if speed.swerving:
                # Dikkatsiz sürüş kritik andır: delil kalitesi için QoD optimize iste.
                self.qod.request_optimize(tid, "swerving")
            # Yaklaşma tetiği: bbox alanı pencerede growth katına çıktıysa araç
            # kameraya YAKLAŞIYOR → şartnamenin birincil QoD senaryosu.
            if self.approach_enabled:
                h_, w_ = frame.shape[0], frame.shape[1]
                area_norm = (det.bbox.width * det.bbox.height) / max(1.0, float(h_ * w_))
                ah = self._area_hist.setdefault(tid, self._deque(maxlen=self.approach_window))
                ah.append(area_norm)
                if (
                    len(ah) >= self.approach_window
                    and ah[-1] >= self.approach_min_area
                    and ah[0] > 0
                    and (ah[-1] / ah[0]) >= self.approach_growth
                ):
                    self.qod.request_optimize(tid, "vehicle_approach")
            # Hız anomalisi → bu track için QoD'dan kalite yükseltme iste (plaka/delil yakalama anı).
            # KATI tabela-takibi: tabela limiti varsa DOĞRUDAN onu kullan (120 bölgesinde 100 =
            # yasal → tetiklemez; 50 bölgesinde 60 → tetikler), tabela yoksa high_speed tabanına düş.
            active_limit = self.sign_tracker.active_limit
            speed_threshold = active_limit if active_limit is not None else self.high_speed
            if speed.relative_velocity_flag or (
                speed.value_kmh is not None and speed.value_kmh >= speed_threshold
            ):
                self.qod.request_optimize(tid, "speed_anomaly")

            # ÇIKTI kapısı (phantom bastırma) — accumulator'dan ÖNCE: track yeterli
            # KÜMÜLATİF görünürlüğe (min_output_frames) ulaşmadıysa accumulator'a HİÇ
            # girmez. Böylece bastırılmış karelerde accumulator durumu İLERLEMEZ ve kapı
            # açılan ilk GÖRÜNÜR karede tüm tek-atış geçişleri (DETECTION_UPDATE new /
            # PLATE_CONFIRMED / RISK_ALERT / SPEED_LIMIT_VIOLATION) DOĞRU üretilir. (Eski
            # sıra: acc.update_track state'i ilerletip event'i `continue` ile düşürüyordu
            # → geçiş kalıcı kayboluyordu.) Vars. min_output=min_track → pencere BOŞ,
            # davranış birebir korunur; heavy aşamalar (driver/plate/speed) zaten
            # min_track_frames'te çalışıp kendi kalıcı durumlarını kurmuştur.
            if age < self.min_output_frames:
                continue

            # Bu track için QoD'un güncel durumunu oku (aktif mi, hangi profil).
            qod_active, qod_profile = self.qod.state(tid)
            # Tüm alt-sonuçları accumulator'a ver: track durumunu günceller,
            # durum değişimi varsa (ihlal başladı/bitti vb.) event üretir.
            rec, ev = self.acc.update_track(
                tid,
                frame_idx=idx,
                bbox=det.bbox,
                vehicle_class=det.bbox.cls,
                plate=plate,
                driver=driver,
                speed=speed,
                qod_active=qod_active,
                qod_profile=qod_profile,
            )

            events.extend(ev)  # accumulator'ın ürettiği event'leri kare listesine ekle

            # Sürücü kimliğini kayda yaz; yeni kilitlendiyse event üret
            rec.driver_id = assign.driver_id  # kilitli sürücünün kimliği (yoksa None)
            rec.driver_locked = assign.locked  # bu araçta sürücü kilidi kurulu mu
            if assign.newly_locked:  # kilit tam bu karede kurulduysa tek seferlik event üret
                events.append(
                    make_event(
                        tid,
                        "DRIVER_LOCKED",
                        {"driver_id": assign.driver_id, "confirm_frames": assign.streak},
                        ts=now,  # deterministik frame-saati (accumulator/qod ile aynı eksen)
                    )
                )

            # Kaydı dashboard'un anlayacağı düz sözlüğe çevir.
            adict = record_to_annotation(rec)
            if self.zwp is not None:  # §8.1 sıfır-atık payload
                # Opsiyonel modül açıksa annotation'a sıkıştırılmış payload ekle.
                adict["zwp"] = self.zwp.build_payload(adict, plate_roi)
            track_dicts.append(adict)  # bu aracı kare çıktısına ekle
            emitted_vehicle_ids.add(tid)  # persons (sürücü/yolcu) yalnız bu araca bağlanabilir

        # --- kare sonu temizlik/ilerletme (araç döngüsü dışında) --- #
        self.driver_lock.prune(idx)  # uzun süredir görülmeyen sürücü kilitlerini düşür
        self.driver.prune(idx)  # giden araçların sürücü-durum tamponlarını düşür (bellek)
        self.speed.prune(
            idx
        )  # giden track'lerin hız/kalibrasyon durumunu düşür (bellek + tripwire bayat-durum)
        # Uzun-süreli akış bellek hijyeni (DF-001/MEM-001..004): kalan per-track
        # sözlüklerini de giden track'ler için düşür. Hepsi max_age GRACE'li
        # (_last_seen/_track_last_seen tabanlı) → kısa oklüzyon/recycled-id davranışı
        # KORUNUR; yalnız max_age'den uzun süredir görünmeyen track durumu düşer.
        self.plate.prune(idx)  # MEM-001/CL-003/DF-002: plaka _state/_pools/_reads
        self.acc.prune(idx)  # DF-001/MEM-002: accumulator.tracks bayat kayıtları
        self.stability.prune_aged(idx)  # MEM-003: 16/8 pencere/commit (track-bazlı)
        self.cls_voter.prune_aged(idx)  # MEM-004: sınıf-oyu sözlüğü
        # MEM-005: pipeline'ın kendi per-track sözlükleri (_area_hist yaklaşma-deque,
        # _track_age kümülatif görünürlük) giden track'ler için kalıcı birikiyordu.
        # max_age grace'li (speed/driver_lock deseni): yalnız max_age'den UZUN süredir
        # görünmeyen track düşer. DAVRANIŞ-KORUYAN: kısa kesinti/oklüzyonda kümülatif
        # age korunur → çıktı kapısı (min_output_frames) etkilenmez.
        dead_tracks = [
            tid for tid, seen in self._track_last_seen.items() if idx - seen > self._track_max_age
        ]
        for tid in dead_tracks:
            self._track_age.pop(tid, None)
            self._area_hist.pop(tid, None)
            self._track_last_seen.pop(tid, None)
        self.qod.tick()  # QoD zamanlayıcısını bir adım ilerlet (süresi dolan optimizasyonları kapat)
        self.qod.prune()  # MEM-00x: cooldown'ı geçmiş _last_release girdilerini düşür (sızıntı önle)
        events.extend(self.qod.drain_events())  # QoD'un kendi ürettiği event'leri (aç/kapa) topla

        # Sahne tabelalarını dashboard'un çizebileceği düz sözlüklere çevir (km/h çözülü).
        sign_dicts = [
            {
                "bbox": [s.bbox.x1, s.bbox.y1, s.bbox.x2, s.bbox.y2],
                "cls": s.cls,
                "conf": s.bbox.conf,
                "speed_limit_kmh": self.sign_tracker.limit_of(s.cls),
            }
            for s in signs
        ]

        # Kişileri SÜRÜCÜ/YOLCU rolüyle çıktıya ekle: dashboard insanları 'person' diye
        # değil, sürücü kilidinin kararıyla sürücü (yeşil) ve yolcu (turuncu) olarak çizer.
        # Sürücü = atamanın driver_bbox'ı (kilitli ya da aday); yolcular = passenger_ids.
        person_box = {p.track_id: p.bbox for p in persons if p.track_id is not None}
        person_dicts: list[dict] = []
        seen_pid: set[int] = set()
        for (vid, _), assign in zip(vehicles, driver_assignments, strict=True):
            # ÇIKTI-tutarlılık kapısı: persons YALNIZ track_dicts'e yazılmış (çıktı
            # kapısını + track_id>=0 guard'ını geçen) araçlara bağlanır. Takipsiz
            # (vid=-1) ve bastırılmış/genç (boş-track / age<min_output_frames) araçlar
            # emitted_vehicle_ids'te DEĞİL → sürücü/yolcu kutusu ÜRETMEZ (orphan/phantom
            # person sızıntısı kapatıldı; track_dicts ile persons birebir tutarlı).
            if vid not in emitted_vehicle_ids:
                continue
            if (
                assign.driver_bbox is not None
                and assign.driver_id is not None
                and assign.driver_id not in seen_pid
            ):
                b = assign.driver_bbox
                person_dicts.append(
                    {
                        "bbox": [b.x1, b.y1, b.x2, b.y2],
                        "role": "driver",
                        "track_id": assign.driver_id,
                        "vehicle_id": vid,
                        "locked": assign.locked,
                    }
                )
                seen_pid.add(assign.driver_id)
            for pid in assign.passenger_ids:
                b = person_box.get(pid)
                if b is None or pid in seen_pid:
                    continue
                person_dicts.append(
                    {
                        "bbox": [b.x1, b.y1, b.x2, b.y2],
                        "role": "passenger",
                        "track_id": pid,
                        "vehicle_id": vid,
                        "locked": False,
                    }
                )
                seen_pid.add(pid)

        # İki-kanal çıktı: tüm araçların kutuları tek annotation karesinde toplanır.
        anno = AnnotationFrame(
            frame_id=idx,
            tracks=track_dicts,
            persons=person_dicts,
            signs=sign_dicts,
            scene=scene.model_dump(),
        )
        for e in events:
            self.emitter.emit_event(e)  # her event'i downstream'e yayınla
        self.emitter.emit_annotation(anno)  # annotation karesini dashboard'a yayınla
        self.frame_idx = idx + 1  # iç sayacı ilerlet (bir sonraki çağrı için)
        return anno, events

    # --- video / kamera ---------------------------------------------------- #
    def frames(
        self, source, max_frames: int | None = None
    ) -> Iterator[tuple[np.ndarray, AnnotationFrame, list[RoadGuardEvent]]]:
        """Kaynağı aç ve (frame, annotation, events) üret. Kaynak: path | index | URL."""
        import cv2  # ağır bağımlılık; sadece video gerçekten işleneceği zaman import edilir

        # "0" gibi sayısal string ise kamera indeksine çevir; değilse dosya yolu/URL olarak bırak.
        src = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = cv2.VideoCapture(src)  # video/kamera akışını aç
        if not cap.isOpened():  # açılamadıysa erken ve açık hata ver
            raise RuntimeError(f"Kaynak açılamadı: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)  # kaynağın kare hızını oku
        self.fps = fps if fps and fps > 0 else 30.0  # geçersizse güvenli varsayılana (30) düş
        self.speed.fps = self.fps  # hız tahmincisi de aynı fps'i kullansın (km/s doğru çıksın)
        i = 0  # kare sayacı
        try:
            while True:
                # istenen kare sınırına ulaşıldıysa dur (max_frames=None ise sınırsız)
                if max_frames is not None and i >= max_frames:
                    break
                ok, frame = cap.read()  # sıradaki kareyi oku
                if not ok:  # akış bittiyse/okuma başarısızsa döngüyü bitir
                    break
                anno, events = self.process_frame(frame, i)  # kareyi tam pipeline'dan geçir
                yield frame, anno, events  # tüketiciye (dashboard/eval) akıt; tembel üretim
                i += 1
        finally:
            cap.release()  # hata olsa da olmasa da kaynağı serbest bırak (kaynak sızıntısını önler)

    def run_video(self, source, max_frames: int | None = None) -> list[RoadGuardEvent]:
        """Tüm kaynağı işle, üretilen tüm event'leri döndür (offline/eval kullanımı)."""
        # frames() jeneratörünü sonuna kadar tüketir; kareleri atıp sadece event'leri biriktirir.
        all_events: list[RoadGuardEvent] = []
        for _frame, _anno, events in self.frames(source, max_frames):
            all_events.extend(events)
        return all_events

    def close(self) -> None:
        # Pipeline kapanışında kaynak tutan alt modülleri (ör. model oturumu) serbest bırak.
        self.detector.close()
