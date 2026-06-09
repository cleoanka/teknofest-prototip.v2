"""Pipeline orkestratörü.

Akış (plan.md §6.9):
  preprocessing → detection+track → ROI → stability ⊗ (driver_state ∥ plate)
                → speed → accumulator → events + annotations

İki-kanal çıktı: `AnnotationFrame` (kare başına bbox, dashboard canvas için) ve
`AuraEvent` (durum değişimleri). Pipeline upstream/downstream'i bilmez.
"""

# `from __future__ import annotations`: tip ipuçlarını metin (lazy) olarak değerlendir;
# böylece `np.ndarray` gibi ağır importları runtime'da yapmadan imza yazabiliriz.
from __future__ import annotations

import logging  # pipeline olaylarını "aura.pipeline" kanalına loglamak için
from collections.abc import Iterator  # frames() jeneratörünün dönüş tipini belirtmek için
from typing import TYPE_CHECKING  # sadece tip-denetiminde çalışan, runtime'da atlanan import bloğu

# --- Pipeline'ın orkestra ettiği alt modüller (her biri akışın bir aşaması) --- #
# Aşağıdaki her import bir boru-hattı aşamasını getirir; ne işe yaradıkları:
#   Accumulator           → track durumunu biriktirir, durum değişiminde event üretir
#   build_detector/crop_* → araç (ve kişi) tespiti + ROI kırpma yardımcıları
#   build_driver_classifier → Stage-2 sürücü davranışı (telefon/sigara/kemer/yorgunluk)
#   EventEmitter          → event ve annotation'ları downstream'e (dashboard) yayınlar
#   DriverLock            → sürücüyü araca kilitleyen kimlik takipçisi
#   get_optional          → §8 opsiyonel modülleri tembel (lazy) yükler
#   PlateReader           → plaka OCR okuyucu
#   Preprocessor          → kare ön-işleme (yeniden boyut/normalizasyon vb.)
#   QoDController         → Quality-on-Demand: anomalide ilgili track'in kalitesini yükseltir
#   schema.*              → ortak veri sözleşmeleri (TrackRecord, event tipleri, annotation karesi)
#   SpeedEstimator        → bbox hareketinden hız/göreli hız tahmini
#   StabilityTracker      → 16/8 kararlılık süzgeci (titreşimli bayrakları yumuşatır)
from aura.accumulator.accumulator import Accumulator
from aura.detection.detector import build_detector, crop_person_roi, crop_rois
from aura.driver_state.classifier import build_driver_classifier
from aura.events.emitter import EventEmitter
from aura.identity.driver_lock import DriverLock
from aura.optional.loader import get_optional
from aura.plate.reader import PlateReader
from aura.preprocessing.preprocess import Preprocessor
from aura.qod.client import QoDController
from aura.scene.sign_tracker import SignTracker
from aura.schema import AnnotationFrame, AuraEvent, TrackRecord, make_event
from aura.speed.estimator import SpeedEstimator
from aura.stability.state_machine import StabilityTracker

# Bu blok yalnızca tip-denetleyiciler (mypy vb.) için çalışır, çalışma anında atlanır;
# numpy'yi runtime'da import etmeden `np.ndarray` imzası yazmamızı sağlar.
if TYPE_CHECKING:
    import numpy as np

# Modül seviyesinde tek logger: tüm pipeline mesajları "aura.pipeline" altında toplanır.
log = logging.getLogger("aura.pipeline")

# Kararlılık süzgecinden geçirilen 4 sürücü-durumu bayrağı (her biri ayrı izlenir).
_DRIVER_FIELDS = ("phone", "smoking", "no_seatbelt", "fatigue")


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
        "relative_velocity_flag": rec.speed.relative_velocity_flag,  # ego'ya göre hızlı yaklaşıyor mu
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
        self.driver = build_driver_classifier(cfg)  # Stage-2 sürücü davranış sınıflandırıcısı
        self.driver_lock = DriverLock(cfg)  # sürücüyü araca kilitleyen kimlik takipçisi
        # Sürücü kutusunu kırparken etrafa eklenen oran (varsayılan %15 dolgu).
        self.driver_roi_pad = float(cfg.get("driver_lock.roi_pad", 0.15))
        self.qod = QoDController(cfg)  # Quality-on-Demand kontrolcüsü (anomalide kalite yükseltir)
        self.plate = PlateReader(cfg, qod=self.qod)  # plaka okuyucu; QoD ile koordineli çalışır
        self.speed = SpeedEstimator(cfg)  # hız/göreli hız tahmincisi
        # MUTLAK yüksek-hız tabanı (km/s): yalnızca tabela YOKKEN devreye girer.
        # QoD tetiği (aşağıda) tabela varsa doğrudan onun limitini kullanır, yoksa bu tabana düşer.
        # (Aynı mantık accumulator'daki 'speed.speeding' dikkatsiz-sürüş kuralında da geçerli.)
        self.high_speed = float(cfg.get("risk.high_speed_kmh", 90))
        self.acc = Accumulator(cfg)  # track durumunu biriktirir ve durum değişiminde event üretir
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
    ) -> tuple[AnnotationFrame, list[AuraEvent]]:
        # Tek bir kareyi baştan sona işler ve (annotation, events) ikilisi döndürür.
        # Kare no verilmezse iç sayacı kullan (canlı kamera/akış senaryosu).
        idx = self.frame_idx if frame_idx is None else frame_idx
        # QoD'a "şu anki zaman" bilgisini ver (kare no / fps = saniye); 1e-6 ile sıfıra bölme koruması.
        self.qod.set_now(idx / max(self.fps, 1e-6))
        frame = self.pre.process(frame)  # 1) ön-işleme: kareyi standart hale getir
        detections = self.detector.detect(frame)  # 2) tespit+takip: araç kutuları + track_id'ler
        # Sürücü kilidi için aynı karede tespit edilen kişiler (YOLO; mock'ta boş olabilir)
        persons = getattr(self.detector, "last_persons", [])
        # Sahne tabelaları (YOLO; mock'ta sign.mock_synthetic açıksa sentetik)
        signs = getattr(self.detector, "last_signs", [])

        events: list[AuraEvent] = []  # bu karede üretilen tüm event'ler burada toplanır
        track_dicts: list[dict] = []  # bu karedeki her aracın annotation sözlüğü

        # Sahne-seviyesi tabela bağlamı: aktif hız limitini çıkar ve accumulator'a ver.
        # Araç döngüsünden ÖNCE yapılır — çünkü 'speed.over_limit' risk koşulu bunu kullanır.
        scene, scene_events = self.sign_tracker.update(signs, idx)
        events.extend(scene_events)
        self.acc.set_scene(scene)

        # Her tespit edilen araç için aşamaları sırayla uygula:
        for det in detections:
            # track_id yoksa -1 ile işaretle (takip kurulmamış geçici tespit).
            tid = det.track_id if det.track_id is not None else -1
            # Araç kutusundan iki ROI kes: kabin (sürücü bölgesi) ve plaka bölgesi.
            cabin, plate_roi = crop_rois(frame, det.bbox)

            # Sürücü kimlik kilidi: sağ-alt aday → 5 kare tutarlıysa araca kilitle
            assign = self.driver_lock.update(tid, det.bbox, persons, idx)

            # Sürücü ROI: kilitli/aday sürücünün kutusundan kes (kesin);
            # kişi yoksa geometrik kabin crop'una düş (geriye dönük uyumluluk).
            if assign.driver_bbox is not None:
                driver_roi = crop_person_roi(frame, assign.driver_bbox, self.driver_roi_pad)
                if driver_roi is None:
                    driver_roi = cabin
            else:
                driver_roi = cabin

            # Stage-2 sürücü durumu → 16/8 kararlılık süzgeci (alan-bazında)
            driver = self.driver.infer(driver_roi)  # ham tahmin: telefon/sigara/kemer/yorgunluk
            # Her bayrağı kendi anahtarıyla (track+alan) kararlılık süzgecinden geçir:
            # tek karelik yanlış pozitiflerin event'e dönüşmesini engeller.
            for f in _DRIVER_FIELDS:
                stable = self.stability.update(
                    f"{tid}:driver.{f}", getattr(driver, f), driver.confidence.get(f, 1.0)
                )
                setattr(driver, f, bool(stable))  # ham değeri kararlı (süzülmüş) değerle değiştir

            # Plaka OCR: ilgili ROI'den oku; track'e göre sonucu biriktirir/günceller.
            plate = self.plate.update(tid, plate_roi, det.bbox, frame.shape, frame=frame)
            # Hız tahmini: bbox'ın kareler arası hareketinden km/s ve göreli hız bayrağı.
            speed = self.speed.update(tid, det.bbox, idx, frame.shape)
            # göreli hız bayrağını da 16/8 süzgecinden geçir (eşik civarı salınımı önle)
            speed.relative_velocity_flag = bool(
                self.stability.update(f"{tid}:speed.rel", speed.relative_velocity_flag)
            )
            # Hız anomalisi → bu track için QoD'dan kalite yükseltme iste (plaka/delil yakalama anı).
            # KATI tabela-takibi: tabela limiti varsa DOĞRUDAN onu kullan (120 bölgesinde 100 =
            # yasal → tetiklemez; 50 bölgesinde 60 → tetikler), tabela yoksa high_speed tabanına düş.
            active_limit = self.sign_tracker.active_limit
            speed_threshold = active_limit if active_limit is not None else self.high_speed
            if speed.relative_velocity_flag or (
                speed.value_kmh is not None and speed.value_kmh >= speed_threshold
            ):
                self.qod.request_optimize(tid, "speed_anomaly")

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
                    )
                )

            # Kaydı dashboard'un anlayacağı düz sözlüğe çevir.
            adict = record_to_annotation(rec)
            if self.zwp is not None:  # §8.1 sıfır-atık payload
                # Opsiyonel modül açıksa annotation'a sıkıştırılmış payload ekle.
                adict["zwp"] = self.zwp.build_payload(adict, plate_roi)
            track_dicts.append(adict)  # bu aracı kare çıktısına ekle

        # --- kare sonu temizlik/ilerletme (araç döngüsü dışında) --- #
        self.driver_lock.prune(idx)  # uzun süredir görülmeyen sürücü kilitlerini düşür
        self.qod.tick()  # QoD zamanlayıcısını bir adım ilerlet (süresi dolan optimizasyonları kapat)
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

        # İki-kanal çıktı: tüm araçların kutuları tek annotation karesinde toplanır.
        anno = AnnotationFrame(
            frame_id=idx, tracks=track_dicts, signs=sign_dicts, scene=scene.model_dump()
        )
        for e in events:
            self.emitter.emit_event(e)  # her event'i downstream'e yayınla
        self.emitter.emit_annotation(anno)  # annotation karesini dashboard'a yayınla
        self.frame_idx = idx + 1  # iç sayacı ilerlet (bir sonraki çağrı için)
        return anno, events

    # --- video / kamera ---------------------------------------------------- #
    def frames(
        self, source, max_frames: int | None = None
    ) -> Iterator[tuple[np.ndarray, AnnotationFrame, list[AuraEvent]]]:
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

    def run_video(self, source, max_frames: int | None = None) -> list[AuraEvent]:
        """Tüm kaynağı işle, üretilen tüm event'leri döndür (offline/eval kullanımı)."""
        # frames() jeneratörünü sonuna kadar tüketir; kareleri atıp sadece event'leri biriktirir.
        all_events: list[AuraEvent] = []
        for _frame, _anno, events in self.frames(source, max_frames):
            all_events.extend(events)
        return all_events

    def close(self) -> None:
        # Pipeline kapanışında kaynak tutan alt modülleri (ör. model oturumu) serbest bırak.
        self.detector.close()
