"""SORUN 2 regresyon kapanı: track_id=-1 sızıntısı + phantom track çıktı bastırma.

Model/ağırlık gerektirmez (CI-uyumlu): pipeline mock modda kurulur, dedektör
scriptli bir sahte dedektörle değiştirilir → çıktı kapısı davranışı izole edilir.
"""

from __future__ import annotations

import numpy as np

from aura.detection.detector import Detection, Person
from aura.schema import BBox


class _ScriptedDetector:
    """Kare başına önceden belirlenmiş Detection listesi döndüren sahte dedektör.

    Pipeline'ın okuduğu sözleşmeyi taşır: detect(frame), close(), last_persons/
    last_signs/last_aux. Ağır model yok — çıktı kapısı saf mantığı test edilir.
    """

    def __init__(
        self,
        script: list[list[Detection]],
        persons_script: list[list[Person]] | None = None,
    ):
        self._script = script
        # Kare-başına kişi script'i (sürücü/yolcu testleri için); verilmezse hep boş.
        self._persons_script = persons_script
        self._i = 0
        self.last_persons: list = []
        self.last_signs: list = []
        self.last_aux: list = []

    def detect(self, frame):
        dets = self._script[self._i] if self._i < len(self._script) else []
        if self._persons_script is not None:
            self.last_persons = (
                self._persons_script[self._i] if self._i < len(self._persons_script) else []
            )
        self._i += 1
        return dets

    def close(self):  # pragma: no cover - kaynak yok
        pass


def _det(track_id, cls="car"):
    # Kare ortasında makul boyutlu bir araç kutusu (sweet-spot içinde).
    bbox = BBox(x1=240.0, y1=140.0, x2=400.0, y2=300.0, conf=0.9, cls=cls)
    return Detection(bbox=bbox, track_id=track_id)


def _person_in_vehicle(track_id):
    # _det() aracının (240,140)-(400,300) İÇİNE tam düşen kişi kutusu →
    # DriverLock containment >= min_containment → sürücü olarak seçilir (driver_bbox dolu).
    return Person(bbox=BBox(x1=300.0, y1=180.0, x2=360.0, y2=290.0, conf=0.9), track_id=track_id)


def _build_pipeline(cfg):
    from aura.pipeline import Pipeline

    p = Pipeline(cfg)
    return p


def _track_ids(anno):
    return [t["track_id"] for t in anno.tracks]


def test_untracked_detection_never_reaches_output(cfg):
    # track_id=None (takip kurulmamış) tespit ASLA annotation/event üretmez,
    # tekrarlasa bile (eski hata: hepsi tek '-1' kimliğine çöküp sayaç şişiyordu).
    p = _build_pipeline(cfg)
    p.detector = _ScriptedDetector([[_det(None)] for _ in range(10)])
    frame = np.zeros((360, 640, 3), np.uint8)
    leaked_tracks = []
    leaked_events = []
    for i in range(10):
        anno, events = p.process_frame(frame, i)
        leaked_tracks.extend(_track_ids(anno))
        leaked_events.extend(events)
    assert leaked_tracks == []  # hiçbir kare track çıktısı üretmedi
    assert all(e.track_id != -1 for e in leaked_events)  # -1 event'i de yok


def test_short_phantom_track_suppressed(cfg):
    # 1-2 karelik hayalet track (min_output_frames=3 altında) çıktı üretmez.
    cfg.data["tracking"]["min_output_frames"] = 3
    cfg.data["tracking"]["min_track_frames"] = 3
    p = _build_pipeline(cfg)
    p.detector = _ScriptedDetector([[_det(7)], [_det(7)]])  # yalnız 2 kare yaşar
    frame = np.zeros((360, 640, 3), np.uint8)
    seen = []
    for i in range(2):
        anno, _ = p.process_frame(frame, i)
        seen.extend(_track_ids(anno))
    assert seen == []  # phantom çıktıya sızmadı


def test_long_lived_track_reaches_output(cfg):
    # Uzun-ömürlü gerçek track (eşiği geçen) normal çıktı ÜRETİR (davranış korunur).
    cfg.data["tracking"]["min_output_frames"] = 3
    cfg.data["tracking"]["min_track_frames"] = 3
    p = _build_pipeline(cfg)
    p.detector = _ScriptedDetector([[_det(9)] for _ in range(6)])
    frame = np.zeros((360, 640, 3), np.uint8)
    seen = []
    for i in range(6):
        anno, _ = p.process_frame(frame, i)
        seen.extend(_track_ids(anno))
    # ilk (min_output_frames-1) kare bastırılır, sonrası çıktı verir
    assert 9 in seen
    assert seen.count(9) == 6 - (3 - 1)  # 4 kare çıktı


def test_output_gate_stricter_than_heavy_gate(cfg):
    # min_output_frames > min_track_frames: heavy aşama 2'de açılır ama çıktı 4'te.
    cfg.data["tracking"]["min_track_frames"] = 2
    cfg.data["tracking"]["min_output_frames"] = 4
    p = _build_pipeline(cfg)
    assert p.min_output_frames == 4 and p.min_track_frames == 2
    p.detector = _ScriptedDetector([[_det(3)] for _ in range(6)])
    frame = np.zeros((360, 640, 3), np.uint8)
    seen = []
    for i in range(6):
        anno, _ = p.process_frame(frame, i)
        seen.append(_track_ids(anno))
    # kare 0..2 (age 1,2,3 < 4): çıktı yok; kare 3..5 (age 4,5,6): çıktı var
    assert seen[0] == [] and seen[1] == [] and seen[2] == []
    assert seen[3] == [3] and seen[4] == [3] and seen[5] == [3]


def test_output_gate_floored_to_heavy_gate(cfg):
    # min_output_frames < min_track_frames verilse bile heavy kapısının altına inmez.
    cfg.data["tracking"]["min_track_frames"] = 5
    cfg.data["tracking"]["min_output_frames"] = 1
    p = _build_pipeline(cfg)
    assert p.min_output_frames == 5  # max(5, 1)


# --- MAJOR: phantom/orphan PERSONS sızıntısı (persons çıktı-tutarlılık kapısı) ---
def _persons(anno):
    return list(anno.persons)


def test_suppressed_young_vehicle_emits_no_persons(cfg):
    # Bastırılmış (genç, min_output_frames altı) araç: kişi kabinde olsa bile
    # sürücü/yolcu kutusu ÇIKMAZ (track_dicts ile persons tutarlı; orphan yok).
    cfg.data["tracking"]["min_output_frames"] = 3
    cfg.data["tracking"]["min_track_frames"] = 3
    p = _build_pipeline(cfg)
    # vehicle_id=7 yalnız 2 kare yaşar (age 1,2 < 3 → çıktı kapısını GEÇMEZ);
    # her karede kabininde bir kişi var (sürücü adayı).
    p.detector = _ScriptedDetector(
        script=[[_det(7)], [_det(7)]],
        persons_script=[[_person_in_vehicle(50)], [_person_in_vehicle(50)]],
    )
    frame = np.zeros((360, 640, 3), np.uint8)
    seen_tracks, seen_persons = [], []
    for i in range(2):
        anno, _ = p.process_frame(frame, i)
        seen_tracks.extend(_track_ids(anno))
        seen_persons.extend(_persons(anno))
    assert seen_tracks == []  # araç çıktıya girmedi (mevcut davranış)
    assert seen_persons == []  # ⇒ sürücü person_dict'i de ÜRETİLMEDİ (orphan kapatıldı)


def test_untracked_vehicle_emits_no_persons(cfg):
    # Takipsiz araç (track_id=None → vehicle_id=-1): kişi kabinde olsa bile
    # sürücü/yolcu kutusu ÇIKMAZ (vid=-1 emitted_vehicle_ids'te değil).
    cfg.data["tracking"]["min_output_frames"] = 1
    cfg.data["tracking"]["min_track_frames"] = 1
    p = _build_pipeline(cfg)
    p.detector = _ScriptedDetector(
        script=[[_det(None)] for _ in range(5)],
        persons_script=[[_person_in_vehicle(51)] for _ in range(5)],
    )
    frame = np.zeros((360, 640, 3), np.uint8)
    seen_persons = []
    for i in range(5):
        anno, _ = p.process_frame(frame, i)
        seen_persons.extend(_persons(anno))
    assert seen_persons == []  # vehicle_id=-1 için orphan sürücü kutusu KALMADI


def test_long_lived_vehicle_emits_persons(cfg):
    # Uzun-ömürlü gerçek araç (çıktı kapısını geçen): kabinindeki kişi sürücü olarak
    # çıktıya YANSIR (davranış korunur — persons tamamen kaybolmadı).
    cfg.data["tracking"]["min_output_frames"] = 3
    cfg.data["tracking"]["min_track_frames"] = 3
    p = _build_pipeline(cfg)
    p.detector = _ScriptedDetector(
        script=[[_det(9)] for _ in range(6)],
        persons_script=[[_person_in_vehicle(60)] for _ in range(6)],
    )
    frame = np.zeros((360, 640, 3), np.uint8)
    seen_persons = []
    last_anno = None
    for i in range(6):
        anno, _ = p.process_frame(frame, i)
        last_anno = anno
        seen_persons.extend(_persons(anno))
    # Çıktı kapısını geçen karelerde sürücü person_dict'i üretildi, vehicle_id=9 ile bağlı.
    assert seen_persons, "uzun-ömürlü araç için sürücü kutusu üretilmedi (persons gate fazla sıkı)"
    drivers = [pd for pd in seen_persons if pd["role"] == "driver"]
    assert drivers and all(pd["track_id"] == 60 for pd in drivers)
    assert all(pd["vehicle_id"] == 9 for pd in seen_persons)
    # Son karede araç da kişi de çıktıda → tam tutarlılık.
    assert 9 in _track_ids(last_anno) and _persons(last_anno)


# --- SORUN 3: devasa kabin-fallback ROI sınırı (pipeline entegrasyonu) ----------
def _spy_driver(p):
    """driver.process'e ulaşan ROI'leri yakala (kişi yok → kabin fallback yolu)."""
    seen = {"rois": []}
    orig = p.driver.process

    def wrapper(track_id, cabin_roi, *a, **k):
        seen["rois"].append(cabin_roi)
        return orig(track_id, cabin_roi, *a, **k)

    p.driver.process = wrapper
    return seen


def test_oversized_cabin_fallback_roi_is_capped(cfg):
    # Kişi-kutusu YOK → kabin fallback. Devasa araç (neredeyse tüm kare) → sürücü ROI
    # kare alanının max_roi_area_ratio'suna kırpılır (FP koruma; minimum-alan ilkesi).
    cfg.data["models"]["driver_state"]["driver_crop"]["max_roi_area_ratio"] = 0.10
    cfg.data["models"]["driver_state"]["driver_crop"]["skip_if_oversized"] = False
    cfg.data["tracking"]["min_track_frames"] = 1
    cfg.data["tracking"]["min_output_frames"] = 1
    p = _build_pipeline(cfg)
    seen = _spy_driver(p)
    # Devasa araç kutusu: 600x340 (kabin = üst %55 = 600x187 = ~%49 kare alanı)
    big = Detection(
        bbox=BBox(x1=20.0, y1=10.0, x2=620.0, y2=350.0, conf=0.9, cls="car"),
        track_id=11,
    )
    p.detector = _ScriptedDetector([[big]])
    frame = np.zeros((360, 640, 3), np.uint8)
    p.process_frame(frame, 0)
    assert seen["rois"], "driver.process çağrılmadı"
    roi = seen["rois"][0]
    assert roi is not None
    roi_area = roi.shape[0] * roi.shape[1]
    frame_area = 360 * 640
    assert roi_area <= 0.10 * frame_area + 5  # %10 sınırına kırpıldı


def test_oversized_cabin_fallback_roi_skipped_when_configured(cfg):
    # skip_if_oversized=True → devasa fallback karesinde sürücü ROI None (çıkarım atlanır).
    cfg.data["models"]["driver_state"]["driver_crop"]["max_roi_area_ratio"] = 0.10
    cfg.data["models"]["driver_state"]["driver_crop"]["skip_if_oversized"] = True
    cfg.data["tracking"]["min_track_frames"] = 1
    cfg.data["tracking"]["min_output_frames"] = 1
    p = _build_pipeline(cfg)
    seen = _spy_driver(p)
    big = Detection(
        bbox=BBox(x1=20.0, y1=10.0, x2=620.0, y2=350.0, conf=0.9, cls="car"),
        track_id=12,
    )
    p.detector = _ScriptedDetector([[big]])
    frame = np.zeros((360, 640, 3), np.uint8)
    p.process_frame(frame, 0)
    assert seen["rois"] and seen["rois"][0] is None  # çıkarım atlandı


def test_small_vehicle_fallback_roi_unchanged(cfg):
    # Küçük araç → kabin ROI zaten eşiğin altında → kırpma YOK (davranış değişmez).
    cfg.data["models"]["driver_state"]["driver_crop"]["max_roi_area_ratio"] = 0.10
    cfg.data["tracking"]["min_track_frames"] = 1
    cfg.data["tracking"]["min_output_frames"] = 1
    p = _build_pipeline(cfg)
    seen = _spy_driver(p)
    # Küçük araç: 120x120, kabin = 120x66 = ~7920 px = %3.4 < %10
    small = Detection(
        bbox=BBox(x1=300.0, y1=160.0, x2=420.0, y2=280.0, conf=0.9, cls="car"),
        track_id=13,
    )
    p.detector = _ScriptedDetector([[small]])
    frame = np.zeros((360, 640, 3), np.uint8)
    p.process_frame(frame, 0)
    assert seen["rois"] and seen["rois"][0] is not None
    roi_area = seen["rois"][0].shape[0] * seen["rois"][0].shape[1]
    # tam kabin alanı korunur (kırpılmadı)
    assert roi_area > 0.03 * 360 * 640
