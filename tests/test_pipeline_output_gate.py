"""SORUN 2 regresyon kapanı: track_id=-1 sızıntısı + phantom track çıktı bastırma.

Model/ağırlık gerektirmez (CI-uyumlu): pipeline mock modda kurulur, dedektör
scriptli bir sahte dedektörle değiştirilir → çıktı kapısı davranışı izole edilir.
"""

from __future__ import annotations

import numpy as np

from aura.detection.detector import Detection
from aura.schema import BBox


class _ScriptedDetector:
    """Kare başına önceden belirlenmiş Detection listesi döndüren sahte dedektör.

    Pipeline'ın okuduğu sözleşmeyi taşır: detect(frame), close(), last_persons/
    last_signs/last_aux. Ağır model yok — çıktı kapısı saf mantığı test edilir.
    """

    def __init__(self, script: list[list[Detection]]):
        self._script = script
        self._i = 0
        self.last_persons: list = []
        self.last_signs: list = []
        self.last_aux: list = []

    def detect(self, frame):
        dets = self._script[self._i] if self._i < len(self._script) else []
        self._i += 1
        return dets

    def close(self):  # pragma: no cover - kaynak yok
        pass


def _det(track_id, cls="car"):
    # Kare ortasında makul boyutlu bir araç kutusu (sweet-spot içinde).
    bbox = BBox(x1=240.0, y1=140.0, x2=400.0, y2=300.0, conf=0.9, cls=cls)
    return Detection(bbox=bbox, track_id=track_id)


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
