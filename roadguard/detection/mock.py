"""Deterministik mock dedektör — model/ağırlık olmadan uçtan-uca çalışma için.

Sentetik test videosundaki parlak araç bloklarını eşikleme + kontur ile bulur ve
basit IoU tabanlı bir takipçi ile ID atar (ByteTrack'in mock muadili). Bu sayede
tüm pipeline, dashboard ve testler gerçek ağırlık olmadan da çalışır. `ai_mode=mock`
veya `auto` (ağırlık yok) iken devreye girer.
"""

from __future__ import annotations

import cv2
import numpy as np

from roadguard.detection.detector import Detection, Detector, Person, Sign, crop_rois
from roadguard.schema import BBox


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


class SimpleIoUTracker:
    """Greedy IoU eşleme ile kalıcı track ID — ByteTrack'in hafif muadili (mock)."""

    def __init__(self, iou_thr: float = 0.3, max_age: int = 10):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, boxes: list[tuple]) -> list[tuple[int, tuple]]:
        assigned: dict[int, int] = {}
        used: set[int] = set()
        for tid, tr in list(self.tracks.items()):
            best, best_iou = -1, self.iou_thr
            for i, b in enumerate(boxes):
                if i in used:
                    continue
                v = _iou(tr["bbox"], b)
                if v >= best_iou:
                    best, best_iou = i, v
            if best >= 0:
                assigned[best] = tid
                self.tracks[tid] = {"bbox": boxes[best], "age": 0}
                used.add(best)
            else:
                tr["age"] += 1
        for tid in [t for t, tr in self.tracks.items() if tr["age"] > self.max_age]:
            del self.tracks[tid]
        result = []
        for i, b in enumerate(boxes):
            tid = assigned.get(i)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": b, "age": 0}
            result.append((tid, b))
        return result


class MockDetector(Detector):
    def __init__(self, cfg):
        super().__init__()  # last_persons/last_signs/last_aux'u örnek-seviyesinde kur
        self.conf = float(cfg.get("models.detector.conf", 0.35))
        classes = cfg.get("models.detector.vehicle_classes", ["car"])
        self.cls0 = classes[0] if classes else "car"
        self.bright_thr = int(cfg.get("models.detector.mock_bright_threshold", 90))
        self.min_area = int(cfg.get("models.detector.mock_min_area", 300))
        self.tracker = SimpleIoUTracker()
        # Mock'ta gerçek kişi tespiti yok; demo/sunum için sentetik sürücü üretilebilir.
        self.synthetic_person = bool(cfg.get("driver_lock.mock_synthetic_person", False))
        self.last_persons: list[Person] = []
        # Mock'ta gerçek tabela tespiti yok; ağırlıksız demo için sentetik hız-limiti üretilebilir.
        self.sign_synthetic = bool(cfg.get("sign.mock_synthetic", False))
        self.sign_synth_limit = int(cfg.get("sign.mock_speed_limit", 50))
        self.last_signs: list[Sign] = []
        # Morfoloji çekirdeği sabittir → bir kez kur (her karede yeniden allocate yok).
        self._morph_kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.bright_thr, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[tuple] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area or w < 15 or h < 12:
                continue
            ar = w / max(h, 1)
            if ar < 0.2 or ar > 6:  # şerit çizgisi gibi ince-uzun yapıları ele
                continue
            boxes.append((x, y, x + w, y + h))

        dets = []
        self.last_persons = []
        self.last_signs = []
        if self.sign_synthetic:
            self.last_signs.append(self._synthetic_sign(frame))
        for tid, (x1, y1, x2, y2) in self.tracker.update(boxes):
            bbox = BBox(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), conf=0.9, cls=self.cls0
            )
            d = Detection(bbox=bbox, track_id=tid)
            d.cabin_roi, d.plate_roi = crop_rois(frame, bbox)
            dets.append(d)
            if self.synthetic_person:
                # Kabinin sağ-alt çeyreğine deterministik bir sürücü yerleştir;
                # kişi ID'si araç ID'sine bağlı sabit → 5 kare sonra kilit gözlemlenir.
                self.last_persons.append(self._synthetic_driver(bbox, tid))
        return dets

    def _synthetic_driver(self, v: BBox, vehicle_tid: int) -> Person:
        """Araç kutusunun sağ-alt çeyreğinde deterministik sentetik sürücü (mock demo)."""
        cx = v.x1 + v.width * 0.70
        cy = v.y1 + v.height * 0.70
        hw, hh = v.width * 0.12, v.height * 0.12
        pbox = BBox(x1=cx - hw, y1=cy - hh, x2=cx + hw, y2=cy + hh, conf=0.9, cls="person")
        return Person(bbox=pbox, track_id=100000 + vehicle_tid)

    def _synthetic_sign(self, frame: np.ndarray) -> Sign:
        """Karenin sağ-üst köşesinde sabit bir hız-limiti tabelası (ağırlıksız demo)."""
        h, w = frame.shape[:2]
        cls = f"speed_limit_{self.sign_synth_limit}"
        bbox = BBox(x1=float(w - 90), y1=10.0, x2=float(w - 10), y2=70.0, conf=0.95, cls=cls)
        return Sign(bbox=bbox, cls=cls, track_id=None)
