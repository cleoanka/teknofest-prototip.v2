"""YOLO26Detector.detect() + _dedup() — gerçek model GEREKMEZ (sahte model.track).

Gerçek ultralytics ağırlığı yüklemeden, `model.track`'in döndürdüğü sonucu taklit
eden hafif sahteler (FakeBox/FakeResult) ile detect()'in tüm yönlendirme dallarını
(person/sign/aux/vehicle), kanonik ad eşlemesini, track_id None yolunu, boş/None
sonuç erken-dönüşlerini ve _dedup() kopya-bastırma davranışını sınar. Detector
nesnesi __new__ ile kurulur ve __init__ alanları elle set edilir (YOLO yüklenmez).
"""

from __future__ import annotations

import numpy as np

from aura.detection.yolo import YOLO26Detector
from aura.schema import BBox


# --------------------------------------------------------------------------- #
# model.track çıktısını taklit eden hafif sahteler
# --------------------------------------------------------------------------- #
class _Scalar:
    """`.item()` destekleyen tek-değer sarmalayıcı (b.cls/b.conf/b.id muadili)."""

    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _Row:
    """`b.xyxy[0].tolist()` muadili."""

    def __init__(self, xyxy):
        self._xyxy = xyxy

    def __getitem__(self, i):
        assert i == 0
        return self

    def tolist(self):
        return list(self._xyxy)


class FakeBox:
    def __init__(self, cls_idx, xyxy, conf, tid=None):
        self.cls = _Scalar(cls_idx)
        self.conf = _Scalar(conf)
        self.id = _Scalar(tid) if tid is not None else None
        self.xyxy = _Row(xyxy)


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class FakeModel:
    """`track()` çağrısı önceden kurulmuş results'ı döndürür (frame'i yok sayar)."""

    def __init__(self, results, names):
        self._results = results
        self.names = names

    def track(self, *a, **k):
        return self._results


def _make_detector(
    results,
    names,
    *,
    vehicle_classes=("car", "truck", "bus"),
    person_classes=("person",),
    sign_classes=("speed_limit_50",),
    aux_classes=("phone", "smoking"),
    sign_enabled=True,
    dedup_iou=0.80,
):
    """__init__'i (YOLO yükler) atlayarak detect() için minimal dedektör kur."""
    det = YOLO26Detector.__new__(YOLO26Detector)
    det.model = FakeModel(results, names)
    det.conf = 0.1
    det.iou = 0.45
    det.imgsz = 640
    det.tracker = "bytetrack"
    det.tracker_yaml = "bytetrack.yaml"
    det.vehicle_classes = set(vehicle_classes)
    det.person_classes = set(person_classes)
    det.sign_enabled = sign_enabled
    det.sign_classes = set(sign_classes)
    det.aux_classes = set(aux_classes)
    det.dedup_iou = dedup_iou
    det.device = "cpu"
    det.last_persons = []
    det.last_signs = []
    det.last_aux = []
    return det


_FRAME = np.zeros((200, 200, 3), np.uint8)


def test_detect_routes_each_category():
    names = {0: "car", 1: "person", 2: "speed_limit_50", 3: "cell phone"}
    boxes = [
        FakeBox(0, (10, 10, 60, 60), 0.9, tid=1),  # araç
        FakeBox(1, (70, 70, 90, 120), 0.8, tid=2),  # kişi
        FakeBox(2, (150, 5, 190, 40), 0.95, tid=3),  # tabela
        FakeBox(3, (20, 20, 30, 30), 0.7, tid=4),  # 'cell phone' → 'phone' (aux)
    ]
    det = _make_detector([FakeResult(names, boxes)], names)
    dets = det.detect(_FRAME)
    assert len(dets) == 1 and dets[0].track_id == 1 and dets[0].bbox.cls == "car"
    assert dets[0].cabin_roi is not None and dets[0].plate_roi is not None
    assert len(det.last_persons) == 1 and det.last_persons[0].track_id == 2
    assert len(det.last_signs) == 1 and det.last_signs[0].cls == "speed_limit_50"
    # 'cell phone' kanonikleştirilip aux'a düşmeli
    assert len(det.last_aux) == 1 and det.last_aux[0].cls == "phone"


def test_detect_track_id_none_when_box_id_missing():
    names = {0: "car"}
    det = _make_detector([FakeResult(names, [FakeBox(0, (0, 0, 50, 50), 0.9, tid=None)])], names)
    dets = det.detect(_FRAME)
    assert len(dets) == 1 and dets[0].track_id is None


def test_detect_empty_results_returns_empty():
    det = _make_detector([], {0: "car"})
    assert det.detect(_FRAME) == []
    assert det.last_persons == [] and det.last_signs == [] and det.last_aux == []


def test_detect_boxes_none_returns_empty():
    names = {0: "car"}
    det = _make_detector([FakeResult(names, None)], names)
    assert det.detect(_FRAME) == []


def test_detect_empty_vehicle_classes_treats_unknown_as_vehicle():
    # vehicle_classes boş → kişi/tabela/aux DIŞI her şey araç sayılır.
    names = {0: "forklift"}
    det = _make_detector(
        [FakeResult(names, [FakeBox(0, (0, 0, 40, 40), 0.9, tid=5)])],
        names,
        vehicle_classes=(),
    )
    dets = det.detect(_FRAME)
    assert len(dets) == 1 and dets[0].bbox.cls == "forklift"


def test_detect_sign_disabled_routes_sign_to_vehicle_or_dropped():
    # sign_enabled=False → tabela sınıfı 'is_sign' olamaz; vehicle_classes'ta da yoksa
    # ve aux/person değilse, vehicle_classes boş-olmadığından elenir (atlanır).
    names = {0: "speed_limit_50"}
    det = _make_detector(
        [FakeResult(names, [FakeBox(0, (0, 0, 40, 40), 0.9, tid=1)])],
        names,
        sign_enabled=False,
        vehicle_classes=("car",),
    )
    dets = det.detect(_FRAME)
    assert dets == [] and det.last_signs == []


def test_detect_names_as_list():
    # names liste olduğunda indeksle erişim dalı.
    names = ["car", "person"]
    det = _make_detector([FakeResult(names, [FakeBox(0, (0, 0, 50, 50), 0.9, tid=1)])], names)
    dets = det.detect(_FRAME)
    assert len(dets) == 1 and dets[0].bbox.cls == "car"


# --------------------------------------------------------------------------- #
# _dedup()
# --------------------------------------------------------------------------- #
def _det(x1, y1, x2, y2, conf, cls="car"):
    from aura.detection.detector import Detection

    return Detection(bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, cls=cls))


def test_dedup_suppresses_high_iou_keeps_best_conf():
    det = _make_detector([], {0: "car"}, dedup_iou=0.80)
    a = _det(0, 0, 100, 100, 0.95)
    b = _det(2, 2, 100, 100, 0.60)  # a ile ~%96 IoU → bastırılır, yüksek conf kalır
    kept = det._dedup([b, a])
    assert len(kept) == 1 and kept[0].bbox.conf == 0.95


def test_dedup_class_independent():
    # Farklı sınıf ('car' vs 'truck') ama yüksek IoU → yine bastırılır (sınıf-bağımsız).
    det = _make_detector([], {0: "car"}, dedup_iou=0.80)
    a = _det(0, 0, 100, 100, 0.9, cls="car")
    b = _det(1, 1, 99, 99, 0.5, cls="truck")
    kept = det._dedup([a, b])
    assert len(kept) == 1 and kept[0].bbox.cls == "car"


def test_dedup_keeps_disjoint_boxes():
    det = _make_detector([], {0: "car"}, dedup_iou=0.80)
    a = _det(0, 0, 50, 50, 0.9)
    b = _det(100, 100, 150, 150, 0.8)  # örtüşme yok
    kept = det._dedup([a, b])
    assert len(kept) == 2


def test_dedup_disabled_when_iou_threshold_ge_one():
    det = _make_detector([], {0: "car"}, dedup_iou=1.0)
    a = _det(0, 0, 100, 100, 0.9)
    b = _det(0, 0, 100, 100, 0.8)  # birebir aynı kutu
    assert det._dedup([a, b]) == [a, b]  # devre dışı → ikisi de kalır (sıra korunur)


def test_dedup_short_circuits_single_det():
    det = _make_detector([], {0: "car"}, dedup_iou=0.80)
    a = _det(0, 0, 100, 100, 0.9)
    assert det._dedup([a]) == [a]


def test_iou_precomputed_area_matches_property():
    # Mikro-opt davranış-koruma: önceden hesaplanmış alan ile property yolu aynı sonucu verir.
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=5, y1=5, x2=15, y2=15)
    via_prop = YOLO26Detector._iou(a, b)
    via_area = YOLO26Detector._iou(a, b, a.width * a.height, b.width * b.height)
    assert via_prop == via_area
    assert YOLO26Detector._iou(a, b, 100.0, 100.0) > 0  # örtüşme var
