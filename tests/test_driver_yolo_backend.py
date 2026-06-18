"""YOLO26lDriverClassifier + classifier fabrika (resolve_driver_backend) testleri.

MODEL GEREKTİRMEZ: YOLO26lDriverClassifier `object.__new__` ile kurulur, model
yerine ultralytics çıktı şeklini taklit eden sahte nesne enjekte edilir; backend
çözümleyiciler gerçek config ile (ağırlık varlığı diske göre) doğrulanır.

Kapsanan boşluklar (INCELEME test_gaps):
  - yolo.py infer: canonical eşleme, boxes None, classes filtresi, conf max
  - classifier.py resolve_driver_backend (pose/yolo/auto), _pose_weight_exists
"""

from __future__ import annotations

from aura.driver_state import classifier as clf_mod
from aura.driver_state.classifier import resolve_driver_backend
from aura.driver_state.yolo import YOLO26lDriverClassifier
from aura.schema import DriverState


class _Scalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _Box:
    def __init__(self, cls, conf):
        self.cls = _Scalar(cls)
        self.conf = _Scalar(conf)


class _Result:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeModel:
    def __init__(self, result, names):
        self._result = result
        self.names = names

    def predict(self, *_a, **_k):
        return [self._result] if self._result is not None else []


def _yolo_clf(model, classes=None):
    c = object.__new__(YOLO26lDriverClassifier)
    c.model = model
    c.path = "fake.pt"
    c.conf = 0.40
    c.imgsz = 320
    c.classes = classes or ["phone", "smoking", "seatbelt", "fatigue"]
    c.device = "cpu"
    return c


def _roi():
    import numpy as np

    return np.zeros((50, 50, 3), np.uint8)


# --------------------------------------------------------------------------- #
# YOLO26lDriverClassifier.infer
# --------------------------------------------------------------------------- #
def test_yolo_infer_canonical_mapping():
    """'cell phone' → canonical 'phone' bayrağı set edilir."""
    names = {0: "cell phone"}
    model = _FakeModel(_Result([_Box(cls=0, conf=0.8)], names), names)
    ds = _yolo_clf(model).infer(_roi())
    assert ds.phone is True
    assert ds.confidence["phone"] == 0.8


def test_yolo_infer_filters_classes_not_in_config():
    """Config classes dışı kanonik ad (ör. 'car') bayrak set etmez."""
    names = {0: "car"}
    model = _FakeModel(_Result([_Box(cls=0, conf=0.9)], names), names)
    ds = _yolo_clf(model).infer(_roi())
    assert ds.active_flags() == []


def test_yolo_infer_conf_max_merge():
    """Aynı sınıf iki kutu → en yüksek conf saklanır."""
    names = {0: "cigarette"}
    boxes = [_Box(cls=0, conf=0.5), _Box(cls=0, conf=0.9)]
    model = _FakeModel(_Result(boxes, names), names)
    ds = _yolo_clf(model).infer(_roi())
    assert ds.smoking is True
    assert ds.confidence["smoking"] == 0.9


def test_yolo_infer_boxes_none_returns_empty():
    names = {0: "phone"}
    model = _FakeModel(_Result(None, names), names)
    assert _yolo_clf(model).infer(_roi()).active_flags() == []


def test_yolo_infer_no_results_returns_empty():
    model = _FakeModel(None, {0: "phone"})
    assert _yolo_clf(model).infer(_roi()).active_flags() == []


def test_yolo_infer_none_roi_returns_empty():
    model = _FakeModel(None, {0: "phone"})
    out = _yolo_clf(model).infer(None)
    assert isinstance(out, DriverState) and out.active_flags() == []


def test_yolo_infer_names_as_list():
    """names liste olarak gelirse indeksle erişilir (dict değil)."""
    names = ["car", "cell phone"]  # index 1 → canonical 'phone'
    model = _FakeModel(_Result([_Box(cls=1, conf=0.7)], names), names)
    ds = _yolo_clf(model).infer(_roi())
    assert ds.phone is True


def test_yolo_infer_accepts_track_id_kwarg():
    """ABC sözleşmesi: track_id kabul edilir (durumsuz → yok sayılır)."""
    names = {0: "phone"}
    model = _FakeModel(_Result([_Box(cls=0, conf=0.6)], names), names)
    ds = _yolo_clf(model).infer(_roi(), track_id=42)
    assert ds.phone is True


# --------------------------------------------------------------------------- #
# resolve_driver_backend / _pose_weight_exists
# --------------------------------------------------------------------------- #
def test_resolve_backend_explicit_pose(cfg):
    cfg.data["models"]["driver_state"]["backend"] = "pose"
    assert resolve_driver_backend(cfg) == "pose"


def test_resolve_backend_explicit_yolo(cfg):
    cfg.data["models"]["driver_state"]["backend"] = "yolo"
    assert resolve_driver_backend(cfg) == "yolo"


def test_resolve_backend_auto_pose_when_weight_present(cfg, monkeypatch):
    """auto + pose ağırlığı var → pose."""
    cfg.data["models"]["driver_state"]["backend"] = "auto"
    monkeypatch.setattr(clf_mod, "_pose_weight_exists", lambda _c: True)
    assert resolve_driver_backend(cfg) == "pose"


def test_resolve_backend_auto_yolo_when_pose_absent(cfg, monkeypatch):
    """auto + pose ağırlığı yok → yolo fallback."""
    cfg.data["models"]["driver_state"]["backend"] = "auto"
    monkeypatch.setattr(clf_mod, "_pose_weight_exists", lambda _c: False)
    assert resolve_driver_backend(cfg) == "yolo"


def test_pose_weight_exists_true_when_l_pose(cfg, monkeypatch, tmp_path):
    """Yapılandırılan l-pose ağırlığı diskte → True (s fallback'e bakmaya gerek yok)."""
    wp = tmp_path / "yolo26l-pose.pt"
    wp.write_bytes(b"x")
    cfg.data["models"]["driver_state"]["pose_path"] = str(wp)
    assert clf_mod._pose_weight_exists(cfg) is True


def test_pose_weight_exists_false_when_neither(cfg):
    """Ne l-pose ne s-pose diskte → False (var olmayan yola işaret et)."""
    cfg.data["models"]["driver_state"]["pose_path"] = "weights/__nope_l_pose__.pt"
    # s-pose fallback gerçek repoda yoksa False; varsa bu test ortama bağlı kalmasın:
    import pathlib

    s_pose = pathlib.Path(clf_mod.__file__).resolve().parents[2] / "weights/yolo26s-pose.pt"
    if s_pose.exists():
        import pytest

        pytest.skip("s-pose fallback diskte var — neither senaryosu kurulamaz")
    assert clf_mod._pose_weight_exists(cfg) is False
