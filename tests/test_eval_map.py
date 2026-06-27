"""İstatistiksel mAP harness (roadguard/eval/map_eval.py) birim testleri.

ultralytics gerçek modeli yerine sahte bir ``YOLO`` enjekte ederek
``run_map``'in box.map/map50/mp/mr okuyup md+json ürettiğini doğrular;
ultralytics-yok / dosya-yok dallarında ``None`` döndüğünü kanıtlar.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from roadguard.eval.map_eval import run_map
from roadguard.eval.report import render_markdown


class _FakeBox:
    map = 0.6123
    map50 = 0.8011
    mp = 0.7502
    mr = 0.7100
    maps = [0.55, 0.67]
    ap_class_index = [0, 1]


class _FakeMetrics:
    def __init__(self):
        self.box = _FakeBox()
        self.save_dir = None
        self.names = {0: "vehicle", 1: "plate"}


class _FakeModel:
    names = {0: "vehicle", 1: "plate"}

    def __init__(self, weights):
        self._weights = weights

    def val(self, data, verbose=False):
        return _FakeMetrics()


def _inject_fake_ultralytics(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _FakeModel
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def _make_files(tmp_path):
    w = tmp_path / "fake.pt"
    w.write_bytes(b"\x00")
    d = tmp_path / "data.yaml"
    d.write_text("path: .\nval: images\nnames: [vehicle, plate]\n", encoding="utf-8")
    return w, d


def test_run_map_produces_report(monkeypatch, tmp_path):
    _inject_fake_ultralytics(monkeypatch)
    weights, data_yaml = _make_files(tmp_path)
    out = tmp_path / "eval_out"

    res = run_map(weights, data_yaml, out_dir=out)

    assert res is not None
    assert res["map50_95"] == 0.6123
    assert res["map50"] == 0.8011
    assert res["precision"] == 0.7502
    assert res["recall"] == 0.71
    # sınıf-bazlı tablo isimlerle çözüldü
    names = {row["class_name"] for row in res["per_class"]}
    assert names == {"vehicle", "plate"}

    md = (out / "map_report.md").read_text(encoding="utf-8")
    assert "İstatistiksel mAP (geniş set)" in md
    assert "0.6123" in md
    saved = json.loads((out / "map_report.json").read_text(encoding="utf-8"))
    assert saved["map50"] == 0.8011


def test_run_map_no_ultralytics_returns_none(monkeypatch, tmp_path):
    # ultralytics import edilemezse exception DEĞİL None dönmeli
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    weights, data_yaml = _make_files(tmp_path)
    res = run_map(weights, data_yaml, out_dir=tmp_path / "out")
    assert res is None
    assert not (tmp_path / "out" / "map_report.md").exists()


def test_run_map_missing_weights_returns_none(tmp_path):
    _, data_yaml = _make_files(tmp_path)
    res = run_map(tmp_path / "yok.pt", data_yaml, out_dir=tmp_path / "out")
    assert res is None


def test_run_map_missing_data_returns_none(tmp_path):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"\x00")
    res = run_map(weights, tmp_path / "yok.yaml", out_dir=tmp_path / "out")
    assert res is None


def test_run_map_val_exception_returns_none(monkeypatch, tmp_path):
    class _BoomModel:
        def __init__(self, weights):
            pass

        def val(self, data, verbose=False):
            raise RuntimeError("CUDA patladı")

    mod = types.ModuleType("ultralytics")
    mod.YOLO = _BoomModel
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    weights, data_yaml = _make_files(tmp_path)
    res = run_map(weights, data_yaml, out_dir=tmp_path / "out")
    assert res is None


def test_render_markdown_honest_note_without_map():
    # map_report yoksa render_markdown dürüst not basmalı, mAP uydurmamalı
    report = {"detectors": {}, "min_frames": 3}
    md = render_markdown(report)
    assert "İstatistiksel mAP (geniş set)" in md
    assert "Henüz üretilmedi" in md


def test_render_markdown_includes_map_when_present():
    report = {
        "detectors": {},
        "min_frames": 3,
        "map": {
            "map50_95": 0.61,
            "map50": 0.80,
            "precision": 0.75,
            "recall": 0.71,
            "weights": "w.pt",
            "data": "d.yaml",
            "pr_curve": "/tmp/PR_curve.png",
        },
    }
    md = render_markdown(report)
    assert "0.61" in md
    assert "PR eğrisi" in md


# --------------------------------------------------------------------------- #
# _class_table / _find_pr_curve dal kapsamı (sürüm-kırılgan yardımcılar)
# --------------------------------------------------------------------------- #
from roadguard.eval.map_eval import _class_table, _find_pr_curve, _safe_float  # noqa: E402


def test_safe_float_invalid_returns_none():
    assert _safe_float("abc") is None
    assert _safe_float(None) is None
    assert _safe_float(0.61239) == 0.6124  # 4 hane yuvarlama


def test_class_table_no_box_returns_empty():
    class _M:
        box = None

    assert _class_table(_M(), {0: "car"}) == []


def test_class_table_no_maps_returns_empty():
    class _Box:
        maps = None
        ap_class_index = [0]

    class _M:
        box = _Box()

    assert _class_table(_M(), {0: "car"}) == []


def test_class_table_index_none_uses_enumerate():
    # ap_class_index None → enumerate(maps) ile pozisyonel class_id (else-dalı)
    class _Box:
        maps = [0.5, 0.7]
        ap_class_index = None

    class _M:
        box = _Box()

    table = _class_table(_M(), {0: "car", 1: "plate"})
    assert [r["class_id"] for r in table] == [0, 1]
    assert [r["class_name"] for r in table] == ["car", "plate"]
    assert table[0]["map50_95"] == 0.5


def test_class_table_index_none_unknown_name_falls_back_to_str():
    class _Box:
        maps = [0.5]
        ap_class_index = None

    class _M:
        box = _Box()

    table = _class_table(_M(), None)  # names yok → str(ci)
    assert table[0]["class_name"] == "0"


def test_class_table_index_out_of_range_caught():
    # maps[ci] IndexError → except dalı: boş tablo (raporu düşürmez)
    class _Box:
        maps = [0.5]  # tek eleman
        ap_class_index = [0, 5]  # 5 sınırların dışında

    class _M:
        box = _Box()

    assert _class_table(_M(), {0: "car"}) == []


def test_find_pr_curve_none_save_dir():
    assert _find_pr_curve(None) is None


def test_find_pr_curve_missing_dir():
    assert _find_pr_curve("/yok/olmayan/dizin/xyz") is None


def test_find_pr_curve_named_file(tmp_path):
    (tmp_path / "PR_curve.png").write_bytes(b"\x89PNG")
    found = _find_pr_curve(tmp_path)
    assert found is not None and found.endswith("PR_curve.png")


def test_find_pr_curve_glob_fallback(tmp_path):
    # standart adlar yok ama *PR_curve.png glob'u bulur
    (tmp_path / "BoxF1PR_curve.png").write_bytes(b"\x89PNG")
    found = _find_pr_curve(tmp_path)
    assert found is not None and found.endswith("PR_curve.png")


def test_find_pr_curve_no_match(tmp_path):
    (tmp_path / "results.png").write_bytes(b"\x89PNG")
    assert _find_pr_curve(tmp_path) is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
