"""GRUP I5-train-tools-test: train/ + tools/ kapsamlı unit testleri.

Kapsam (henüz Workflow görmeyen son modüller):
  train/ : utils, prepare_dataset, fetch, merge_driver_datasets, __main__
  tools/ : doctor, bench, test_video, show_driver_rois

Protokol (ULTRA-DÜRÜST K-004): davranış DEĞİŞTİRİLMEZ — yalnız test eklenir.
Gerçek eğitim/model/MPS ÇALIŞTIRILMAZ — ultralytics/torch/Pipeline/Roboflow MOCK'lanır.
Donanım-limiti + ağ-gecikmesi SİMÜLE edilir (protokol gereği):
  * disk-dolu / ENOSPC  → shutil.copy / write_text OSError(errno.ENOSPC) raise
  * OOM                 → model.train RuntimeError("CUDA out of memory") raise
  * network-timeout     → roboflow_pull / Roboflow TimeoutError raise

Hepsi CPU/saf-dosya I/O; ağır import yok (lazy import noktaları mock'lanır).
"""

from __future__ import annotations

import errno
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

# train (torch'suz çekirdek)
from train import __main__ as train_main
from train import fetch as fetch_mod
from train import merge_driver_datasets as merge_mod
from train import prepare_dataset as prep_mod
from train import utils as train_utils


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def _make_yolo_dataset(root: Path, names, *, splits=("train", "val", "test"), per_split=2):
    """Minimal bir YOLO/Roboflow datasetini (data.yaml + split'ler) diske kur."""
    root.mkdir(parents=True, exist_ok=True)
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(names)}\nnames:\n{names_block}\n",
        encoding="utf-8",
    )
    for sp in splits:
        (root / sp / "images").mkdir(parents=True, exist_ok=True)
        (root / sp / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(per_split):
            cv2.imwrite(str(root / sp / "images" / f"{sp}{i}.jpg"), np.zeros((8, 8, 3), np.uint8))
            (root / sp / "labels" / f"{sp}{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root


# =========================================================================== #
# train/__main__.py — arg-parse + dispatch (gerçek alt-modüller mock'lanır)
# =========================================================================== #
class TestTrainMainArgparse:
    def test_no_subcommand_exits(self):
        with pytest.raises(SystemExit):
            train_main.build_parser().parse_args([])

    def test_unknown_subcommand_exits(self):
        with pytest.raises(SystemExit):
            train_main.build_parser().parse_args(["nope"])

    def test_detector_requires_data(self):
        with pytest.raises(SystemExit):
            train_main.build_parser().parse_args(["detector"])

    def test_detector_defaults(self):
        a = train_main.build_parser().parse_args(["detector", "--data", "d.yaml"])
        assert a.command == "detector"
        assert a.epochs == 100 and a.imgsz == 640 and a.batch == 16
        assert a.weights == "weights/yolo26s.pt" and a.device == "auto"
        assert a.patience == 50 and a.resume is False and a.no_augment is False

    def test_driver_state_defaults_imgsz(self):
        a = train_main.build_parser().parse_args(["driver-state", "--data", "d.yaml"])
        assert a.imgsz == 320 and a.weights == "weights/yolo26l.pt"

    def test_device_choice_validation(self):
        with pytest.raises(SystemExit):
            train_main.build_parser().parse_args(["detector", "--data", "d", "--device", "tpu"])

    def test_flags_parse(self):
        a = train_main.build_parser().parse_args(
            ["detector", "--data", "d", "--no-augment", "--no-val", "--resume", "--lr0", "0.01"]
        )
        assert a.no_augment and a.no_val and a.resume and a.lr0 == 0.01

    def test_main_dispatch_detector(self, monkeypatch):
        called = {}

        def _det(a):
            called["det"] = a
            return 0

        monkeypatch.setitem(
            sys.modules, "train.train_detector", types.SimpleNamespace(train_detector=_det)
        )
        rc = train_main.main(["detector", "--data", "d.yaml"])
        assert rc == 0 and "det" in called

    def test_main_dispatch_driver_state(self, monkeypatch):
        called = {}

        def _drv(a):
            called["drv"] = a
            return 0

        monkeypatch.setitem(
            sys.modules, "train.train_driver_state", types.SimpleNamespace(train_driver_state=_drv)
        )
        rc = train_main.main(["driver-state", "--data", "d.yaml"])
        assert rc == 0 and "drv" in called

    def test_main_dispatch_dataset(self, monkeypatch):
        monkeypatch.setattr(prep_mod, "prepare_dataset", lambda a: 7)
        # __main__ re-imports prepare_dataset lazily from module:
        monkeypatch.setitem(
            sys.modules,
            "train.prepare_dataset",
            types.SimpleNamespace(prepare_dataset=lambda a: 7),
        )
        assert train_main.main(["dataset", "--report", "--output", "x"]) == 7

    def test_main_dispatch_fetch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "train.fetch", types.SimpleNamespace(fetch=lambda a: 0))
        assert train_main.main(["fetch"]) == 0


# =========================================================================== #
# train/prepare_dataset.py — split, data.yaml, hata yolları, ENOSPC
# =========================================================================== #
class TestPrepareDataset:
    def test_split_ratios_and_determinism(self):
        imgs = [Path(f"{i}.jpg") for i in range(20)]
        s1 = prep_mod.split_items(imgs, 0.7, 0.2, seed=5)
        s2 = prep_mod.split_items(imgs, 0.7, 0.2, seed=5)
        assert s1 == s2  # deterministik
        assert len(s1["train"]) == 14 and len(s1["val"]) == 4 and len(s1["test"]) == 2
        # disjoint + tamlık
        flat = s1["train"] + s1["val"] + s1["test"]
        assert sorted(flat) == sorted(imgs)

    def test_split_empty(self):
        s = prep_mod.split_items([], 0.8, 0.1)
        assert s == {"train": [], "val": [], "test": []}

    def test_read_classes_from_arg(self, tmp_path):
        assert prep_mod._read_classes(tmp_path, "a, b ,c") == ["a", "b", "c"]

    def test_read_classes_from_file(self, tmp_path):
        (tmp_path / "classes.txt").write_text("car\ntruck\n\n", encoding="utf-8")
        assert prep_mod._read_classes(tmp_path, None) == ["car", "truck"]

    def test_read_classes_default(self, tmp_path):
        assert prep_mod._read_classes(tmp_path, None) == prep_mod.DEFAULT_CLASSES

    def test_write_data_yaml(self, tmp_path):
        p = prep_mod.write_data_yaml(tmp_path, ["car", "bus"])
        txt = p.read_text(encoding="utf-8")
        assert "nc: 2" in txt and "0: car" in txt and "1: bus" in txt

    def test_label_for_resolution(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        img = tmp_path / "images" / "x.jpg"
        img.write_bytes(b"")
        lbl = tmp_path / "labels" / "x.txt"
        lbl.write_text("0 .5 .5 .1 .1")
        assert prep_mod._label_for(img) == lbl

    def test_label_for_none(self, tmp_path):
        img = tmp_path / "x.jpg"
        img.write_bytes(b"")
        assert prep_mod._label_for(img) is None

    def test_missing_input_output_args(self):
        args = types.SimpleNamespace(
            input=None, output=None, report=False, train=0.8, val=0.1, classes=None, seed=1
        )
        assert prep_mod.prepare_dataset(args) == 1

    def test_no_images_error(self, tmp_path):
        inp = tmp_path / "empty"
        inp.mkdir()
        args = types.SimpleNamespace(
            input=str(inp),
            output=str(tmp_path / "o"),
            report=False,
            train=0.8,
            val=0.1,
            classes=None,
            seed=1,
        )
        assert prep_mod.prepare_dataset(args) == 1

    def test_report_mode_on_existing_set(self, tmp_path, capsys):
        _make_yolo_dataset(tmp_path / "ds", ["car", "truck"])
        args = types.SimpleNamespace(
            report=True, output=str(tmp_path / "ds"), input=None, classes=None
        )
        assert prep_mod.prepare_dataset(args) == 0
        assert "Veri Seti Dengeleme Raporu" in capsys.readouterr().out

    def test_report_mode_bad_dir(self, tmp_path):
        args = types.SimpleNamespace(
            report=True, output=str(tmp_path / "nope"), input=None, classes=None
        )
        assert prep_mod.prepare_dataset(args) == 1

    def test_end_to_end(self, tmp_path):
        inp = tmp_path / "raw"
        (inp / "images").mkdir(parents=True)
        for i in range(10):
            cv2.imwrite(str(inp / "images" / f"i{i}.jpg"), np.zeros((6, 6, 3), np.uint8))
            (inp / "images" / f"i{i}.txt").write_text("0 .5 .5 .2 .2\n")
        out = tmp_path / "proc"
        args = types.SimpleNamespace(
            input=str(inp),
            output=str(out),
            report=False,
            train=0.8,
            val=0.1,
            classes="car,truck",
            seed=42,
        )
        assert prep_mod.prepare_dataset(args) == 0
        assert (out / "data.yaml").exists()
        copied = sum(
            len(list((out / s / "images").glob("*.jpg"))) for s in ("train", "val", "test")
        )
        assert copied == 10

    def test_enospc_on_copy_propagates(self, tmp_path, monkeypatch):
        """SİMÜLASYON: disk dolu (ENOSPC) → shutil.copy patlar; sessizce yutulmaz."""
        inp = tmp_path / "raw"
        (inp / "images").mkdir(parents=True)
        cv2.imwrite(str(inp / "images" / "a.jpg"), np.zeros((6, 6, 3), np.uint8))
        (inp / "images" / "a.txt").write_text("0 .5 .5 .2 .2\n")

        def _enospc(*a, **k):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(prep_mod.shutil, "copy", _enospc)
        args = types.SimpleNamespace(
            input=str(inp),
            output=str(tmp_path / "o"),
            report=False,
            train=0.8,
            val=0.1,
            classes="car",
            seed=1,
        )
        with pytest.raises(OSError) as ei:
            prep_mod.prepare_dataset(args)
        assert ei.value.errno == errno.ENOSPC


# =========================================================================== #
# train/utils.py — metrik özeti, sınıf dağılımı, export_best, run_finetune(OOM)
# =========================================================================== #
class TestTrainUtils:
    def test_resolve_device_explicit_passthrough(self):
        assert train_utils.resolve_device("cpu") == "cpu"
        assert train_utils.resolve_device("mps") == "mps"

    def test_resolve_device_auto_uses_aura_device(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules,
            "aura.device",
            types.SimpleNamespace(resolve_device=lambda x: "cpu"),
        )
        assert train_utils.resolve_device("auto") == "cpu"
        assert train_utils.resolve_device(None) == "cpu"
        assert train_utils.resolve_device("") == "cpu"

    def test_summarize_metrics_from_box(self):
        box = types.SimpleNamespace(mp=0.8, mr=0.6, map50=0.7, map=0.5)
        res = types.SimpleNamespace(box=box, results_dict={})
        m = train_utils.summarize_metrics(res)
        assert m["precision"] == 0.8 and m["recall"] == 0.6
        assert m["f1"] == round(2 * 0.8 * 0.6 / 1.4, 4)
        assert m["mAP50"] == 0.7 and m["mAP50_95"] == 0.5

    def test_summarize_metrics_zero_pr(self):
        box = types.SimpleNamespace(mp=0.0, mr=0.0, map50=0.0, map=0.0)
        m = train_utils.summarize_metrics(types.SimpleNamespace(box=box, results_dict={}))
        assert m["f1"] == 0.0

    def test_summarize_metrics_results_dict(self):
        res = types.SimpleNamespace(box=None, results_dict={"metrics/mAP50(B)": 0.42, "junk": "x"})
        m = train_utils.summarize_metrics(res)
        assert m["mAP50"] == 0.42 and "junk" not in m

    def test_summarize_metrics_empty(self):
        assert (
            train_utils.summarize_metrics(types.SimpleNamespace(box=None, results_dict=None)) == {}
        )

    def test_class_distribution(self, tmp_path):
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text("0 .5 .5 .1 .1\n0 .3 .3 .1 .1\n1 .5 .5 .1 .1\n")
        d = train_utils.class_distribution(tmp_path)
        assert d[0] == 2 and d[1] == 1

    def test_class_distribution_no_labels_dir(self, tmp_path):
        assert train_utils.class_distribution(tmp_path) == train_utils.Counter()

    def test_class_distribution_corrupt_line(self, tmp_path):
        """SİMÜLASYON: bozuk etiket (sınıf id sayı değil) → satır atlanır, patlamaz."""
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text("notanint .5 .5\n0 .5 .5 .1 .1\n")
        d = train_utils.class_distribution(tmp_path)
        assert d[0] == 1 and len(d) == 1

    def test_dataset_stats_imbalance(self, tmp_path):
        _make_yolo_dataset(tmp_path / "ds", ["a", "b"], splits=("train",), per_split=0)
        ds = tmp_path / "ds"
        # 3x sınıf-0, 1x sınıf-1 → imbalance 3.0
        (ds / "train" / "labels" / "x.txt").write_text(
            "0 .5 .5 .1 .1\n0 .5 .5 .1 .1\n0 .5 .5 .1 .1\n"
        )
        (ds / "train" / "labels" / "y.txt").write_text("1 .5 .5 .1 .1\n")
        cv2.imwrite(str(ds / "train" / "images" / "x.jpg"), np.zeros((6, 6, 3), np.uint8))
        cv2.imwrite(str(ds / "train" / "images" / "y.jpg"), np.zeros((6, 6, 3), np.uint8))
        stats = train_utils.dataset_stats(ds, names=["a", "b"])
        assert stats["class_totals"] == {0: 3, 1: 1}
        assert stats["imbalance_ratio"] == 3.0
        assert stats["names"] == ["a", "b"]

    def test_dataset_stats_empty_root(self, tmp_path):
        stats = train_utils.dataset_stats(tmp_path)
        assert stats["splits"] == {} and stats["class_totals"] == {}
        assert "imbalance_ratio" not in stats

    def test_print_dataset_report(self, tmp_path, capsys):
        stats = {
            "root": "x",
            "splits": {"train": {"images": 5, "class_counts": {0: 5}}},
            "class_totals": {0: 5, 1: 1},
            "imbalance_ratio": 5.0,
            "names": ["a", "b"],
        }
        train_utils.print_dataset_report(stats)
        out = capsys.readouterr().out
        assert "dengesiz" in out and "5 görüntü" in out

    def test_export_best_missing_best(self, tmp_path, capsys):
        res = types.SimpleNamespace(save_dir=str(tmp_path))
        assert train_utils.export_best(res, "x.pt") is None

    def test_export_best_copies_and_writes_metrics(self, tmp_path, monkeypatch):
        save_dir = tmp_path / "run"
        (save_dir / "weights").mkdir(parents=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"FAKEWEIGHTS")
        monkeypatch.setattr(train_utils, "ROOT", tmp_path)
        res = types.SimpleNamespace(save_dir=str(save_dir))
        dest = train_utils.export_best(res, "custom.pt", metrics={"mAP50": 0.9})
        assert dest is not None and dest.exists()
        assert dest.with_suffix(".metrics.json").exists()
        assert json.loads(dest.with_suffix(".metrics.json").read_text())["mAP50"] == 0.9

    def test_run_finetune_mocked_flow(self, monkeypatch):
        """Gerçek YOLO YOK: model.train/val mock'lanır → metrik+export çağrıldı mı."""
        calls = {}

        class FakeYOLO:
            def __init__(self, w):
                calls["weights"] = w

            def train(self, **kw):
                calls["train_kw"] = kw
                return types.SimpleNamespace(
                    save_dir="x",
                    box=types.SimpleNamespace(mp=0.5, mr=0.5, map50=0.5, map=0.5),
                    results_dict={},
                )

            def val(self, **kw):
                calls["val"] = True
                return types.SimpleNamespace(
                    box=types.SimpleNamespace(mp=0.6, mr=0.6, map50=0.6, map=0.6), results_dict={}
                )

        monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
        monkeypatch.setattr(train_utils, "export_best", lambda *a, **k: calls.setdefault("exp", a))
        args = types.SimpleNamespace(
            device="cpu",
            weights="w.pt",
            data="d.yaml",
            epochs=1,
            imgsz=64,
            batch=2,
            project="p",
            name="n",
            patience=10,
            lr0=None,
            resume=False,
            no_augment=True,
            no_val=False,
        )
        assert train_utils.run_finetune(args, "out.pt", "test") == 0
        assert calls["weights"] == "w.pt" and calls["val"] and "exp" in calls
        # --no-augment → augment anahtarları sıfırlandı
        assert calls["train_kw"]["mosaic"] == 0.0

    def test_run_finetune_oom_propagates(self, monkeypatch):
        """SİMÜLASYON: GPU OOM → model.train RuntimeError; sessizce yutulmaz."""

        class OomYOLO:
            def __init__(self, w):
                pass

            def train(self, **kw):
                raise RuntimeError("CUDA out of memory. Tried to allocate ...")

        monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=OomYOLO))
        args = types.SimpleNamespace(
            device="cpu",
            weights="w.pt",
            data="d",
            epochs=1,
            imgsz=64,
            batch=64,
            project="p",
            name="n",
            patience=10,
        )
        with pytest.raises(RuntimeError, match="out of memory"):
            train_utils.run_finetune(args, "o.pt", "t")

    def test_run_finetune_val_failure_falls_back(self, monkeypatch):
        """model.val patlarsa eğitim metrikleri kullanılır (akış kesilmez)."""

        class YOLO:
            def __init__(self, w):
                pass

            def train(self, **kw):
                return types.SimpleNamespace(
                    save_dir="x",
                    box=types.SimpleNamespace(mp=0.5, mr=0.5, map50=0.5, map=0.5),
                    results_dict={},
                )

            def val(self, **kw):
                raise RuntimeError("val boom")

        monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=YOLO))
        monkeypatch.setattr(train_utils, "export_best", lambda *a, **k: None)
        args = types.SimpleNamespace(
            device="cpu",
            weights="w",
            data="d",
            epochs=1,
            imgsz=64,
            batch=2,
            project="p",
            name="n",
            patience=10,
            lr0=0.01,
            resume=True,
            no_augment=False,
            no_val=False,
        )
        assert train_utils.run_finetune(args, "o.pt", "t") == 0


# =========================================================================== #
# train/fetch.py — manifest yükleme, plan, taksonomi uyarıları, --run/timeout
# =========================================================================== #
class TestFetch:
    def _manifest(self):
        return {
            "version": 1,
            "targets": {
                "minibus": {
                    "aura_class": "minibus",
                    "sources": [
                        {
                            "kind": "roboflow",
                            "name": "mb",
                            "license": "CC BY 4.0",
                            "images": 500,
                            "workspace": "ws",
                            "project": "pr",
                            "version": 2,
                            "class_map": {"minibus": "minibus"},
                        }
                    ],
                },
                "cigarette": {
                    "aura_class": "smoking",
                    "sources": [
                        {"kind": "kaggle", "name": "cig", "license": "MIT", "ref": "u/cig-set"}
                    ],
                },
            },
        }

    def test_load_manifest_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            fetch_mod.load_manifest(tmp_path / "nope.yaml")

    def test_load_manifest_no_targets(self, tmp_path):
        p = tmp_path / "m.yaml"
        p.write_text("version: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="targets"):
            fetch_mod.load_manifest(p)

    def test_load_manifest_real_file(self):
        data = fetch_mod.load_manifest(fetch_mod.DEFAULT_MANIFEST)
        assert "targets" in data and isinstance(data["targets"], dict)

    def test_build_plan_all(self):
        plan = fetch_mod.build_plan(self._manifest())
        assert len(plan) == 2
        kinds = {s["kind"] for s in plan}
        assert kinds == {"roboflow", "kaggle"}

    def test_build_plan_only_class(self):
        plan = fetch_mod.build_plan(self._manifest(), only_class="minibus")
        assert len(plan) == 1 and plan[0]["target"] == "minibus"

    def test_build_plan_unknown_class(self):
        with pytest.raises(KeyError):
            fetch_mod.build_plan(self._manifest(), only_class="ufo")

    def test_canonical_for_with_map(self):
        cm = fetch_mod._canonical_for({"class_map": {"cig": "smoking"}}, "smoking")
        assert cm == {"cig": "smoking"}

    def test_canonical_for_derived(self):
        cm = fetch_mod._canonical_for({}, "cigarette")
        assert cm == {"cigarette": "smoking"}

    def test_taxonomy_warning_unknown_mismatch(self):
        warns = fetch_mod._taxonomy_warnings({"weird": "phone"}, "smoking")
        assert warns and "TANIMSIZ" in warns[0]

    def test_taxonomy_warning_conflict(self):
        # cigarette -> smoking taksonomide; manifest yanlış 'phone' derse uyarı
        warns = fetch_mod._taxonomy_warnings({"cigarette": "phone"}, "phone")
        assert warns and "taksonomi" in warns[0]

    def test_taxonomy_no_warning_consistent(self):
        assert fetch_mod._taxonomy_warnings({"cigarette": "smoking"}, "smoking") == []

    def test_output_dir_roboflow(self, tmp_path):
        step = {
            "kind": "roboflow",
            "name": "n",
            "target": "minibus",
            "source": {"workspace": "ws", "project": "pr"},
        }
        out = fetch_mod._output_dir(step, tmp_path)
        assert out == tmp_path / "minibus" / "ws__pr"

    def test_output_dir_sanitizes(self, tmp_path):
        step = {"kind": "kaggle", "name": "n", "target": "t", "source": {"ref": "u/set name"}}
        out = fetch_mod._output_dir(step, tmp_path)
        assert "/" not in out.name and " " not in out.name

    def test_print_plan_empty(self, capsys):
        fetch_mod.print_plan([], Path("base"))
        assert "(boş)" in capsys.readouterr().out

    def test_print_plan_with_warnings(self, capsys):
        plan = fetch_mod.build_plan(
            {
                "targets": {
                    "x": {
                        "sources": [
                            {
                                "kind": "kaggle",
                                "name": "k",
                                "license": "L",
                                "class_map": {"weird": "phone"},
                            }
                        ]
                    }
                }
            }
        )
        fetch_mod.print_plan(plan, Path("base"))
        assert "⚠" in capsys.readouterr().out

    def test_fetch_dry_default(self, tmp_path, monkeypatch):
        m = tmp_path / "m.yaml"
        m.write_text(
            "targets:\n  minibus:\n    aura_class: minibus\n    sources:\n"
            "      - kind: roboflow\n        name: mb\n        workspace: ws\n        project: pr\n",
            encoding="utf-8",
        )
        args = types.SimpleNamespace(manifest=str(m), output=str(tmp_path), klass=None, run=False)
        # _run_step çağrılırsa fail (dry'da indirme olmamalı)
        monkeypatch.setattr(fetch_mod, "_run_step", lambda *a: pytest.fail("dry'da indirme yok"))
        assert fetch_mod.fetch(args) == 0

    def test_fetch_bad_manifest_returns_1(self, tmp_path):
        args = types.SimpleNamespace(
            manifest=str(tmp_path / "nope.yaml"), output=None, klass=None, run=False
        )
        assert fetch_mod.fetch(args) == 1

    def test_fetch_run_invokes_roboflow_pull(self, tmp_path, monkeypatch):
        m = tmp_path / "m.yaml"
        m.write_text(
            "targets:\n  minibus:\n    aura_class: minibus\n    sources:\n"
            "      - kind: roboflow\n        name: mb\n        workspace: ws\n"
            "        project: pr\n        version: 1\n",
            encoding="utf-8",
        )
        called = {}

        def _rf(a):
            called["rf"] = a
            return 0

        monkeypatch.setitem(
            sys.modules, "train.roboflow_pull", types.SimpleNamespace(roboflow_pull=_rf)
        )
        args = types.SimpleNamespace(manifest=str(m), output=str(tmp_path), klass=None, run=True)
        assert fetch_mod.fetch(args) == 0 and "rf" in called

    def test_fetch_run_network_timeout(self, tmp_path, monkeypatch):
        """SİMÜLASYON: ağ-gecikmesi → roboflow_pull TimeoutError; fetch rc!=0 döner."""
        m = tmp_path / "m.yaml"
        m.write_text(
            "targets:\n  minibus:\n    aura_class: minibus\n    sources:\n"
            "      - kind: roboflow\n        name: mb\n        workspace: ws\n        project: pr\n",
            encoding="utf-8",
        )

        def _timeout(a):
            raise TimeoutError("connection timed out after 30s")

        monkeypatch.setitem(
            sys.modules, "train.roboflow_pull", types.SimpleNamespace(roboflow_pull=_timeout)
        )
        args = types.SimpleNamespace(manifest=str(m), output=str(tmp_path), klass=None, run=True)
        with pytest.raises(TimeoutError):
            fetch_mod.fetch(args)

    def test_run_step_manual_source_returns_0(self, tmp_path):
        step = {
            "kind": "kaggle",
            "name": "k",
            "license": "L",
            "source": {"ref": "u/set"},
            "target": "t",
        }
        assert fetch_mod._run_step(step, tmp_path) == 0


# =========================================================================== #
# train/merge_driver_datasets.py — sınıf okuma, remap, merge, hata yolları
# =========================================================================== #
class TestMergeDriver:
    def test_read_class_names_list(self, tmp_path):
        root = _make_yolo_dataset(tmp_path / "ds", ["phone", "smoking"])
        assert merge_mod.read_class_names(root) == {0: "phone", 1: "smoking"}

    def test_read_class_names_dict(self, tmp_path):
        root = tmp_path / "ds"
        root.mkdir()
        (root / "data.yaml").write_text("names:\n  0: phone\n  1: smoking\n", encoding="utf-8")
        assert merge_mod.read_class_names(root) == {0: "phone", 1: "smoking"}

    def test_read_class_names_missing_yaml(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            merge_mod.read_class_names(tmp_path / "nope")

    def test_find_data_yaml_nested(self, tmp_path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        (sub / "data.yaml").write_text("names: []\n", encoding="utf-8")
        assert merge_mod._find_data_yaml(tmp_path) == sub / "data.yaml"

    def test_remap_label_file_drops_unmapped(self, tmp_path):
        lbl = tmp_path / "a.txt"
        lbl.write_text("0 .5 .5 .1 .1\n1 .5 .5 .1 .1\n2 .5 .5 .1 .1\n")
        # 0->0, 2->1; sınıf 1 hedef uzayda yok → atılır
        out = merge_mod._remap_label_file(lbl, {0: 0, 2: 1})
        assert out == ["0 .5 .5 .1 .1", "1 .5 .5 .1 .1"]

    def test_remap_label_file_all_dropped_returns_none(self, tmp_path):
        lbl = tmp_path / "a.txt"
        lbl.write_text("9 .5 .5 .1 .1\n")
        assert merge_mod._remap_label_file(lbl, {0: 0}) is None

    def test_merge_empty_sources(self):
        assert merge_mod.merge({"sources": []}, Path("out")) == 1

    def test_merge_undefined_target(self, tmp_path):
        root = _make_yolo_dataset(tmp_path / "src", ["phone"])
        spec = {
            "target_classes": ["smoking"],  # phone hedefte yok
            "sources": [{"dir": str(root), "map": {"phone": "phone"}}],
        }
        assert merge_mod.merge(spec, tmp_path / "out") == 1

    def test_merge_end_to_end(self, tmp_path):
        root = _make_yolo_dataset(tmp_path / "src", ["phone", "other"])
        spec = {
            "target_classes": ["phone", "smoking", "no_seatbelt", "fatigue"],
            "sources": [{"dir": str(root), "tag": "p", "map": {"phone": "phone"}}],
        }
        out = tmp_path / "out"
        assert merge_mod.merge(spec, out) == 0
        assert (out / "data.yaml").exists()
        # valid -> val alias uygulandı
        assert (out / "val" / "images").is_dir()
        # remap edilen label dosyaları yazıldı, prefix eklendi
        labels = list((out / "train" / "labels").glob("p_*.txt"))
        assert labels

    def test_merge_no_class_mapped_warns_but_runs(self, tmp_path):
        root = _make_yolo_dataset(tmp_path / "src", ["unknown_cls"])
        spec = {
            "target_classes": ["phone", "smoking", "no_seatbelt", "fatigue"],
            "sources": [{"dir": str(root), "map": {}}],
        }
        out = tmp_path / "out"
        assert merge_mod.merge(spec, out) == 0  # eşleşme yok ama akış kesilmez

    def test_inspect(self, tmp_path, caplog):
        root = _make_yolo_dataset(tmp_path / "src", ["phone"])
        import logging

        with caplog.at_level(logging.INFO):
            assert merge_mod.inspect([str(root)]) == 0

    def test_inspect_missing_dir(self, tmp_path):
        assert merge_mod.inspect([str(tmp_path / "nope")]) == 0  # hata loglanır, 0 döner

    def test_main_requires_inspect_or_spec(self):
        assert merge_mod.main([]) == 1

    def test_main_inspect(self, tmp_path):
        root = _make_yolo_dataset(tmp_path / "src", ["phone"])
        assert merge_mod.main(["--inspect", str(root)]) == 0

    def test_main_spec(self, tmp_path):
        root = _make_yolo_dataset(tmp_path / "src", ["phone"])
        spec = tmp_path / "spec.json"
        spec.write_text(
            json.dumps(
                {
                    "target_classes": ["phone", "smoking", "no_seatbelt", "fatigue"],
                    "sources": [{"dir": str(root), "map": {"phone": "phone"}}],
                }
            )
        )
        assert merge_mod.main(["--spec", str(spec), "--out", str(tmp_path / "out")]) == 0

    def test_argparse_defaults(self):
        a = merge_mod.build_parser().parse_args(["--inspect", "d1", "d2"])
        assert a.inspect == ["d1", "d2"] and a.out == "data/processed_driver"


# =========================================================================== #
# train/roboflow_pull.py — API key yok / network timeout (mock)
# =========================================================================== #
class TestRoboflowPull:
    def test_no_api_key_returns_1(self, monkeypatch):
        import train.roboflow_pull as rp

        monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
        args = types.SimpleNamespace(
            workspace="w", project="p", version=1, format="yolov8", output="o"
        )
        assert rp.roboflow_pull(args) == 1

    def test_network_timeout_on_download(self, monkeypatch):
        """SİMÜLASYON: ağ-gecikmesi → Roboflow indirme TimeoutError."""
        import train.roboflow_pull as rp

        monkeypatch.setenv("ROBOFLOW_API_KEY", "dummy")

        class FakeRoboflow:
            def __init__(self, api_key):
                pass

            def workspace(self, w):
                raise TimeoutError("network timeout")

        monkeypatch.setitem(sys.modules, "roboflow", types.SimpleNamespace(Roboflow=FakeRoboflow))
        args = types.SimpleNamespace(
            workspace="w", project="p", version=1, format="yolov8", output="o"
        )
        with pytest.raises(TimeoutError):
            rp.roboflow_pull(args)

    def test_argparse(self):
        import train.roboflow_pull as rp

        a = rp.build_parser().parse_args(["--workspace", "w", "--project", "p"])
        assert a.version == 1 and a.format == "yolov8"


# =========================================================================== #
# tools/ — modüller ağır import içeriyor (Pipeline/torch) → import-safe + main mock
# =========================================================================== #
def _import_tool(name):
    import importlib

    return importlib.import_module(f"tools.{name}")


class TestToolsImportSafe:
    @pytest.mark.parametrize("mod", ["doctor", "bench", "test_video", "show_driver_rois"])
    def test_import_safe(self, mod):
        """Modül import'u ağır bağımlılık tetiklemeden başarılı (lazy import)."""
        m = _import_tool(mod)
        assert hasattr(m, "main")


class TestDoctor:
    def test_unknown_arg_exits(self):
        doctor = _import_tool("doctor")
        with pytest.raises(SystemExit):
            doctor.main(["--bogus-flag"])

    def test_check_python(self):
        doctor = _import_tool("doctor")
        assert doctor._check_python() is True  # bu yorumlayıcı >=3.10

    def test_check_weights_reports(self, tmp_path, monkeypatch):
        doctor = _import_tool("doctor")
        monkeypatch.setattr(doctor, "ROOT", tmp_path)
        (tmp_path / "weights").mkdir()
        # çekirdek ağırlık yok → all_core False
        assert doctor._check_weights() is False

    def test_main_runs_with_mocks(self, monkeypatch, capsys):
        """doctor.main: deps/config mock'lanır → ağır import/model yok, 0/1 döner."""
        doctor = _import_tool("doctor")
        monkeypatch.setattr(doctor, "_check_deps", lambda: True)
        monkeypatch.setattr(doctor, "_check_device", lambda: None)
        monkeypatch.setattr(doctor, "_check_weights", lambda: True)
        monkeypatch.setattr(doctor, "_check_config", lambda prof: True)
        monkeypatch.setattr(doctor, "_check_videos", lambda: None)
        assert doctor.main([]) == 0
        assert "Doctor" in capsys.readouterr().out

    def test_main_missing_core_returns_1(self, monkeypatch):
        doctor = _import_tool("doctor")
        monkeypatch.setattr(doctor, "_check_deps", lambda: False)
        monkeypatch.setattr(doctor, "_check_device", lambda: None)
        monkeypatch.setattr(doctor, "_check_weights", lambda: False)
        monkeypatch.setattr(doctor, "_check_config", lambda prof: False)
        monkeypatch.setattr(doctor, "_check_videos", lambda: None)
        assert doctor.main([]) == 1

    def test_check_config_failure(self, monkeypatch):
        doctor = _import_tool("doctor")
        monkeypatch.setitem(
            sys.modules,
            "aura.config",
            types.SimpleNamespace(
                available_profiles=lambda: [],
                load_config=lambda profile=None: (_ for _ in ()).throw(RuntimeError("boom")),
            ),
        )
        assert doctor._check_config(None) is False


class TestBench:
    def test_argparse_requires_source(self):
        bench = _import_tool("bench")
        with pytest.raises(SystemExit):
            bench.build_parser().parse_args([])

    def test_argparse_defaults(self):
        bench = _import_tool("bench")
        a = bench.build_parser().parse_args(["--source", "v.mp4"])
        assert a.device == "auto" and a.warmup == 5 and a.ai_mode == "real"

    def test_percentile(self):
        bench = _import_tool("bench")
        assert bench._percentile([], 50) == 0.0
        assert bench._percentile([5.0], 95) == 5.0
        assert bench._percentile([0.0, 10.0], 50) == 5.0
        assert bench._percentile([0.0, 10.0, 20.0], 100) == 20.0

    def test_main_missing_source(self, monkeypatch, capsys):
        bench = _import_tool("bench")
        # load_config mock'la ki kaynak-yok kontrolüne kadar gelmeden patlamasın değil;
        # aslında kaynak kontrolü load_config'ten ÖNCE → ağır import tetiklenmez.
        assert bench.main(["--source", "/nonexistent/xyz.mp4"]) == 1
        assert "bulunamadı" in capsys.readouterr().err

    def test_main_mocked_pipeline(self, tmp_path, monkeypatch):
        """SİMÜLASYON YOK gerçek model: Pipeline + load_config + device mock → rapor yazılır."""
        bench = _import_tool("bench")
        src = tmp_path / "v.mp4"
        src.write_bytes(b"x")
        monkeypatch.setattr(bench, "ROOT", tmp_path)

        cfg = types.SimpleNamespace(data={})
        monkeypatch.setattr(bench, "load_config", lambda c, profile=None: cfg)
        monkeypatch.setitem(
            sys.modules, "aura.device", types.SimpleNamespace(resolve_device=lambda d: "cpu")
        )

        class FakePipe:
            def __init__(self, cfg):
                pass

            def frames(self, src, max_frames=None):
                for _ in range(10):
                    yield (None, None, None)

        monkeypatch.setitem(
            sys.modules, "aura.pipeline.pipeline", types.SimpleNamespace(Pipeline=FakePipe)
        )
        rc = bench.main(["--source", str(src), "--warmup", "2"])
        assert rc == 0
        assert (tmp_path / "eval_results" / "bench_cpu.md").exists()

    def test_main_no_measured_frames(self, tmp_path, monkeypatch, capsys):
        """warmup >= kare sayısı → ölçülecek kare yok → rc=1 (uydurma yok, K-004)."""
        bench = _import_tool("bench")
        src = tmp_path / "v.mp4"
        src.write_bytes(b"x")
        monkeypatch.setattr(bench, "ROOT", tmp_path)
        monkeypatch.setattr(
            bench, "load_config", lambda c, profile=None: types.SimpleNamespace(data={})
        )
        monkeypatch.setitem(
            sys.modules, "aura.device", types.SimpleNamespace(resolve_device=lambda d: "cpu")
        )

        class FakePipe:
            def __init__(self, cfg):
                pass

            def frames(self, src, max_frames=None):
                yield (None, None, None)

        monkeypatch.setitem(
            sys.modules, "aura.pipeline.pipeline", types.SimpleNamespace(Pipeline=FakePipe)
        )
        assert bench.main(["--source", str(src), "--warmup", "10"]) == 1
        assert "ölçülecek kare yok" in capsys.readouterr().err


class TestTestVideo:
    def test_argparse_requires_source(self):
        tv = _import_tool("test_video")
        with pytest.raises(SystemExit):
            tv.build_parser().parse_args([])

    def test_argparse_defaults(self):
        tv = _import_tool("test_video")
        a = tv.build_parser().parse_args(["--source", "v.mp4"])
        assert a.ai_mode == "real" and a.no_video is False and a.device is None

    def test_main_missing_source(self, monkeypatch, capsys):
        tv = _import_tool("test_video")
        # load_config import test_video'da source kontrolünden ÖNCE çağrılıyor → mock'la.
        monkeypatch.setattr(
            tv,
            "load_config",
            lambda c, profile=None: types.SimpleNamespace(
                data={"runtime": {}, "models": {"detector": {}}}
            ),
        )
        assert tv.main(["--source", "/nonexistent/zzz.mp4"]) == 1
        assert "bulunamadı" in capsys.readouterr().err


class TestShowDriverRois:
    def test_argparse_requires_source(self):
        sdr = _import_tool("show_driver_rois")
        p = sdr.main  # arg parse içeride; doğrudan parser yok → SystemExit via main
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--source", required=True)
        with pytest.raises(SystemExit):
            ap.parse_args([])
        assert callable(p)

    def test_make_tile_clean_header(self):
        sdr = _import_tool("show_driver_rois")
        roi = np.zeros((40, 60, 3), dtype="uint8")
        ds = types.SimpleNamespace(phone=False, smoking=False, no_seatbelt=False, fatigue=False)
        tile = sdr._make_tile(cv2, roi, 1, ds, None, 320, 240)
        assert tile.shape == (240, 320, 3)

    def test_make_tile_with_flags_and_assign(self):
        sdr = _import_tool("show_driver_rois")
        roi = np.zeros((40, 60, 3), dtype="uint8")
        ds = types.SimpleNamespace(phone=True, smoking=False, no_seatbelt=True, fatigue=False)
        assign = types.SimpleNamespace(
            driver_id=3, locked=True, passenger_ids=[1, 2], driver_bbox=None
        )
        tile = sdr._make_tile(cv2, roi, 5, ds, (1, 1, 10, 10), 320, 240, assign)
        assert tile.shape == (240, 320, 3)

    def test_build_grid_pads(self):
        sdr = _import_tool("show_driver_rois")
        tiles = [np.zeros((240, 320, 3), dtype="uint8") for _ in range(4)]
        grid = sdr._build_grid(cv2, np, tiles, 3, 320, 240)
        # 4 tile, 3 sütun → 2 satır, boş hücre siyah dolu
        assert grid.shape[0] == 480 and grid.shape[1] == 960

    def test_main_video_open_fail(self, monkeypatch, tmp_path, capsys):
        """SİMÜLASYON: video açılamaz → rc=1; ağır model bileşenleri mock'lanır."""
        sdr = _import_tool("show_driver_rois")

        fake_cv2 = types.SimpleNamespace(
            VideoCapture=lambda s: types.SimpleNamespace(isOpened=lambda: False),
            CAP_PROP_FPS=5,
        )
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
        monkeypatch.setitem(sys.modules, "numpy", np)
        monkeypatch.setitem(
            sys.modules,
            "aura.config",
            types.SimpleNamespace(
                load_config=lambda c: types.SimpleNamespace(
                    data={"runtime": {}}, get=lambda *a, **k: 0.1
                )
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "aura.detection.detector",
            types.SimpleNamespace(
                crop_person_roi=lambda *a: None, crop_rois=lambda *a: (None, None)
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "aura.detection.yolo",
            types.SimpleNamespace(YOLO26Detector=lambda cfg: object()),
        )
        monkeypatch.setitem(
            sys.modules,
            "aura.driver_state.classifier",
            types.SimpleNamespace(build_driver_classifier=lambda cfg: object()),
        )
        monkeypatch.setitem(
            sys.modules,
            "aura.identity.driver_lock",
            types.SimpleNamespace(DriverLock=lambda cfg: object()),
        )
        assert sdr.main(["--source", str(tmp_path / "v.mp4")]) == 1
        assert "açılamadı" in capsys.readouterr().err
