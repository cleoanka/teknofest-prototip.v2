"""Eğitim: dataset split + data.yaml (torch gerektirmez, CI-uyumlu)."""

from __future__ import annotations

import types
from pathlib import Path

import cv2
import numpy as np

from train.prepare_dataset import prepare_dataset, split_items, write_data_yaml
from train.utils import class_distribution, dataset_stats, summarize_metrics


def test_split_ratios_deterministic():
    imgs = [Path(f"{i}.jpg") for i in range(10)]
    s = split_items(imgs, 0.8, 0.1, seed=1)
    assert len(s["train"]) == 8 and len(s["val"]) == 1 and len(s["test"]) == 1
    assert split_items(imgs, 0.8, 0.1, seed=1) == s  # deterministik


def test_write_data_yaml(tmp_path):
    txt = write_data_yaml(tmp_path, ["car", "truck"]).read_text()
    assert "nc: 2" in txt and "0: car" in txt and "1: truck" in txt


def test_prepare_dataset_end_to_end(tmp_path):
    inp = tmp_path / "raw"
    (inp / "images").mkdir(parents=True)
    for i in range(10):
        cv2.imwrite(str(inp / "images" / f"img{i}.jpg"), np.zeros((20, 20, 3), np.uint8))
        (inp / "images" / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    out = tmp_path / "proc"
    args = types.SimpleNamespace(
        input=str(inp), output=str(out), train=0.8, val=0.1, classes="car,truck", seed=42
    )
    assert prepare_dataset(args) == 0
    assert (out / "data.yaml").exists()
    total = sum(len(list((out / s / "images").glob("*.jpg"))) for s in ("train", "val", "test"))
    assert total == 10
    labels = sum(len(list((out / s / "labels").glob("*.txt"))) for s in ("train", "val", "test"))
    assert labels == 10


# --- veri-dengeleme raporu (FTR §2) + metrik özeti --------------------------- #
def _make_yolo_split(root: Path, split: str, class_lines: dict[str, list[str]]):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for stem, lines in class_lines.items():
        cv2.imwrite(str(root / split / "images" / f"{stem}.jpg"), np.zeros((8, 8, 3), np.uint8))
        (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def test_class_distribution_counts_instances(tmp_path):
    _make_yolo_split(
        tmp_path,
        "train",
        {"a": ["0 .5 .5 .2 .2", "0 .3 .3 .1 .1"], "b": ["1 .5 .5 .2 .2"]},
    )
    dist = class_distribution(tmp_path / "train")
    assert dist[0] == 2 and dist[1] == 1


def test_dataset_stats_imbalance_ratio(tmp_path):
    _make_yolo_split(tmp_path, "train", {"a": ["0 .5 .5 .2 .2"] * 6, "b": ["1 .5 .5 .2 .2"]})
    _make_yolo_split(tmp_path, "val", {"c": ["0 .5 .5 .2 .2"]})
    stats = dataset_stats(tmp_path, names=["car", "truck"])
    assert stats["splits"]["train"]["images"] == 2
    assert stats["class_totals"][0] == 7  # 6 train + 1 val
    assert stats["class_totals"][1] == 1
    assert stats["imbalance_ratio"] == 7.0  # 7 / 1


def test_dataset_report_only_mode(tmp_path):
    _make_yolo_split(tmp_path, "train", {"a": ["0 .5 .5 .2 .2"]})
    args = types.SimpleNamespace(
        report=True, output=str(tmp_path), input=None, classes="car", train=0.8, val=0.1, seed=42
    )
    assert prepare_dataset(args) == 0  # rapor modu: kopyalama yok, hata yok


def test_summarize_metrics_handles_empty():
    assert summarize_metrics(object()) == {}  # box yoksa boş sözlük (çökmeden)
