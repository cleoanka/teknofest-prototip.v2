"""Eğitim: dataset split + data.yaml (torch gerektirmez, CI-uyumlu)."""

from __future__ import annotations

import types
from pathlib import Path

import cv2
import numpy as np

from train.prepare_dataset import prepare_dataset, split_items, write_data_yaml


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
