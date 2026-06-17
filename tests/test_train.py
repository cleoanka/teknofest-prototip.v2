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


# --- Eksik-sınıf manifesti + fetch --dry planı (AĞ YOK) --------------------- #
from train.fetch import (  # noqa: E402
    DEFAULT_MANIFEST,
    build_plan,
    fetch,
    load_manifest,
)


def test_default_manifest_parses_and_covers_target_classes():
    m = load_manifest(DEFAULT_MANIFEST)
    targets = m["targets"]
    # Görevde adı geçen tüm hedef sınıflar manifestte tanımlı olmalı.
    for cls in ("cigarette", "seatbelt", "fatigue", "minibus", "license_plate"):
        assert cls in targets, cls
        assert "aura_class" in targets[cls]


def test_build_plan_maps_to_aura_taxonomy():
    m = load_manifest(DEFAULT_MANIFEST)
    plan = build_plan(m, only_class="cigarette")
    assert plan, "cigarette için en az bir kaynak olmalı"
    # cigarette -> smoking (aura.taxonomy ile tutarlı)
    assert all(step["aura_class"] == "smoking" for step in plan)
    assert all("smoking" in step["class_map"].values() for step in plan)
    # Her kaynak lisans taşımalı (FTR §5 kaynakça).
    assert all(step["license"] for step in plan)


def test_build_plan_empty_sources_yields_no_steps():
    # ONUR: teyitli açık set yoksa (sources: []) plan boş kalır, sayı uydurulmaz.
    m = load_manifest(DEFAULT_MANIFEST)
    assert build_plan(m, only_class="license_plate") == []
    assert build_plan(m, only_class="fatigue") == []


def test_build_plan_unknown_class_raises():
    m = load_manifest(DEFAULT_MANIFEST)
    import pytest

    with pytest.raises(KeyError):
        build_plan(m, only_class="yok_boyle_sinif")


def test_seatbelt_keeps_raw_class_no_false_warning():
    # seatbelt NESNESİ ham tutulur; aura_class=no_seatbelt_evidence olduğundan
    # taksonomi uyarısı BASTIRILIR (bilinçli karar, yanlış-pozitif değil).
    m = load_manifest(DEFAULT_MANIFEST)
    plan = build_plan(m, only_class="seatbelt")
    assert plan
    assert all(step["warnings"] == [] for step in plan)


def test_custom_manifest_taxonomy_warning_surfaces(tmp_path):
    # Gerçek tutarsızlık (cigarette -> phone) PLANDA uyarı olarak görünmeli.
    mf = tmp_path / "m.yaml"
    mf.write_text(
        "version: 1\n"
        "targets:\n"
        "  test:\n"
        "    aura_class: phone\n"
        "    sources:\n"
        "      - kind: roboflow\n"
        "        name: t\n"
        "        workspace: w\n"
        "        project: p\n"
        "        version: 1\n"
        "        license: CC0\n"
        "        class_map:\n"
        "          cigarette: phone\n",
        encoding="utf-8",
    )
    plan = build_plan(load_manifest(mf), only_class="test")
    assert plan and plan[0]["warnings"], "cigarette->phone tutarsızlığı uyarı üretmeli"


def test_load_manifest_missing_and_malformed(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "yok.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(bad)


def test_fetch_dry_run_no_network(capsys):
    # Varsayılan kuru: ağ yok, çıktı=0, plan basılır. (run=False)
    args = types.SimpleNamespace(manifest=None, klass="minibus", output=None, run=False)
    assert fetch(args) == 0
    out = capsys.readouterr().out
    assert "Çekme Planı" in out and "minibus" in out


def test_fetch_unknown_class_returns_error(capsys):
    args = types.SimpleNamespace(manifest=None, klass="yok", output=None, run=False)
    assert fetch(args) == 1
