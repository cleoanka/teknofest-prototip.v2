"""Ham veriyi YOLO formatına hazırla: train/val/test split + data.yaml.

Augmentasyon (mozaik, flip, renk jitter, karartma) eğitim sırasında ultralytics
tarafından uygulanır; bu modül split + dizin yapısı + data.yaml üretir. Torch
gerektirmez (saf dosya işlemi).
"""
from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

log = logging.getLogger("aura.train.dataset")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CLASSES = ["car", "truck", "bus", "minibus"]


def _find_images(inp: Path) -> list[Path]:
    imgs: list[Path] = []
    for d in (inp, inp / "images"):
        if d.is_dir():
            imgs += [p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
    return sorted(set(imgs))


def _label_for(img: Path) -> Path | None:
    for c in (img.parent / f"{img.stem}.txt",
              img.parent.parent / "labels" / f"{img.stem}.txt",
              img.parent / "labels" / f"{img.stem}.txt"):
        if c.exists():
            return c
    return None


def split_items(images: list[Path], train_r: float, val_r: float, seed: int = 42) -> dict:
    rng = random.Random(seed)
    items = list(images)
    rng.shuffle(items)
    n = len(items)
    nt = int(n * train_r)
    nv = int(n * val_r)
    return {"train": items[:nt], "val": items[nt:nt + nv], "test": items[nt + nv:]}


def _read_classes(inp: Path, classes_arg: str | None) -> list[str]:
    if classes_arg:
        return [c.strip() for c in classes_arg.split(",") if c.strip()]
    cf = inp / "classes.txt"
    if cf.exists():
        return [ln.strip() for ln in cf.read_text().splitlines() if ln.strip()]
    return DEFAULT_CLASSES


def write_data_yaml(out: Path, classes: list[str]) -> Path:
    lines = [f"path: {out.resolve()}", "train: train/images", "val: val/images",
             "test: test/images", "", f"nc: {len(classes)}", "names:"]
    lines += [f"  {i}: {c}" for i, c in enumerate(classes)]
    p = out / "data.yaml"
    p.write_text("\n".join(lines) + "\n")
    return p


def prepare_dataset(args) -> int:
    inp, out = Path(args.input), Path(args.output)
    images = _find_images(inp)
    if not images:
        log.error("Girdi dizininde görüntü yok: %s (images/ alt klasörü de denendi)", inp)
        return 1
    split = split_items(images, args.train, args.val, seed=args.seed)
    for name, items in split.items():
        (out / name / "images").mkdir(parents=True, exist_ok=True)
        (out / name / "labels").mkdir(parents=True, exist_ok=True)
        for img in items:
            shutil.copy(img, out / name / "images" / img.name)
            lbl = _label_for(img)
            if lbl:
                shutil.copy(lbl, out / name / "labels" / f"{img.stem}.txt")
    classes = _read_classes(inp, args.classes)
    yaml_path = write_data_yaml(out, classes)
    log.info("Dataset hazır: train=%d val=%d test=%d, sınıflar=%s → %s",
             len(split["train"]), len(split["val"]), len(split["test"]), classes, yaml_path)
    return 0
