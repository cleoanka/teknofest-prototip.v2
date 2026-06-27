"""Ham veriyi YOLO formatına hazırla: train/val/test split + data.yaml.

Augmentasyon (mozaik, flip, renk jitter, karartma) eğitim sırasında ultralytics
tarafından uygulanır; bu modül split + dizin yapısı + data.yaml üretir. Torch
gerektirmez (saf dosya işlemi).
"""

from __future__ import annotations

import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from train.utils import dataset_stats, print_dataset_report

log = logging.getLogger("roadguard.train.dataset")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CLASSES = ["car", "truck", "bus", "minibus"]


def oversample_train_split(
    out: Path, names: list[str] | None, max_ratio: float = 3.0
) -> dict[str, int]:
    """TRAIN split'inde seyrek sınıfları içeren görüntüleri çoğaltarak (image+label kopyası)
    sınıf dengesini artırır — FTR §2.3 'oversampling' adımının fiili uygulaması.

    Dengesizlik oranı (en kalabalık/sınıf) ``max_ratio``'yu aşan her seyrek sınıf için, o
    sınıfı içeren görüntüler hedef (= en kalabalık/``max_ratio``) sayıya yaklaşana dek
    çoğaltılır; çoğaltma faktörü üst sınırı 8×'tir (aşırı şişme yok). val/test'e DOKUNULMAZ
    (değerlendirme sızıntısı olmaz). Döndürür: {sınıf: eklenen kopya}.
    """
    img_dir, lbl_dir = out / "train" / "images", out / "train" / "labels"
    if not lbl_dir.is_dir():
        return {}
    counts: Counter = Counter()
    imgs_of: dict[int, list] = defaultdict(list)
    for lbl in sorted(lbl_dir.glob("*.txt")):
        present: set[int] = set()
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if p:
                ci = int(float(p[0]))
                counts[ci] += 1
                present.add(ci)
        img = next(
            (img_dir / f"{lbl.stem}{e}" for e in IMG_EXT if (img_dir / f"{lbl.stem}{e}").exists()),
            None,
        )
        if img:
            for ci in present:
                imgs_of[ci].append((img, lbl))
    if not counts:
        return {}
    target = max(counts.values())
    goal = target / max_ratio  # hedef: seyrek sınıfı bu sayıya çıkar (oran ≤ max_ratio)
    added: dict[str, int] = {}
    for ci in sorted(counts):
        cnt = counts[ci]
        if cnt == 0 or target / cnt <= max_ratio:
            continue  # zaten dengeli
        factor = min(max(1, round(goal / cnt)), 8)  # çoğaltma faktörü (üst sınır 8×)
        n = 0
        for r in range(1, factor):  # factor-1 EK kopya
            for img, lbl in imgs_of[ci]:
                shutil.copy(img, img_dir / f"{img.stem}__os{ci}_{r}{img.suffix}")
                shutil.copy(lbl, lbl_dir / f"{lbl.stem}__os{ci}_{r}.txt")
                n += 1
        if n:
            added[names[ci] if names and ci < len(names) else str(ci)] = n
    return added


def _find_images(inp: Path) -> list[Path]:
    imgs: list[Path] = []
    for d in (inp, inp / "images"):
        if d.is_dir():
            imgs += [p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
    return sorted(set(imgs))


def _label_for(img: Path) -> Path | None:
    for c in (
        img.parent / f"{img.stem}.txt",
        img.parent.parent / "labels" / f"{img.stem}.txt",
        img.parent / "labels" / f"{img.stem}.txt",
    ):
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
    return {"train": items[:nt], "val": items[nt : nt + nv], "test": items[nt + nv :]}


def _read_classes(inp: Path, classes_arg: str | None) -> list[str]:
    if classes_arg:
        return [c.strip() for c in classes_arg.split(",") if c.strip()]
    cf = inp / "classes.txt"
    if cf.exists():
        return [ln.strip() for ln in cf.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return DEFAULT_CLASSES


def write_data_yaml(out: Path, classes: list[str]) -> Path:
    lines = [
        f"path: {out.resolve()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines += [f"  {i}: {c}" for i, c in enumerate(classes)]
    p = out / "data.yaml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def prepare_dataset(args) -> int:
    # --report: mevcut bir YOLO setini analiz et (kopyalama yok) → veri-dengeleme raporu.
    if getattr(args, "report", False):
        root = Path(args.output or args.input or ".")
        if not root.is_dir():
            log.error("Rapor için geçerli bir dizin verin (--output/--input): %s", root)
            return 1
        classes = _read_classes(root, args.classes) if (root / "classes.txt").exists() else None
        print_dataset_report(dataset_stats(root, names=classes))
        return 0

    if not args.input or not args.output:
        log.error("Hazırlık için --input ve --output gerekir (yalnız rapor için --report).")
        return 1
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
    log.info(
        "Dataset hazır: train=%d val=%d test=%d, sınıflar=%s → %s",
        len(split["train"]),
        len(split["val"]),
        len(split["test"]),
        classes,
        yaml_path,
    )
    # FTR §2.3 "data balancing — oversampling": seyrek sınıfları TRAIN'de çoğalt (opsiyonel).
    if getattr(args, "oversample", False):
        added = oversample_train_split(out, classes)
        log.info("Oversampling (train): %s", added or "gerekmedi (oran ≤ 3)")
    # FTR §2 "data balancing": split başına sınıf-örnek dağılımını bas (oversampling sonrası).
    print_dataset_report(dataset_stats(out, names=classes))
    return 0
