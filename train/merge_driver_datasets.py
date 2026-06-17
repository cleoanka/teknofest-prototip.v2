"""Birden fazla Roboflow YOLO datasetini tek BİRLEŞİK driver_state datasetine birleştir.

Her kaynak dataset kendi sınıf uzayında gelir (ör. telefon dataseti phone=0; kemer
dataseti seatbelt/no_seatbelt; sigara dataseti cigarette=0). Bu betik her kaynağın
etiket indekslerini ORTAK hedef uzaya yeniden eşler ve hepsini tek dataset altında
birleştirir:

    0: phone   1: smoking   2: no_seatbelt   3: fatigue

Akış:
  1) İNCELE — her kaynağın gerçek sınıf isimlerini gör (eşlemeyi buna göre kurarız):
       python -m train.merge_driver_datasets --inspect data/raw/phone data/raw/seatbelt ...
  2) BİRLEŞTİR — spec (eşleme) dosyasıyla birleşik dataset üret:
       python -m train.merge_driver_datasets --spec train/configs/driver_merge.json \
              --out data/processed_driver

Torch GEREKTİRMEZ (saf dosya/etiket işlemi). Çıktı: data/processed_driver/data.yaml
→ `python -m train driver-state --data data/processed_driver/data.yaml` ile eğitilir.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import yaml

log = logging.getLogger("aura.train.merge")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# Roboflow split adı → bizim pipeline'ın beklediği ad (prepare_dataset ile uyumlu).
SPLIT_ALIAS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
DEFAULT_TARGETS = ["phone", "smoking", "no_seatbelt", "fatigue"]


def _find_data_yaml(root: Path) -> Path | None:
    """Dataset kökünde (ya da bir alt klasörde) data.yaml'ı bul."""
    direct = root / "data.yaml"
    if direct.exists():
        return direct
    hits = sorted(root.glob("**/data.yaml"))
    return hits[0] if hits else None


def read_class_names(root: Path) -> dict[int, str]:
    """Bir Roboflow/YOLO datasetinin sınıf isimlerini {indeks: ad} olarak oku.

    `names` hem liste (['phone']) hem sözlük ({0: 'phone'}) formatında olabilir.
    """
    dy = _find_data_yaml(root)
    if not dy:
        raise FileNotFoundError(f"data.yaml bulunamadı: {root}")
    data = yaml.safe_load(dy.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def _iter_split_dirs(root: Path):
    """(roboflow_split, images_dir, labels_dir) üçlülerini üret (mevcut olanlar)."""
    for split in ("train", "valid", "val", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if img_dir.is_dir():
            yield split, img_dir, lbl_dir


def _remap_label_file(src_lbl: Path, idx_remap: dict[int, int]) -> list[str] | None:
    """Bir YOLO etiket dosyasını yeni sınıf indekslerine göre yeniden yaz.

    Hedef uzayda karşılığı OLMAYAN (idx_remap'te bulunmayan) sınıfların kutuları atılır.
    Hiç satır kalmazsa None döner (negatif/boş görüntü olarak yine de kopyalanır).
    """
    out_lines: list[str] = []
    for ln in src_lbl.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        old = int(float(parts[0]))
        if old in idx_remap:
            parts[0] = str(idx_remap[old])
            out_lines.append(" ".join(parts))
    return out_lines or None


def merge(spec: dict, out: Path) -> int:
    targets: list[str] = spec.get("target_classes", DEFAULT_TARGETS)
    target_idx = {name: i for i, name in enumerate(targets)}
    sources = spec.get("sources", [])
    if not sources:
        log.error("spec.sources boş — birleştirilecek dataset yok.")
        return 1

    totals = {"train": 0, "val": 0, "test": 0}
    for src in sources:
        root = Path(src["dir"])
        name_map: dict[str, str] = src.get("map", {})
        tag = src.get("tag") or root.name
        src_names = read_class_names(root)  # {idx: ad}
        # kaynak indeks → hedef indeks (yalnızca map'te olan sınıflar)
        idx_remap: dict[int, int] = {}
        for s_idx, s_name in src_names.items():
            tgt = name_map.get(s_name)
            if tgt is None:
                continue
            if tgt not in target_idx:
                log.error("Hedef sınıf tanımsız: '%s' (target_classes: %s)", tgt, targets)
                return 1
            idx_remap[s_idx] = target_idx[tgt]
        if not idx_remap:
            log.warning("[%s] hiçbir sınıf eşlenmedi (map=%s, kaynak=%s)", tag, name_map, src_names)
        log.info("[%s] eşleme: %s", tag, {src_names[i]: targets[j] for i, j in idx_remap.items()})

        for split, img_dir, lbl_dir in _iter_split_dirs(root):
            dst_split = SPLIT_ALIAS.get(split, split)
            (out / dst_split / "images").mkdir(parents=True, exist_ok=True)
            (out / dst_split / "labels").mkdir(parents=True, exist_ok=True)
            for img in img_dir.iterdir():
                if img.suffix.lower() not in IMG_EXT:
                    continue
                # çakışmayı önlemek için dosya adına kaynak etiketi ekle
                new_stem = f"{tag}_{img.stem}"
                shutil.copy(img, out / dst_split / "images" / f"{new_stem}{img.suffix}")
                src_lbl = lbl_dir / f"{img.stem}.txt"
                lines = _remap_label_file(src_lbl, idx_remap) if src_lbl.exists() else None
                # etiket dosyasını her zaman yaz (boşsa negatif örnek = arka plan)
                (out / dst_split / "labels" / f"{new_stem}.txt").write_text(
                    ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
                )
                totals[dst_split] = totals.get(dst_split, 0) + 1

    _write_data_yaml(out, targets)
    log.info("Birleşik dataset hazır: %s | sınıflar=%s", dict(totals), targets)
    log.info("data.yaml → %s", (out / "data.yaml").resolve())
    return 0


def _write_data_yaml(out: Path, classes: list[str]) -> Path:
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


def inspect(dirs: list[str]) -> int:
    """Her kaynağın sınıf isimlerini yazdır (eşlemeyi kurmadan önce)."""
    for d in dirs:
        root = Path(d)
        try:
            names = read_class_names(root)
            n_imgs = {s: len(list(img.glob("*"))) for s, img, _ in _iter_split_dirs(root)}
            log.info("[%s] sınıflar=%s | görüntü=%s", root.name, names, n_imgs)
        except FileNotFoundError as e:
            log.error("%s", e)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m train.merge_driver_datasets",
        description="Çoklu Roboflow YOLO datasetini birleşik driver_state datasetine birleştir.",
    )
    p.add_argument(
        "--inspect", nargs="+", metavar="DIR", help="Kaynak datasetlerin sınıflarını yaz"
    )
    p.add_argument("--spec", help="Birleştirme spec'i (JSON): target_classes + sources[{dir,map}]")
    p.add_argument("--out", default="data/processed_driver", help="Birleşik dataset çıkış dizini")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.inspect:
        return inspect(args.inspect)
    if not args.spec:
        log.error("Ya --inspect ya da --spec verin. (--help)")
        return 1
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    return merge(spec, Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
