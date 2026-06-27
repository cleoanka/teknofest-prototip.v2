"""Eğitim yardımcıları — cihaz çözümleme, doğrulama+metrik export, ağırlık swap, veri istatistiği.

Tasarım: torch/ultralytics importları LAZY (fonksiyon içinde) → `python -m train --help`
ağır bağımlılık gerektirmez. Tüm eğitim çıktıları runs/ altında; en iyi ağırlık + metrik
weights/ altına kopyalanır ki config tek satırla bu modele geçebilsin.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from pathlib import Path

log = logging.getLogger("roadguard.train")

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_device(device: str | None):
    """Eğitim için cihaz çöz. 'auto' → roadguard.device (CUDA→MPS→CPU) ile AÇIKÇA seç.

    ultralytics'in kendi 'auto'su (None) macOS'ta MPS'i çoğu zaman atlayıp CPU'da
    sürünüyor; bu yüzden auto'da roadguard.device.resolve_device kullanılır (gerçek cihaz).
    Açık değer ('cpu'/'mps'/'cuda') aynen geçer.
    """
    if device in (None, "auto", ""):
        from roadguard.device import resolve_device as _rd  # lazy: torch importu

        dev = _rd("auto")
        log.info("Eğitim cihazı (auto → çözüldü): %s", dev)
        return dev
    return device


def summarize_metrics(results) -> dict:
    """ultralytics train/val sonucundan kompakt metrik sözlüğü (mAP/P/R/F1)."""
    out: dict = {}
    box = getattr(results, "box", None)
    try:
        if box is not None:
            p = float(getattr(box, "mp", 0.0))  # mean precision
            r = float(getattr(box, "mr", 0.0))  # mean recall
            out["precision"] = round(p, 4)
            out["recall"] = round(r, 4)
            out["f1"] = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
            out["mAP50"] = round(float(getattr(box, "map50", 0.0)), 4)
            out["mAP50_95"] = round(float(getattr(box, "map", 0.0)), 4)
    except Exception as e:  # noqa: BLE001
        log.warning("Metrik özeti çıkarılamadı: %s", e)
    rd = getattr(results, "results_dict", None) or {}
    for k, v in rd.items():
        try:
            out.setdefault(k.replace("metrics/", "").replace("(B)", ""), round(float(v), 4))
        except (TypeError, ValueError):
            pass
    return out


def export_best(results, dest_name: str, metrics: dict | None = None) -> Path | None:
    """best.pt'yi weights/ altına kopyala + (varsa) metrik json'u yaz (config swap için)."""
    save_dir = Path(getattr(results, "save_dir", "") or "")
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        log.warning("best.pt bulunamadı (eğitim çıktısı: %s)", save_dir or "?")
        return None
    dest = ROOT / "weights" / dest_name
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    log.info("Custom ağırlık kaydedildi: %s", dest)
    if metrics:
        mpath = dest.with_suffix(".metrics.json")
        mpath.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Metrikler: %s  → %s", mpath, metrics)
    log.info(
        "config/default.yaml → ilgili models.*.path değerini 'weights/%s' ile güncelleyin.",
        dest_name,
    )
    return dest


# --no-augment ile sıfırlanan ultralytics augment anahtarları (ablation/küçük-veri için).
_AUG_OFF = dict(
    mosaic=0.0,
    mixup=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    fliplr=0.0,
    scale=0.0,
    translate=0.0,
    degrees=0.0,
    erasing=0.0,
)


def run_finetune(args, out_name: str, task_label: str) -> int:
    """Ortak YOLO26 fine-tune akışı: eğit → doğrula (model.val) → metrik+best export.

    Dedektör ve sürücü-durum eğitimi bunu paylaşır. Hiperparametreler args'tan gelir
    (epochs/imgsz/batch/lr0/patience/resume/no_augment/no_val) — hepsi ayarlanabilir.
    """
    from ultralytics import YOLO  # lazy: --help torch gerektirmesin

    device = resolve_device(args.device)
    log.info(
        "%s fine-tune: base=%s data=%s epochs=%d imgsz=%d batch=%d device=%s",
        task_label,
        args.weights,
        args.data,
        args.epochs,
        args.imgsz,
        args.batch,
        device,
    )
    model = YOLO(args.weights)
    kw = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=int(getattr(args, "patience", 50)),
    )
    if getattr(args, "lr0", None) is not None:
        kw["lr0"] = float(args.lr0)
    # Regularizasyon (dropout): küçük setlerde (seatbelt/smoking ~500-3000 görsel)
    # eğitim eğrisi analizi hafif aşırı-uyum eğilimi gösterdi (val mAP peak sonrası
    # düşüş). Varsayılan 0.0 (geriye uyum, doğrulanmış modeller değişmez); küçük-set
    # retrain'inde --dropout 0.10-0.15 önerilir (genelleme artar).
    if getattr(args, "dropout", 0.0):
        kw["dropout"] = float(args.dropout)
        log.info("Dropout regularizasyonu: %.3f", float(args.dropout))
    if getattr(args, "resume", False):
        kw["resume"] = True
    if getattr(args, "no_augment", False):
        kw.update(_AUG_OFF)
        log.info("Augmentasyon kapalı (--no-augment).")
    results = model.train(**kw)
    metrics = summarize_metrics(results)
    if not getattr(args, "no_val", False):
        try:
            log.info("Doğrulama (model.val) çalışıyor…")
            val = model.val(data=args.data, imgsz=args.imgsz, device=device)
            metrics = summarize_metrics(val) or metrics
            log.info("Doğrulama metrikleri: %s", metrics)
        except Exception as e:  # noqa: BLE001
            log.warning("Doğrulama atlandı (%s) — eğitim metrikleri kullanılıyor.", e)
    export_best(results, getattr(args, "out", None) or out_name, metrics=metrics)
    return 0


def class_distribution(split_dir: Path) -> Counter:
    """Bir split'in labels/ klasöründeki YOLO .txt'lerinden sınıf-örnek sayımı."""
    counts: Counter = Counter()
    labels = split_dir / "labels"
    if not labels.is_dir():
        return counts
    for txt in labels.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if parts:
                try:
                    counts[int(parts[0])] += 1
                except ValueError:
                    continue
    return counts


def dataset_stats(root: Path, names: list[str] | None = None) -> dict:
    """train/val/test için görüntü + sınıf-örnek dağılımı (FTR §2 veri dengeleme raporu)."""
    stats: dict = {"root": str(root), "splits": {}, "class_totals": {}}
    totals: Counter = Counter()
    for split in ("train", "val", "test"):
        sdir = root / split
        if not sdir.is_dir():
            continue
        imgs = [p for p in (sdir / "images").glob("*") if p.suffix.lower() in IMG_EXT]
        dist = class_distribution(sdir)
        stats["splits"][split] = {"images": len(imgs), "class_counts": dict(dist)}
        totals.update(dist)
    stats["class_totals"] = dict(totals)
    if names:
        stats["names"] = names
    # Dengesizlik oranı (en kalabalık / en seyrek sınıf) — FTR "data balancing" sinyali.
    if totals:
        mx, mn = max(totals.values()), min(totals.values())
        stats["imbalance_ratio"] = round(mx / mn, 2) if mn else None
    return stats


def print_dataset_report(stats: dict) -> None:
    """dataset_stats çıktısını okunaklı bas (FTR §2 için kopyalanabilir)."""
    names = stats.get("names")
    print(f"\n=== Veri Seti Dengeleme Raporu: {stats['root']} ===")
    for split, s in stats["splits"].items():
        print(f"  {split:5s}: {s['images']} görüntü")
    print("  Sınıf-örnek dağılımı (tüm split'ler):")
    totals = stats.get("class_totals", {})
    for cid in sorted(totals):
        name = names[cid] if names and cid < len(names) else str(cid)
        print(f"    {cid:>2} {name:<14} {totals[cid]}")
    if stats.get("imbalance_ratio") is not None:
        ratio = stats["imbalance_ratio"]
        flag = "  ⚠ dengesiz (oran > 3) — augment/oversample önerilir" if ratio > 3 else ""
        print(f"  Dengesizlik oranı (max/min): {ratio}{flag}")
