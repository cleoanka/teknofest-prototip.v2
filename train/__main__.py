"""AURA model eğitimi CLI — `python -m train` (docs/egitim.md).

Alt komutlar:
  detector       Stage-1 YOLO26 araç/plaka tespit modeli fine-tune
  driver-state   Stage-2 YOLO26 sürücü-durum modeli fine-tune
  dataset        Ham veriyi YOLO formatına böl (+ veri-dengeleme raporu)

Tüm eğitim koşumları doğrulama (model.val) + metrik export (mAP/P/R/F1) yapar ve
en iyi ağırlığı weights/ altına kopyalar (config tek satırla geçer). FTR §2/§4 için
`dataset --report` veri-dengeleme dağılımını basar.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _add_common_train_args(sp, default_weights: str, default_imgsz: int, default_project: str):
    sp.add_argument(
        "--data", required=True, help="data.yaml yolu (veya ultralytics yerleşik: coco128.yaml)"
    )
    sp.add_argument("--epochs", type=int, default=100)
    sp.add_argument("--imgsz", type=int, default=default_imgsz)
    sp.add_argument("--batch", type=int, default=16, help="-1 = otomatik batch (CUDA)")
    sp.add_argument("--weights", default=default_weights, help="Başlangıç ağırlığı (base model)")
    sp.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    sp.add_argument("--project", default=default_project)
    sp.add_argument("--name", default="aura")
    sp.add_argument(
        "--out", default=None, help="weights/ altına kaydedilecek isim (vars: göreve göre)"
    )
    sp.add_argument(
        "--lr0", type=float, default=None, help="Başlangıç öğrenme oranı (vars: ultralytics)"
    )
    sp.add_argument("--patience", type=int, default=50, help="Early-stop sabrı (epoch)")
    sp.add_argument("--resume", action="store_true", help="Yarım kalan koşumu devam ettir")
    sp.add_argument(
        "--no-augment", action="store_true", help="Augmentasyonu kapat (küçük-veri/ablation)"
    )
    sp.add_argument(
        "--no-val", action="store_true", help="Eğitim sonrası model.val doğrulamasını atla"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m train",
        description="AURA model eğitimi (YOLO26 fine-tune + doğrulama + metrik export)",
        epilog=(
            "örnekler:\n"
            "  python -m train detector --data data/processed/data.yaml --epochs 100 --imgsz 768\n"
            "  python -m train detector --data coco128.yaml --weights weights/yolo26s.pt --epochs 3  # boru-hattı doğrulama\n"
            "  python -m train driver-state --data data/driver/data.yaml --imgsz 320\n"
            "  python -m train dataset --input data/raw/ --output data/processed/ --train 0.8 --val 0.1\n"
            "  python -m train dataset --report --output data/processed/   # veri-dengeleme raporu\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="{detector,driver-state,dataset}")

    d = sub.add_parser("detector", help="Stage-1 YOLO26 araç/plaka tespit modeli eğit")
    _add_common_train_args(d, "weights/yolo26s.pt", 640, "runs/detector")

    ds = sub.add_parser("driver-state", help="Stage-2 YOLO26 sürücü-durum modeli eğit")
    _add_common_train_args(ds, "weights/yolo26l.pt", 320, "runs/driver_state")

    dt = sub.add_parser("dataset", help="Ham veriyi YOLO formatına böl + veri-dengeleme raporu")
    dt.add_argument("--input", default=None, help="Ham veri dizini (images/ + labels/)")
    dt.add_argument("--output", default=None, help="Çıktı dizini (YOLO split kökü)")
    dt.add_argument("--train", type=float, default=0.8, help="Train oranı")
    dt.add_argument("--val", type=float, default=0.1, help="Val oranı (kalan → test)")
    dt.add_argument("--classes", default=None, help="Virgülle sınıf listesi (örn. car,truck)")
    dt.add_argument("--seed", type=int, default=42)
    dt.add_argument(
        "--report",
        action="store_true",
        help="Sadece veri-dengeleme raporu bas (mevcut --output/--input YOLO setini analiz et)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.command == "detector":
        from train.train_detector import train_detector

        return train_detector(args)
    if args.command == "driver-state":
        from train.train_driver_state import train_driver_state

        return train_driver_state(args)
    if args.command == "dataset":
        from train.prepare_dataset import prepare_dataset

        return prepare_dataset(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
