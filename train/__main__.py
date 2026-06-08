"""AURA model eğitimi CLI — `python -m train` (plan.md §4.2)."""
from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m train",
        description="AURA model eğitimi",
        epilog=(
            "örnekler:\n"
            "  python -m train detector --data data/detector.yaml --epochs 100\n"
            "  python -m train driver-state --data data/driver.yaml --imgsz 320\n"
            "  python -m train dataset --input data/raw/ --output data/processed/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True,
                           metavar="{detector,driver-state,dataset}")

    d = sub.add_parser("detector", help="Stage-1 araç tespit modelini eğit (YOLO26s fine-tune)")
    d.add_argument("--data", required=True, help="data.yaml yolu")
    d.add_argument("--epochs", type=int, default=100)
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--batch", type=int, default=16)
    d.add_argument("--weights", default="weights/yolo26s.pt", help="Başlangıç ağırlığı")
    d.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    d.add_argument("--project", default="runs/detector")
    d.add_argument("--name", default="aura")

    ds = sub.add_parser("driver-state",
                        help="Stage-2 sürücü durumu modelini eğit (YOLO26l fine-tune)")
    ds.add_argument("--data", required=True, help="data.yaml yolu")
    ds.add_argument("--epochs", type=int, default=100)
    ds.add_argument("--imgsz", type=int, default=320, help="Cabin ROI küçük → 320 önerilir")
    ds.add_argument("--batch", type=int, default=16)
    ds.add_argument("--weights", default="weights/yolo26l.pt")
    ds.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ds.add_argument("--project", default="runs/driver_state")
    ds.add_argument("--name", default="aura")

    dt = sub.add_parser("dataset", help="Ham veriyi YOLO formatına dönüştür ve split uygula")
    dt.add_argument("--input", required=True, help="Ham veri dizini (images/ + labels/)")
    dt.add_argument("--output", required=True, help="Çıktı dizini")
    dt.add_argument("--train", type=float, default=0.8, help="Train oranı")
    dt.add_argument("--val", type=float, default=0.1, help="Val oranı (kalan → test)")
    dt.add_argument("--classes", default=None, help="Virgülle sınıf listesi (örn. car,truck)")
    dt.add_argument("--seed", type=int, default=42)
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
