"""Roboflow'dan veri seti çek (ROBOFLOW_API_KEY). Yoksa local veriyle çalışın.

python -m train.roboflow_pull --workspace W --project P --version 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

log = logging.getLogger("roadguard.train.roboflow")


def roboflow_pull(args) -> int:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        log.error(
            "ROBOFLOW_API_KEY tanımlı değil. .env'e ekleyin veya local veri kullanın "
            "(docs/veri_seti.md)."
        )
        return 1
    from roboflow import Roboflow  # lazy

    rf = Roboflow(api_key=key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.format, location=args.output)
    log.info("İndirildi: %s", getattr(dataset, "location", args.output))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m train.roboflow_pull",
        description="Roboflow veri seti indirme (ROBOFLOW_API_KEY gerekir).",
    )
    p.add_argument("--workspace", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--format", default="yolov8", help="Etiket formatı (YOLO uyumlu)")
    p.add_argument("--output", default="data/raw/roboflow")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return roboflow_pull(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
