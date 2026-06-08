"""AURA değerlendirme CLI — `python -m aura.eval` (plan.md §4.3)."""
from __future__ import annotations

import argparse
import logging
import sys

from aura.config import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m aura.eval",
        description="AURA model değerlendirme — doğruluk metrikleri ve QoD A/B karşılaştırması",
        epilog=(
            "örnekler:\n"
            "  python -m aura.eval --source data/samples/ornek.mp4 "
            "--ground-truth data/samples/ornek_gt.json\n"
            "  python -m aura.eval --source test.mp4 --ground-truth gt.json --qod-comparison\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", default=None, help="Test video dosyası")
    p.add_argument("--ground-truth", default="data/samples/ornek_gt.json",
                   help="Ground-truth JSON dosyası")
    p.add_argument("--qod-comparison", action="store_true",
                   help="QoD açık/kapalı senaryolarını karşılaştır (şartname kanıtı)")
    p.add_argument("--output", default="eval_results", help="Rapor çıktı dizini")
    p.add_argument("--config", default=None, help="Config dosyası")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    source = args.source if args.source else cfg.get("runtime.source")

    from aura.eval.harness import run_eval

    res = run_eval(cfg, source, args.ground_truth,
                   qod_comparison=args.qod_comparison, output_dir=args.output)

    print("\n=== QoD A/B Değerlendirme ===")
    print(f"{'Metrik':<26} {'QoD OFF':>9} {'QoD ON':>9} {'Δ':>8}")
    for m in res["metrics"]:
        d = m["delta_pct"]
        print(f"{m['name']:<26} {m['qod_off']:>9} {m['qod_on']:>9} "
              f"{('+' if d >= 0 else '') + str(d):>8}")
    print(f"\nRapor: {args.output}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
