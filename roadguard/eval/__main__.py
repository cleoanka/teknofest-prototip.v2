"""RoadGuard değerlendirme CLI — `python -m roadguard.eval` (plan.md §4.3)."""

from __future__ import annotations

import argparse
import logging
import sys

from roadguard.config import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m roadguard.eval",
        description="RoadGuard model değerlendirme — doğruluk metrikleri ve QoD A/B karşılaştırması",
        epilog=(
            "örnekler:\n"
            "  python -m roadguard.eval --source data/samples/ornek.mp4 "
            "--ground-truth data/samples/ornek_gt.json\n"
            "  python -m roadguard.eval --source test.mp4 --ground-truth gt.json --qod-comparison\n"
            "  python -m roadguard.eval --metrics-report --summaries eval_results/ab   # FTR §4 P/R/F1\n"
            "  python -m roadguard.eval --map --weights weights/yolo26l.pt --data data/coco.yaml\n"
            "  # FTR §4 istatistiksel mAP\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--map",
        dest="map_eval",
        action="store_true",
        help="FTR §4 istatistiksel mAP/PR raporu: ultralytics YOLO.val() ile mAP50-95, "
        "mAP50, P, R + sınıf-bazlı tablo. --weights ve --data zorunlu.",
    )
    p.add_argument("--weights", default=None, help="--map için YOLO ağırlık dosyası (.pt)")
    p.add_argument(
        "--data",
        default=None,
        help="--map için ultralytics data tanım YAML'ı (val seti + sınıflar)",
    )
    p.add_argument(
        "--metrics-report",
        action="store_true",
        help="FTR §4 metrik raporu: test_video özetlerinden video-düzeyi P/R/F1 + plaka + "
        "dedektör A/B (QoD A/B koşmaz). --summaries dizinindeki *.json'ları kullanır.",
    )
    p.add_argument(
        "--summaries",
        default="eval_results",
        help="--metrics-report için test_video özet JSON dizini (vars: eval_results)",
    )
    p.add_argument(
        "--gt-dir",
        default="data/samples",
        help="--metrics-report için <stem>_gt.json dizini (vars: data/samples)",
    )
    p.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="--metrics-report: bir davranışın pozitif sayılması için min kararlı kare (vars: 3)",
    )
    p.add_argument("--source", default=None, help="Test video dosyası")
    p.add_argument(
        "--ground-truth", default="data/samples/ornek_gt.json", help="Ground-truth JSON dosyası"
    )
    p.add_argument(
        "--qod-comparison",
        action="store_true",
        help="QoD açık/kapalı senaryolarını karşılaştır (şartname kanıtı)",
    )
    p.add_argument("--output", default="eval_results", help="Rapor çıktı dizini")
    p.add_argument("--config", default=None, help="Config dosyası")
    p.add_argument("--profile", default=None, help="Config profili: server | laptop | v4-finetune")
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows konsolu varsayılan kod sayfası (cp1254/cp1252) "Δ" gibi karakterleri
    # kodlayamaz; UTF-8'e geç ki A/B tablosu platform bağımsız basılsın.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- FTR §4 istatistiksel mAP/PR (ultralytics YOLO.val) --- #
    if args.map_eval:
        if not args.weights or not args.data:
            parser.error("--map için --weights ve --data zorunludur")
        from roadguard.eval.map_eval import run_map

        data = run_map(args.weights, args.data, out_dir=args.output)
        if data is None:
            print(
                "UYARI: mAP değerlendirmesi yapılamadı "
                "(ultralytics yok ya da ağırlık/data eksik). Loglara bakın."
            )
            return 1
        print("\n=== FTR §4 İstatistiksel mAP (geniş set) ===")
        print(f"  mAP@50-95={data['map50_95']} | mAP@50={data['map50']} ")
        print(f"  Precision={data['precision']} | Recall={data['recall']}")
        if data.get("pr_curve"):
            print(f"  PR eğrisi: {data['pr_curve']}")
        print(f"\nRapor: {args.output}/map_report.md (+ .json)")
        return 0

    # --- FTR §4 metrik raporu (video-düzeyi P/R/F1 + dedektör A/B) --- #
    if args.metrics_report:
        from roadguard.eval.report import run_metrics_report

        report = run_metrics_report(
            args.summaries, gt_dir=args.gt_dir, output_dir=args.output, min_frames=args.min_frames
        )
        if not report["detectors"]:
            print(f"UYARI: {args.summaries} içinde GT-eşleşen özet bulunamadı.")
            return 1
        print("\n=== FTR §4 Başarım Metrikleri (dedektör bazında) ===")
        for key, d in report["detectors"].items():
            b = d["behavior"]
            print(
                f"  [{key}] makro-F1={b['_macro_f1']} | plaka {d['plate']['correct']}/"
                f"{d['plate']['total']} ({d['plate']['accuracy']}%) CER={d['plate']['mean_cer']} "
                f"| araç={d['vehicle_class_accuracy']}% | {d['mean_fps']} FPS"
            )
        print(f"\nRapor: {args.output}/metrics_report.md (+ .csv, .json)")
        return 0

    cfg = load_config(args.config, profile=args.profile)
    source = args.source if args.source else cfg.get("runtime.source")

    from roadguard.eval.harness import run_eval

    res = run_eval(
        cfg, source, args.ground_truth, qod_comparison=args.qod_comparison, output_dir=args.output
    )

    print("\n=== QoD A/B Değerlendirme ===")
    print(f"{'Metrik':<26} {'QoD OFF':>9} {'QoD ON':>9} {'Δ':>8}")
    for m in res["metrics"]:
        d = m["delta_pct"]
        print(
            f"{m['name']:<26} {m['qod_off']:>9} {m['qod_on']:>9} "
            f"{('+' if d >= 0 else '') + str(d):>8}"
        )
    print(f"\nRapor: {args.output}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
