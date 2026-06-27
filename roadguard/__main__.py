"""RoadGuard ana inference pipeline CLI — `python -m roadguard`.

plan.md §4.1 argparse şablonu.
"""

from __future__ import annotations

import argparse
import logging
import sys

from roadguard.config import load_config

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m roadguard",
        description="RoadGuard inference pipeline — araç, plaka, sürücü durumu ve hız tespiti.",
        epilog=(
            "örnekler:\n"
            "  python -m roadguard --source 0\n"
            "  python -m roadguard --source video.mp4 --device mps\n"
            "  python -m roadguard --source rtsp://10.0.0.5:8554/cam --log-level DEBUG\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Config dosyası (varsayılan: config/default.yaml)",
    )
    p.add_argument(
        "--profile",
        metavar="NAME",
        default=None,
        help="Config profili (config/profiles/*.yaml): server | laptop | v4-finetune. "
        "default.yaml üzerine derin-merge edilir. ROADGUARD_PROFILE env ile de verilebilir.",
    )
    p.add_argument(
        "--source",
        metavar="SOURCE",
        default=None,
        help="Video dosyası, kamera index (0,1,2...) veya RTSP/HTTP URL",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="İşlem birimi (varsayılan: config'ten / auto)",
    )
    p.add_argument(
        "--no-bbox", action="store_true", help="Ham video akışı (annotation overlay olmadan)"
    )
    p.add_argument(
        "--max-frames", type=int, default=None, help="En fazla bu kadar kare işle (test/demo için)"
    )
    p.add_argument(
        "--save-events",
        metavar="PATH",
        default=None,
        help="Üretilen tüm event'leri JSONL olarak bu dosyaya yaz (denetim/kanıt izi)",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Log seviyesi (varsayılan: INFO)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("roadguard")

    cfg = load_config(args.config, profile=args.profile)
    if args.device:
        cfg.data.setdefault("runtime", {})["device"] = args.device
    if args.no_bbox:
        cfg.data.setdefault("dashboard", {})["default_bbox"] = False
    if args.source is not None:
        # --source config'e de yazılır: resolve_ai_mode/is_synthetic_source gerçek
        # kaynağa bakar; aksi halde auto modu config'teki örnek videoya göre karar
        # verip gerçek videoyu mock'ta işleyebiliyordu (D3 düzeltmesi).
        cfg.data.setdefault("runtime", {})["source"] = args.source
    source = args.source if args.source is not None else cfg.get("runtime.source")

    # Pipeline'ı geç import et (ağır CV bağımlılıkları yalnızca gerektiğinde)
    from roadguard.pipeline.pipeline import Pipeline

    pipe = Pipeline(cfg)
    pipe.emitter.on_event(lambda e: log.info("EVENT %s track=%s %s", e.type, e.track_id, e.payload))

    events_file = None
    if args.save_events:
        # Kanıt izi (şartname 4.5): her event satır-başına bir JSON olarak yazılır.
        events_file = open(args.save_events, "w", encoding="utf-8")  # noqa: SIM115
        pipe.emitter.on_event(lambda e: events_file.write(e.model_dump_json() + "\n"))

    log.info(
        "Kaynak: %s | profil: %s | device: %s | ai_mode: %s",
        source,
        cfg.profile or "(yok)",
        cfg.get("runtime.device"),
        cfg.get("runtime.ai_mode"),
    )
    try:
        events = pipe.run_video(source, max_frames=args.max_frames)
    except KeyboardInterrupt:
        log.info("Kullanıcı tarafından durduruldu.")
        return 0
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    finally:
        pipe.close()
        if events_file is not None:
            events_file.close()
            log.info("Event'ler kaydedildi: %s", args.save_events)

    log.info(
        "Tamamlandı: %d event üretildi, %d aktif track.", len(events), len(pipe.acc.active_tracks())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
