"""AURA ana inference pipeline CLI — `python -m aura`.

plan.md §4.1 argparse şablonu.
"""

from __future__ import annotations

import argparse
import logging
import sys

from aura.config import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m aura",
        description="AURA inference pipeline — araç, plaka, sürücü durumu ve hız tespiti.",
        epilog=(
            "örnekler:\n"
            "  python -m aura --source 0\n"
            "  python -m aura --source video.mp4 --device mps\n"
            "  python -m aura --source rtsp://10.0.0.5:8554/cam --log-level DEBUG\n"
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
    log = logging.getLogger("aura")

    cfg = load_config(args.config)
    if args.device:
        cfg.data.setdefault("runtime", {})["device"] = args.device
    if args.no_bbox:
        cfg.data.setdefault("dashboard", {})["default_bbox"] = False
    source = args.source if args.source is not None else cfg.get("runtime.source")

    # Pipeline'ı geç import et (ağır CV bağımlılıkları yalnızca gerektiğinde)
    from aura.pipeline.pipeline import Pipeline

    pipe = Pipeline(cfg)
    pipe.emitter.on_event(lambda e: log.info("EVENT %s track=%s %s", e.type, e.track_id, e.payload))

    log.info(
        "Kaynak: %s | device: %s | ai_mode: %s",
        source,
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

    log.info(
        "Tamamlandı: %d event üretildi, %d aktif track.", len(events), len(pipe.acc.active_tracks())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
