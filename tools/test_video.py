#!/usr/bin/env python3
"""Gerçek video test aracı — annotated mp4 + JSON özet (jüri/denetim kanıtı).

Bir videoyu AURA pipeline'ından geçirir ve iki kanıt üretir:
1. **Annotated mp4**: araç kutuları (risk=kırmızı), plaka (onaylı ✓ / kısmi ?),
   sürücü bayrakları (TELEFON/SİGARA/...), SWERVING işareti, kalibre hız.
   Görselleştirme hijyeni (hidden_prototip dersi): ham model çıktısı değil,
   yalnızca pipeline-onaylı sonuçlar çizilir.
2. **JSON özet**: event sayıları, track başına plaka kararı + oy dökümü,
   sürücü bayrak süreleri (kare), swerving kare sayısı, QoD tetik nedenleri,
   işleme FPS'i. Şartname 4.5 "her hedefin otomatik analiz sonucu üretildiğini
   kanıtlama" yükümlülüğü için zaman damgalı iz.

Örnekler:
    .venv/bin/python tools/test_video.py --source ~/video_1.mp4
    .venv/bin/python tools/test_video.py --source ~/video_2.mp4 --device mps --max-frames 200
    .venv/bin/python tools/test_video.py --source cam.mp4 --no-video   # yalnız JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# Repo kökünden de, tools/ içinden de çalışsın
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aura.config import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python tools/test_video.py",
        description="Videoyu AURA pipeline'ından geçir; annotated mp4 + JSON özet üret.",
        epilog=(
            "örnekler:\n"
            "  python tools/test_video.py --source ~/video_1.mp4\n"
            "  python tools/test_video.py --source ~/video_3.mp4 --device mps --max-frames 200\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", required=True, metavar="PATH", help="İşlenecek video dosyası")
    p.add_argument(
        "--config", default=None, metavar="PATH", help="Config (vars: config/default.yaml)"
    )
    p.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="Config profili: server | laptop | v4-finetune (config/profiles/*.yaml). "
        "Dedektör A/B için pratik: --profile v4-finetune vs (varsayılan yolo26l).",
    )
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default=None, help="İşlem birimi"
    )
    p.add_argument(
        "--ai-mode",
        choices=["auto", "real", "mock"],
        default="real",
        help="AI modu (vars: real — gerçek video testi aracı olduğundan)",
    )
    p.add_argument("--max-frames", type=int, default=None, help="En fazla bu kadar kare işle")
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Dedektör güven eşiği override (stok yolo26s fallback için ~0.10 önerilir)",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Annotated mp4 çıktısı (vars: eval_results/<video>_annotated.mp4)",
    )
    p.add_argument(
        "--json",
        dest="json_path",
        default=None,
        metavar="PATH",
        help="JSON özet çıktısı (vars: eval_results/<video>_summary.json)",
    )
    p.add_argument("--no-video", action="store_true", help="Video yazma (yalnız JSON özet)")
    return p


def _draw(cv2, frame, anno):
    """Pipeline-onaylı sonuçları kareye çiz (ham model çıktısı çizilmez)."""
    img = frame.copy()
    h = img.shape[0]
    fs = max(0.5, h / 1440.0)  # 4K'da okunaklı, 720p'de taşmayan yazı ölçeği
    th = max(1, int(fs * 2))
    for t in anno.tracks:
        x1, y1, x2, y2 = (int(v) for v in t["bbox"])
        risky = bool(t.get("risk_flags"))
        color = (68, 68, 255) if risky else (136, 255, 0)  # BGR: risk kırmızı, normal yeşil
        cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
        lines = [f"ID{t['track_id']} {t.get('cls', '')}".strip()]
        if t.get("plate"):
            lines.append(f"PLAKA: {t['plate']}")
        elif t.get("plate_partial"):
            lines.append(f"plaka? {t['plate_partial']}")
        if t.get("driver"):
            tr_map = {
                "phone": "TELEFON",
                "smoking": "SIGARA",
                "no_seatbelt": "KEMERSIZ",
                "fatigue": "YORGUN",
            }
            lines.append(" ".join(tr_map.get(d, d.upper()) for d in t["driver"]))
        if t.get("swerving"):
            lines.append("SWERVING")
        if t.get("speed_kmh") is not None and t.get("speed_calibrated"):
            lines.append(f"{t['speed_kmh']:.0f} km/h")
        if t.get("qod_active"):
            lines.append("QoD AKTIF")
        for i, line in enumerate(lines):
            ty = max(int(20 * fs), y1 - 8) + int(i * 26 * fs)
            cv2.putText(
                img, line, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, color, th, cv2.LINE_AA
            )
    lim = (getattr(anno, "scene", None) or {}).get("active_speed_limit_kmh")
    if lim is not None:
        cv2.putText(
            img,
            f"LIMIT {lim}",
            (10, int(40 * fs)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (255, 204, 51),
            th,
            cv2.LINE_AA,
        )
    return img


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import cv2

    cfg = load_config(args.config, profile=args.profile)
    cfg.data.setdefault("runtime", {})["ai_mode"] = args.ai_mode
    cfg.data["runtime"]["source"] = args.source
    if args.device:
        cfg.data["runtime"]["device"] = args.device
    if args.conf is not None:
        cfg.data["models"]["detector"]["conf"] = args.conf

    src = Path(args.source).expanduser()
    if not src.exists():
        print(f"HATA: kaynak bulunamadı: {src}", file=sys.stderr)
        return 1
    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    out_video = Path(args.output) if args.output else out_dir / f"{src.stem}_annotated.mp4"
    out_json = Path(args.json_path) if args.json_path else out_dir / f"{src.stem}_summary.json"

    from aura.pipeline.pipeline import Pipeline  # ağır importlar argümanlar doğrulandıktan sonra

    pipe = Pipeline(cfg)

    writer = None
    event_counts: Counter = Counter()
    events_log: list[dict] = []
    driver_frames: dict[int, Counter] = defaultdict(Counter)
    swerve_frames: Counter = Counter()
    qod_reasons: Counter = Counter()
    # Tanılama: track başına yanal yörünge (kare, cx, genişlik) — swerving analizi için
    trajectories: dict[int, list] = defaultdict(list)
    # Hız mutlak-GT doğrulaması (WP-A4): track başına KALİBRE km/h serisi. Yalnız
    # is_calibrated=True kareler toplanır (ısınma öncesi None/kalibresiz değer girmesin,
    # K-004 — uydurma yok). Özette medyan + is_calibrated raporlanır.
    speed_series: dict[int, list] = defaultdict(list)
    # Tanılama: track başına sınıf-DEĞİŞİM izi [(kare, sınıf), ...] — sınıf oylamasının
    # 'car↔truck' titremesini gerçekten bastırdığının kanıtı (yalnız değişimler yazılır).
    class_changes: dict[int, list] = defaultdict(list)
    frames_done = 0
    t0 = time.time()

    try:
        for frame, anno, events in pipe.frames(str(src), max_frames=args.max_frames):
            if not args.no_video:
                if writer is None:
                    fh, fw = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), pipe.fps, (fw, fh)
                    )
                writer.write(_draw(cv2, frame, anno))
            for e in events:
                event_counts[e.type] += 1
                events_log.append(e.model_dump())
                if e.type == "QOD_TRIGGER":
                    qod_reasons[e.payload.get("reason", "?")] += 1
            for t in anno.tracks:
                for d in t.get("driver", []):
                    driver_frames[t["track_id"]][d] += 1
                if t.get("swerving"):
                    swerve_frames[t["track_id"]] += 1
                if t.get("speed_kmh") is not None and t.get("speed_calibrated"):
                    speed_series[t["track_id"]].append(float(t["speed_kmh"]))
                bx = t["bbox"]
                trajectories[t["track_id"]].append(
                    [anno.frame_id, round((bx[0] + bx[2]) / 2, 1), round(bx[2] - bx[0], 1)]
                )
                cc = class_changes[t["track_id"]]
                if not cc or cc[-1][1] != t.get("cls"):
                    cc.append([anno.frame_id, t.get("cls")])
            frames_done += 1
            if frames_done % 50 == 0:
                el = time.time() - t0
                print(f"  {frames_done} kare | {frames_done / el:.1f} FPS", flush=True)
    finally:
        if writer is not None:
            writer.release()
        pipe.close()

    elapsed = max(time.time() - t0, 1e-6)
    tracks = []
    for rec in pipe.acc.active_tracks():
        votes = Counter(rec.plate.votes).most_common(5)
        # Karar şeffaflığı (jüri kanıtı + debug): oy havuzunun GERÇEK ağırlıklı
        # format-geçerli (ikamesiz) dağılımı — kararın neden verildiğini gösterir.
        raw_valid_w: dict[str, float] = {}
        pool = pipe.plate._pools.get(rec.track_id)
        if pool is not None:
            from aura.plate.normalize import normalize_tr as _nt

            for _txt, _w in pool.raw_reads:
                _c, _fx = _nt(_txt)
                if _c is not None and _fx == 0:
                    raw_valid_w[_c] = raw_valid_w.get(_c, 0.0) + _w
            raw_valid_w = dict(sorted(raw_valid_w.items(), key=lambda kv: kv[1], reverse=True)[:6])
        spd = speed_series.get(rec.track_id, [])
        tracks.append(
            {
                "track_id": rec.track_id,
                "vehicle_class": rec.vehicle_class,
                "frames": rec.last_frame - rec.first_frame + 1,
                # Tahmini hız (km/h): kalibre kare serisinin MEDYANI (titreme/aykırı
                # dayanıklı). is_calibrated = en az bir kalibre kare gördük mü; False ise
                # speed_kmh None (metrik iddia yok, K-004).
                "speed_kmh": round(median(spd), 1) if spd else None,
                "speed_is_calibrated": bool(spd),
                "plate": rec.plate.value,
                "plate_status": rec.plate.status,
                "plate_partial": rec.plate.partial,
                "plate_confidence": rec.plate.confidence,
                "plate_top_votes": dict(votes),
                "plate_raw_valid_weighted": {k: round(v, 2) for k, v in raw_valid_w.items()},
                "driver_flag_frames": dict(driver_frames.get(rec.track_id, {})),
                "swerving_frames": int(swerve_frames.get(rec.track_id, 0)),
                "risk_flags": rec.risk_flags,
                "class_changes": class_changes.get(rec.track_id, []),
                "trajectory": trajectories.get(rec.track_id, []),
            }
        )
    summary = {
        "source": str(src),
        "profile": cfg.profile,
        "detector_path": cfg.get("models.detector.path"),
        "detector_conf": cfg.get("models.detector.conf"),
        "device": cfg.get("runtime.device"),
        "frames_processed": frames_done,
        "processing_fps": round(frames_done / elapsed, 2),
        "video_fps": pipe.fps,
        "event_counts": dict(event_counts),
        "qod_trigger_reasons": dict(qod_reasons),
        "tracks": tracks,
        "annotated_video": None if args.no_video else str(out_video),
        "events": events_log,
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✓ {frames_done} kare işlendi ({summary['processing_fps']} FPS)")
    print(f"✓ JSON özet: {out_json}")
    if not args.no_video:
        print(f"✓ Annotated video: {out_video}")
    print("\nEvent dökümü:")
    for et, c in sorted(event_counts.items()):
        print(f"  {et:24s} {c}")
    for t in tracks:
        plate = t["plate"] or f"? {t['plate_partial']}"
        print(
            f"  track {t['track_id']}: {t['vehicle_class']} | plaka={plate} ({t['plate_status']})"
            f" | sürücü={t['driver_flag_frames']} | swerving={t['swerving_frames']} kare"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
