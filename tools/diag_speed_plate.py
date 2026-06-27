#!/usr/bin/env python3
"""A/B tanılama: plaka→hız oto-kalibrasyon bağlantısının gerçek videodaki etkisi.

Aynı videoyu iki kez geçirir:
  • baseline : lp_detector KAPALI → kalibrasyon yalnız araç genişliğiyle (eski davranış)
  • improved : lp_detector AÇIK   → plaka bbox'ı observe_plate'e beslenir (yeni)

Her koşu için en uzun ömürlü track'in: ilk kalibre km/h karesi (ısınma), son km/h ve
plaka örnek sayısı raporlanır. GT hız yok (GT'de speed_kmh=null) → mutlak doğruluk
değil, iyileştirmenin ETKİSİ (ısınma + ppm demiri) ölçülür.

Kullanım: python tools/diag_speed_plate.py --source video_1.mp4 --frames 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadguard.config import load_config  # noqa: E402


def run_once(source: str, frames: int, lp_on: bool, conf: float) -> dict:
    cfg = load_config()
    cfg.data.setdefault("runtime", {})["ai_mode"] = "real"
    cfg.data["runtime"]["source"] = source
    cfg.data["runtime"]["device"] = "cpu"
    cfg.data["models"]["detector"]["conf"] = conf  # stok fallback için düşük eşik
    cfg.data["plate"]["lp_detector"]["enabled"] = lp_on

    from roadguard.pipeline.pipeline import Pipeline

    pipe = Pipeline(cfg)

    # observe_plate çağrı sayacı (pipeline plaka bbox'ı geçirdi mi)
    m = pipe.speed._metric
    orig_plate = m.observe_plate
    stats = {"plate_feeds": 0}

    def _counted(b):
        if b is not None:
            stats["plate_feeds"] += 1
        return orig_plate(b)

    m.observe_plate = _counted

    # En uzun ömürlü track'in km/h izi: track_id -> [(frame, kmh, calibrated), ...]
    traj: dict[int, list] = {}
    t0 = time.time()
    for _frame, anno, _events in pipe.frames(source, max_frames=frames):
        for tk in anno.tracks:
            tid = tk["track_id"]
            traj.setdefault(tid, []).append(
                (anno.frame_id, tk.get("speed_kmh"), tk.get("speed_calibrated"))
            )
    elapsed = time.time() - t0

    scale = pipe.speed._metric.scale
    main = max(traj.items(), key=lambda kv: len(kv[1]), default=(None, []))
    tid, seq = main
    first_cal = next((f for (f, v, c) in seq if c and v is not None), None)
    cal_vals = [v for (_f, v, c) in seq if c and v is not None]
    pipe.close()
    return {
        "lp_on": lp_on,
        "elapsed_s": round(elapsed, 1),
        "frames": len(next(iter(traj.values()), [])) if traj else 0,
        "main_track": tid,
        "main_frames": len(seq),
        "plate_feeds": stats["plate_feeds"],
        "scale_samples": scale.n_samples,
        "scale_ready": scale.is_ready,
        "first_calibrated_frame": first_cal,
        "calibrated_kmh_count": len(cal_vals),
        "first_kmh": round(cal_vals[0], 1) if cal_vals else None,
        "last_kmh": round(cal_vals[-1], 1) if cal_vals else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python tools/diag_speed_plate.py")
    p.add_argument("--source", required=True)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--conf", type=float, default=0.10)
    args = p.parse_args(argv)
    if not Path(args.source).expanduser().exists():
        print(f"HATA: kaynak yok: {args.source}", file=sys.stderr)
        return 1

    print(f"=== A/B: {args.source} | {args.frames} kare | conf={args.conf} ===\n")
    base = run_once(args.source, args.frames, lp_on=False, conf=args.conf)
    print("BASELINE (LP kapalı, yalnız araç-genişliği):")
    for k, v in base.items():
        print(f"  {k:24s} {v}")
    impr = run_once(args.source, args.frames, lp_on=True, conf=args.conf)
    print("\nIMPROVED (LP açık, plaka observe_plate'e besleniyor):")
    for k, v in impr.items():
        print(f"  {k:24s} {v}")

    print("\n=== ÖZET ===")
    print(f"  plaka beslemeleri      : {base['plate_feeds']} → {impr['plate_feeds']}")
    print(f"  ölçek örnekleri        : {base['scale_samples']} → {impr['scale_samples']}")
    print(
        f"  ilk kalibre kare (ısınma): {base['first_calibrated_frame']} → "
        f"{impr['first_calibrated_frame']} (küçük = daha erken)"
    )
    print(f"  son km/h               : {base['last_kmh']} → {impr['last_kmh']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
