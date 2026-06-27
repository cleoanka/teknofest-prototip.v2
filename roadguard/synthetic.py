"""Sentetik trafik test videosu + ground-truth üreticisi.

Gerçek TOGG veri seti gelene kadar deterministik, anlamlı bir trafik senaryosu
simüle eder: birden fazla araç, farklı sürücü davranışları, plaka varyasyonları.
Mock dedektör (ağırlık yokken) parlak araç bloklarını eşikleme ile bulabilsin diye
araçlar koyu yol üzerinde parlak renkli bloklar olarak çizilir.

    python -m roadguard.synthetic --out data/samples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

import cv2  # noqa: E402  (UTF-8 reconfigure ağır importlardan önce yapılmalı)
import numpy as np  # noqa: E402

# Deterministik senaryo: her araç bir şeritte, farklı zamanda sahneye girer.
# (id, renk BGR, şerit_x_norm, giriş_frame, plaka, sürücü_durumu)
SCENARIO = [
    {
        "id": 1,
        "color": (90, 200, 255),
        "lane_x": 0.33,
        "enter": 0,
        "plate": "34ABC123",
        "vehicle_class": "car",
        "driver": {"phone": True, "smoking": False, "no_seatbelt": False, "fatigue": False},
    },
    {
        "id": 2,
        "color": (120, 255, 120),
        "lane_x": 0.55,
        "enter": 15,
        "plate": "06FY4571",
        "vehicle_class": "truck",
        "driver": {"phone": False, "smoking": True, "no_seatbelt": True, "fatigue": False},
    },
    {
        # Sağ şerit: sweet-spot (x:0.30–0.70) DIŞINDA → OCR gating'i gösterir (plaka okunmaz)
        "id": 3,
        "color": (200, 150, 255),
        "lane_x": 0.78,
        "enter": 35,
        "plate": "35TR07",
        "vehicle_class": "car",
        "driver": {"phone": False, "smoking": False, "no_seatbelt": False, "fatigue": True},
    },
]


def _vehicle_box(spec: dict, t: float, W: int, H: int) -> tuple[int, int, int, int] | None:
    """t∈[0,1] araç ilerlemesi → bbox (perspektif: aşağı indikçe büyür)."""
    if t < 0:
        return None
    # araç ekranın üstünden (uzak) altına (yakın) iner
    cy = 0.15 + 0.75 * t
    if cy > 1.05:
        return None
    scale = 0.05 + 0.15 * t  # yakınlaştıkça büyür (şerit çakışmasını önlemek için ölçülü)
    w = scale * W
    h = scale * H * 1.3
    cx = spec["lane_x"] * W
    x1 = int(cx - w / 2)
    y1 = int(cy * H - h / 2)
    return x1, y1, int(x1 + w), int(y1 + h)


def generate(out_dir: Path, frames: int, fps: int, W: int, H: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "ornek.mp4"
    gt_path = out_dir / "ornek_gt.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))

    gt: dict = {"video": "ornek.mp4", "fps": fps, "width": W, "height": H, "frames": []}

    for fi in range(frames):
        frame = np.full((H, W, 3), 40, dtype=np.uint8)  # koyu asfalt
        # şerit çizgileri
        for lx in (0.20, 0.45, 0.68, 0.90):
            x = int(lx * W)
            for y in range(0, H, 40):
                cv2.line(frame, (x, y), (x, y + 20), (180, 180, 180), 2)

        frame_objs = []
        for spec in SCENARIO:
            life = frames - spec["enter"]
            t = (fi - spec["enter"]) / max(life, 1)
            box = _vehicle_box(spec, t, W, H)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            # parlak araç gövdesi
            cv2.rectangle(frame, (x1, y1), (x2, y2), spec["color"], -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            # plaka bölgesi (alt orta) — beyaz zemin, koyu metin
            pw = int((x2 - x1) * 0.6)
            ph = max(8, int((y2 - y1) * 0.18))
            px1 = x1 + (x2 - x1 - pw) // 2
            py1 = y2 - ph - 2
            cv2.rectangle(frame, (px1, py1), (px1 + pw, py1 + ph), (235, 235, 235), -1)
            fs = max(0.3, ph / 22.0)
            cv2.putText(
                frame,
                spec["plate"],
                (px1 + 2, py1 + ph - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )
            frame_objs.append(
                {
                    "id": spec["id"],
                    "bbox": [x1, y1, x2, y2],
                    "vehicle_class": spec["vehicle_class"],
                    "plate": spec["plate"],
                    "driver": spec["driver"],
                    "speed_kmh": round(40 + 50 * t, 1),
                }
            )
        gt["frames"].append({"frame": fi, "objects": frame_objs})
        writer.write(frame)

    writer.release()
    gt_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    return video_path, gt_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m roadguard.synthetic",
        description="Sentetik trafik test videosu + ground-truth üret (deterministik).",
        epilog="örnek:\n  python -m roadguard.synthetic --out data/samples --frames 90",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default="data/samples", help="Çıktı dizini")
    p.add_argument("--frames", type=int, default=90, help="Kare sayısı (varsayılan 90)")
    p.add_argument("--fps", type=int, default=30, help="FPS (varsayılan 30)")
    p.add_argument("--width", type=int, default=640, help="Genişlik")
    p.add_argument("--height", type=int, default=360, help="Yükseklik")
    args = p.parse_args(argv)

    video, gt = generate(Path(args.out), args.frames, args.fps, args.width, args.height)
    print(f"✓ video: {video}")
    print(f"✓ ground-truth: {gt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
