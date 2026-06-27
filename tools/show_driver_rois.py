#!/usr/bin/env python3
"""Stage-2'nin GERÇEKTEN gördüğü sürücü ROI'lerini görselleştir.

Eski sürümü (show_cabin_rois.py) geometrik üst-%55 kabin kırpığını ve stok COCO
yolo26l'yi gösteriyordu — stok COCO 'phone/smoking' sınıfı ÜRETEMEZ, görüntü
yanıltıcıydı. Bu sürüm pipeline ile AYNI bileşenleri kullanır:

  1. Stage-1 dedektör (v4) + DriverLock → kişi-bazlı sürücü ROI'si
     (kişi yoksa kabin fallback'i — pipeline'daki davranışın birebir aynısı)
  2. Pose sınıflandırıcı (yolo26l-pose) → sürücü-içi SIKI kırpık (+%10) ve
     telefon/sigara HAM bayrakları (16/8 süzgeci ÖNCESİ — teşhis amaçlı)

Her tile: sürücü ROI + pose'un içte kırptığı kutu (sarı) + ham bayrak başlığı.

Kullanım:
    .venv/bin/python tools/show_driver_rois.py --source video_1.mp4
    .venv/bin/python tools/show_driver_rois.py --source video_1.mp4 --max-frames 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TILE_W, TILE_H = 320, 240  # her sürücü ROI tile boyutu (px)
GRID_COLS = 3  # yan yana kaç tile


def _make_tile(cv2, roi, track_id, ds, crop_box, tile_w, tile_h, assign=None):
    """Tek araç için tile: sürücü ROI + iç sıkı-kırpık kutusu + ham bayrak başlığı.

    Başlık ayrıca sürücü kilidi durumunu (KILIT/aday/-) ve yolcu sayısını (y:N) gösterir.
    """
    tile = cv2.resize(roi, (tile_w, tile_h))
    if crop_box is not None and roi.shape[0] and roi.shape[1]:
        sx, sy = tile_w / roi.shape[1], tile_h / roi.shape[0]
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(
            tile, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 220, 255), 2
        )
        cv2.putText(
            tile,
            "pose kirpigi",
            (int(x1 * sx) + 3, max(int(y1 * sy) - 4, 32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    tr_map = {
        "phone": "TELEFON",
        "smoking": "SIGARA",
        "no_seatbelt": "KEMERSIZ",
        "fatigue": "YORGUN",
    }
    flags = [tr_map[f] for f in ("phone", "smoking", "no_seatbelt", "fatigue") if getattr(ds, f)]
    header_color = (0, 60, 200) if flags else (0, 150, 0)
    if assign is not None:
        # Sürücü pozisyoneldir (sağ-en-alt, her kare); '+' = sürücü kuruldu (>=confirm).
        drv = ("SURUCU+" if assign.locked else "SURUCU") if assign.driver_id is not None else "-"
        npass = len(assign.passenger_ids)
        prefix = f"ID{track_id} {drv} y:{npass}"
    else:
        prefix = f"ID{track_id} ham:"
    header = prefix + " " + ("  ".join(flags) if flags else "temiz")
    cv2.rectangle(tile, (0, 0), (tile_w, 22), header_color, -1)
    cv2.putText(
        tile, header, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    return tile


def _build_grid(cv2, np, tiles, grid_cols, tile_w, tile_h):
    """Tile listesini grid'e diz; boş hücreleri siyah doldur."""
    while len(tiles) % grid_cols:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype="uint8"))
    rows = []
    for i in range(0, len(tiles), grid_cols):
        rows.append(cv2.hconcat(tiles[i : i + grid_cols]))
    return cv2.vconcat(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Stage-2 sürücü ROI görselleştirici (pipeline-eş)")
    p.add_argument("--source", required=True, help="video dosyası yolu")
    p.add_argument("--max-frames", type=int, default=None, help="işlenecek azami kare")
    p.add_argument("--output", default=None, help="çıktı mp4 yolu (vars: eval_results/)")
    args = p.parse_args(argv)

    import cv2
    import numpy as np

    from roadguard.config import load_config
    from roadguard.detection.detector import crop_person_roi, crop_rois
    from roadguard.detection.yolo import YOLO26Detector
    from roadguard.driver_state.classifier import build_driver_classifier
    from roadguard.identity.driver_lock import DriverLock

    src = Path(args.source).expanduser()
    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"{src.stem}_driver_rois.mp4"

    cfg = load_config(None)
    cfg.data.setdefault("runtime", {})["ai_mode"] = "real"
    cfg.data["runtime"]["source"] = str(src)

    detector = YOLO26Detector(cfg)
    driver_lock = DriverLock(cfg)
    classifier = build_driver_classifier(cfg)
    roi_pad = float(cfg.get("driver_lock.roi_pad", 0.10))

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"HATA: video açılamadı: {src}", file=sys.stderr)
        return 1

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    frame_idx = 0
    print(f"Kaynak : {src}")
    print(f"Backend: {type(classifier).__name__} (pipeline ile aynı)")
    print("İşleniyor...\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        dets = detector.detect(frame)
        persons = getattr(detector, "last_persons", [])
        vehicles = [(d.track_id if d.track_id is not None else -1, d.bbox) for d in dets]
        assignments = driver_lock.assign_frame(vehicles, persons, frame_idx)

        tiles = []
        for det, assign in zip(dets, assignments, strict=True):
            tid = det.track_id if det.track_id is not None else -1
            cabin, _ = crop_rois(frame, det.bbox)
            # Pipeline ile birebir aynı ROI seçimi: kişi kutusu varsa o, yoksa kabin.
            roi = None
            if assign.driver_bbox is not None:
                roi = crop_person_roi(frame, assign.driver_bbox, roi_pad)
            if roi is None:
                roi = cabin
            if roi is None or roi.size == 0:
                continue
            ds = classifier.infer(roi, track_id=tid)
            crop_box = getattr(classifier, "last_crop_box", None)
            tiles.append(_make_tile(cv2, roi, tid, ds, crop_box, TILE_W, TILE_H, assign))

        if not tiles:
            blank = np.zeros((TILE_H, TILE_W, 3), dtype="uint8")
            cv2.putText(
                blank,
                f"kare {frame_idx}: araç yok",
                (8, TILE_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )
            tiles = [blank]

        grid = _build_grid(cv2, np, tiles, GRID_COLS, TILE_W, TILE_H)
        if writer is None:
            gh, gw = grid.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (gw, gh)
            )
        writer.write(grid)
        driver_lock.prune(frame_idx)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}. kare — {len(dets)} araç")

    cap.release()
    if writer:
        writer.release()
    detector.close()

    print(f"\n✓ {frame_idx} kare işlendi")
    print(f"✓ Sürücü ROI videosu: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
