#!/usr/bin/env python3
"""yolo26l'nin gördüğü cabin ROI'ları görselleştir.

Her araç tespiti için üst-%55 cabin crop'ını alır, yolo26l'yi çalıştırır
ve tespit kutularını üzerine çizer. Sonucu grid-video olarak kaydeder.

Kullanım:
    .venv/bin/python tools/show_cabin_rois.py --source video_1.mp4
    .venv/bin/python tools/show_cabin_rois.py --source video_1.mp4 --max-frames 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TILE_W, TILE_H = 320, 240  # her cabin crop'un boyutu (px)
GRID_COLS = 3               # yan yana kaç tile


def _draw_detections(cv2, img, results):
    """yolo26l tahminlerini kırpık ROI üzerine çiz."""
    if not results:
        return img
    r = results[0]
    names = getattr(r, "names", None) or {}
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return img
    tr_map = {"phone": "TELEFON", "smoking": "SIGARA", "no_seatbelt": "KEMERSIS", "fatigue": "YORGUN"}
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        cls_idx = int(b.cls.item())
        name = names[cls_idx] if isinstance(names, (list, tuple)) else names.get(cls_idx, "?")
        label = tr_map.get(name, name.upper())
        conf = float(b.conf.item())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 1, cv2.LINE_AA)
    return img


def _make_tile(cv2, cabin_roi, track_id, driver_state, results, tile_w, tile_h):
    """Tek araç için tile oluştur: kırpık ROI + yolo26l kutuları + başlık."""
    tile = cv2.resize(cabin_roi, (tile_w, tile_h))
    # Tespit kutularını orijinal → tile boyutuna ölçekle
    if results and getattr(results[0], "boxes", None) is not None:
        sy = tile_h / cabin_roi.shape[0] if cabin_roi.shape[0] else 1
        sx = tile_w / cabin_roi.shape[1] if cabin_roi.shape[1] else 1
        names = getattr(results[0], "names", {})
        tr_map = {"phone": "TELEFON", "smoking": "SIGARA", "no_seatbelt": "KEMERSIS", "fatigue": "YORGUN"}
        for b in results[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
            cls_idx = int(b.cls.item())
            name = names[cls_idx] if isinstance(names, (list, tuple)) else names.get(cls_idx, "?")
            label = tr_map.get(name, name.upper())
            conf = float(b.conf.item())
            cv2.rectangle(tile, (x1, y1), (x2, y2), (0, 80, 255), 2)
            cv2.putText(tile, f"{label} {conf:.2f}", (x1, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA)
    flags = [f for f, v in {"phone": driver_state.phone, "smoking": driver_state.smoking,
                             "no_seatbelt": driver_state.no_seatbelt,
                             "fatigue": driver_state.fatigue}.items() if v]
    header_color = (0, 60, 200) if flags else (0, 150, 0)
    header_text = f"ID{track_id}  " + ("  ".join(f.upper() for f in flags) if flags else "temiz")
    cv2.rectangle(tile, (0, 0), (tile_w, 22), header_color, -1)
    cv2.putText(tile, header_text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _build_grid(cv2, tiles, grid_cols, tile_w, tile_h):
    """Tile listesini grid'e diz; boş hücreleri siyah doldur."""
    import numpy as np
    while len(tiles) % grid_cols:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype="uint8"))
    rows = []
    for i in range(0, len(tiles), grid_cols):
        rows.append(cv2.hconcat(tiles[i:i + grid_cols]))
    return cv2.vconcat(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="yolo26l cabin ROI görselleştirici")
    p.add_argument("--source", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--conf", type=float, default=0.30, help="yolo26l eşiği (düşük = daha fazla göster)")
    args = p.parse_args(argv)

    import cv2
    import numpy as np
    from ultralytics import YOLO
    from aura.config import load_config, resolve_repo_path
    from aura.detection.detector import crop_rois
    from aura.device import resolve_device
    from aura.schema import DriverState

    src = Path(args.source).expanduser()
    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"{src.stem}_cabin_rois.mp4"

    cfg = load_config(None)
    cfg.data.setdefault("runtime", {})["ai_mode"] = "real"
    device = resolve_device(cfg.get("runtime.device", "auto"))

    # Stage-1: araç dedektörü
    from aura.detection.yolo import YOLO26Detector
    detector = YOLO26Detector(cfg)

    # Stage-2: yolo26l (sürücü durumu)
    driver_model_path = str(resolve_repo_path(cfg.get("models.driver_state.path", "weights/yolo26l.pt")))
    driver_model = YOLO(driver_model_path)
    imgsz = int(cfg.get("models.driver_state.imgsz", 320))

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"HATA: video açılamadı: {src}", file=sys.stderr)
        return 1

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    frame_idx = 0
    print(f"Kaynak : {src}")
    print(f"Model  : {driver_model_path}")
    print(f"Conf   : {args.conf}  |  imgsz: {imgsz}  |  device: {device}")
    print("İşleniyor...\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        dets = detector.detect(frame)

        tiles = []
        for det in dets:
            if det.cabin_roi is None or det.cabin_roi.size == 0:
                continue
            results = driver_model.predict(
                det.cabin_roi, conf=args.conf, imgsz=imgsz, device=device, verbose=False
            )
            ds = DriverState()
            if results and results[0].boxes is not None:
                names = getattr(results[0], "names", None) or driver_model.names
                for b in results[0].boxes:
                    cls_idx = int(b.cls.item())
                    name = names[cls_idx] if isinstance(names, (list, tuple)) else names.get(cls_idx, "?")
                    if hasattr(ds, name):
                        setattr(ds, name, True)

            tid = det.track_id if det.track_id is not None else frame_idx
            tile = _make_tile(cv2, det.cabin_roi, tid, ds, results, TILE_W, TILE_H)
            tiles.append(tile)

        if not tiles:
            # Araç yok → boş placeholder tile
            blank = np.zeros((TILE_H, TILE_W, 3), dtype="uint8")
            cv2.putText(blank, f"kare {frame_idx}: araç yok", (8, TILE_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            tiles = [blank]

        grid = _build_grid(cv2, tiles, GRID_COLS, TILE_W, TILE_H)

        if writer is None:
            gh, gw = grid.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (gw, gh)
            )

        writer.write(grid)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}. kare — {len(dets)} araç tespit")

    cap.release()
    if writer:
        writer.release()
    detector.close()

    print(f"\n✓ {frame_idx} kare işlendi")
    print(f"✓ Cabin ROI videosu: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
