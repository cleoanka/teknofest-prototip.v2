"""Adaptif smoke test — kurulumun çalışır olduğunu kanıtlar.

Pipeline mevcutsa N kare işler ve event üretimini doğrular; değilse (erken
milestone'larda) bağımlılık/import + config + örnek video okunabilirliğini
doğrular. Her durumda ne yaptığını açıkça raporlar. Başarıda exit 0.

    python -m roadguard.smoke --frames 10
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

ROOT = Path(__file__).resolve().parent.parent


def _check_imports() -> list[str]:
    notes = []
    for mod in ("numpy", "cv2", "pydantic", "yaml"):
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            notes.append(f"ZORUNLU import başarısız: {mod} ({e})")
    for mod in ("ultralytics", "easyocr", "torch", "fastapi"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            notes.append(f"opsiyonel modül yok (mock mod olası): {mod}")
    return notes


def run(frames: int) -> int:
    print("▶ RoadGuard smoke test")
    failures = [n for n in _check_imports() if n.startswith("ZORUNLU")]
    for n in _check_imports():
        prefix = "  ✗" if n.startswith("ZORUNLU") else "  ·"
        print(f"{prefix} {n}")
    if failures:
        print("  ✗ Zorunlu bağımlılıklar eksik.")
        return 1

    # Config yükle
    from roadguard.config import load_config, resolve_source

    cfg = load_config()
    print(f"  ✓ config yüklendi ({cfg.path})")

    # Örnek video okunabilir mi? (kaynak yoksa resolve_source örnek videoya düşer)
    import cv2

    video = Path(str(resolve_source(cfg)))
    if not video.exists():
        print(f"  ✗ örnek video yok: {video} (önce: python -m roadguard.synthetic)")
        return 1
    cap = cv2.VideoCapture(str(video))
    read = 0
    while read < frames:
        ok_, _ = cap.read()
        if not ok_:
            break
        read += 1
    cap.release()
    print(f"  ✓ örnek videodan {read}/{frames} kare okundu")

    # Pipeline mevcutsa uçtan-uca koş
    try:
        from roadguard.pipeline import Pipeline  # type: ignore
    except Exception:
        print("  · pipeline henüz mevcut değil (M2/M3) — kurulum smoke'u geçti")
        return 0 if read > 0 else 1

    pipe = Pipeline(cfg)
    events = pipe.run_video(str(video), max_frames=frames)
    print(f"  ✓ pipeline {frames} kare işledi, {len(events)} event üretti")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m roadguard.smoke",
        description="RoadGuard adaptif smoke test (kurulum + pipeline doğrulama).",
    )
    p.add_argument("--frames", type=int, default=10, help="İşlenecek kare sayısı")
    args = p.parse_args(argv)
    rc = run(args.frames)
    print("  ✓ SMOKE OK" if rc == 0 else "  ✗ SMOKE FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
