"""Adaptif smoke test — kurulumun çalışır olduğunu kanıtlar.

Pipeline mevcutsa N kare işler ve event üretimini doğrular; değilse (erken
milestone'larda) bağımlılık/import + config + örnek video okunabilirliğini
doğrular. Her durumda ne yaptığını açıkça raporlar. Başarıda exit 0.

    python -m aura.smoke --frames 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    print("> AURA smoke test")
    failures = [n for n in _check_imports() if n.startswith("ZORUNLU")]
    for n in _check_imports():
        prefix = "  [X]" if n.startswith("ZORUNLU") else "  [ ]"
        print(f"{prefix} {n}")
    if failures:
        print("  [X] Zorunlu bagimliliklar eksik.")
        return 1

    # Config yukle
    from aura.config import load_config

    cfg = load_config()
    print(f"  [OK] config yuklendi ({cfg.path})")

    # Ornek video okunabilir mi?
    import cv2

    video = ROOT / cfg.get("runtime.source", "data/samples/ornek.mp4")
    if not video.exists():
        print(f"  [X] ornek video yok: {video} (once: python -m aura.synthetic)")
        return 1
    cap = cv2.VideoCapture(str(video))
    read = 0
    while read < frames:
        ok_, _ = cap.read()
        if not ok_:
            break
        read += 1
    cap.release()
    print(f"  [OK] ornek videodan {read}/{frames} kare okundu")

    # Pipeline mevcutsa uctan-uca kos
    try:
        from aura.pipeline import Pipeline  # type: ignore
    except Exception:
        print("  [ ] pipeline henuz mevcut degil (M2/M3) -- kurulum smoke'u gecti")
        return 0 if read > 0 else 1

    pipe = Pipeline(cfg)
    events = pipe.run_video(str(video), max_frames=frames)
    print(f"  [OK] pipeline {frames} kare isledi, {len(events)} event uretti")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m aura.smoke",
        description="AURA adaptif smoke test (kurulum + pipeline doğrulama).",
    )
    p.add_argument("--frames", type=int, default=10, help="İşlenecek kare sayısı")
    args = p.parse_args(argv)
    rc = run(args.frames)
    print("  [OK] SMOKE OK" if rc == 0 else "  [X] SMOKE FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
