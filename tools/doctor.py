#!/usr/bin/env python3
"""RoadGuard sağlık kontrolü — `python tools/doctor.py` (kurulum/ortam teşhisi).

Tek bakışta "her şey hazır mı?" sorusunu yanıtlar: Python sürümü, çekirdek
bağımlılıklar (ultralytics/torch/easyocr), hesaplama cihazı (CUDA/MPS/CPU),
model ağırlıkları, config + aktif profil ve test videoları. Eksikler için NET
düzeltme ipucu verir (sessiz mock'a düşüş yerine erken, anlaşılır uyarı).

Kullanım:
    python tools/doctor.py                 # varsayılan config
    python tools/doctor.py --profile server
"""

from __future__ import annotations

import argparse
import importlib
import os
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

if os.environ.get("ROADGUARD_ASCII") == "1":
    OK, WARN, BAD = "[OK]", "[!]", "[X]"
else:
    OK, WARN, BAD = "✓", "!", "✗"


def _line(mark: str, msg: str) -> None:
    print(f"  {mark} {msg}")


def _check_python() -> bool:
    v = sys.version_info
    ok = v >= (3, 10)
    _line(OK if ok else BAD, f"Python {v.major}.{v.minor}.{v.micro} (>=3.10 gerekli)")
    return ok


def _check_deps() -> bool:
    ok = True
    for mod, hint in (
        ("ultralytics", "pip install ultralytics>=8.4 (YOLO26)"),
        ("torch", "bootstrap.py backend'e göre kurar"),
        ("cv2", "pip install opencv-python"),
        ("easyocr", "pip install easyocr (plaka OCR)"),
        ("fastapi", "pip install fastapi (servisler)"),
    ):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            extra = ""
            if mod == "ultralytics" and ver != "?":
                extra = "  (YOLO26 için >=8.4)" if ver < "8.4" else ""
            _line(OK, f"{mod} {ver}{extra}")
        except Exception as e:  # noqa: BLE001
            ok = False
            _line(BAD, f"{mod} YOK → {hint}  [{type(e).__name__}]")
    return ok


def _check_device() -> None:
    try:
        from roadguard.device import resolve_device

        dev = resolve_device("auto")
        import torch

        detail = []
        if torch.cuda.is_available():
            detail.append(f"CUDA: {torch.cuda.get_device_name(0)}")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            detail.append("MPS (Apple Silicon)")
        _line(OK, f"Cihaz (auto → {dev})  {'; '.join(detail) or 'CPU'}")
    except Exception as e:  # noqa: BLE001
        _line(WARN, f"Cihaz çözümlenemedi: {e}")


def _check_weights() -> bool:
    wdir = ROOT / "weights"
    expected = {
        "yolo26l.pt": "VARSAYILAN dedektör (sunucu)",
        "yolo26s.pt": "hafif/cascade dedektör (laptop profili)",
        "yolo26l-pose.pt": "sürücü pose geometrisi",
        "lp_yolo11n.pt": "sıkı plaka kırpma",
        "yolguvenligi_types_v4.pt": "v4 fine-tune (v4-finetune profili, opsiyonel)",
    }
    all_core = True
    for name, desc in expected.items():
        p = wdir / name
        core = name not in ("yolguvenligi_types_v4.pt",)
        if p.exists():
            mb = p.stat().st_size / 1e6
            _line(OK, f"weights/{name} ({mb:.0f} MB) — {desc}")
        else:
            mark = WARN if not core else BAD
            if core:
                all_core = False
            _line(mark, f"weights/{name} YOK — {desc}  → `python bootstrap.py`")
    return all_core


def _check_config(profile: str | None) -> bool:
    try:
        from roadguard.config import available_profiles, load_config

        cfg = load_config(profile=profile)
        _line(OK, f"config yüklendi: {cfg.path.name}  | profil: {cfg.profile or '(yok)'}")
        _line(
            OK,
            f"dedektör: {cfg.get('models.detector.path')} "
            f"(conf={cfg.get('models.detector.conf')}, imgsz={cfg.get('models.detector.imgsz')})",
        )
        _line(OK, f"ai_mode: {cfg.get('runtime.ai_mode')} | device: {cfg.get('runtime.device')}")
        _line(OK, f"mevcut profiller: {', '.join(available_profiles()) or '(yok)'}")
        # Yapılandırılan dedektör ağırlığı var mı?
        dp = ROOT / str(cfg.get("models.detector.path"))
        if not dp.exists():
            _line(
                WARN,
                f"yapılandırılan dedektör ağırlığı yok ({dp.name}) → stok yolo26s'e LOGLU fallback",
            )
        return True
    except Exception as e:  # noqa: BLE001
        _line(BAD, f"config yüklenemedi: {e}")
        return False


def _check_videos() -> None:
    sample = ROOT / "data" / "samples" / "ornek.mp4"
    _line(
        OK if sample.exists() else WARN,
        f"gömülü demo: {sample.relative_to(ROOT)}"
        + ("" if sample.exists() else "  → `python -m roadguard.synthetic`"),
    )
    found = sorted((Path.home()).glob("video_*.mp4"))
    if found:
        _line(OK, f"test videoları: {', '.join(p.name for p in found)} (~)")
    else:
        _line(WARN, "~/video_*.mp4 test videoları bulunamadı (gerçek footage testi için)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python tools/doctor.py", description="RoadGuard ortam/sağlık kontrolü"
    )
    p.add_argument(
        "--profile",
        default=None,
        help="Kontrol edilecek config profili (server/laptop/v4-finetune)",
    )
    args = p.parse_args(argv)

    print("\n=== RoadGuard Doctor — ortam ve hazırlık kontrolü ===\n")
    print("[Python]")
    py = _check_python()
    print("\n[Bağımlılıklar]")
    deps = _check_deps()
    print("\n[Hesaplama cihazı]")
    _check_device()
    print("\n[Model ağırlıkları]")
    weights = _check_weights()
    print("\n[Yapılandırma]")
    cfg_ok = _check_config(args.profile)
    print("\n[Veri]")
    _check_videos()

    print("\n=== Özet ===")
    if py and deps and weights and cfg_ok:
        print("  ✓ Sistem GERÇEK modda çalışmaya hazır (tüm çekirdek bileşenler mevcut).")
        return 0
    if py and deps and cfg_ok:
        print(
            "  ! Çekirdek tamam ama bazı ağırlıklar eksik → `python bootstrap.py` (yoksa mock mod)."
        )
        return 0
    print("  ✗ Eksikler var (yukarıdaki ✗ satırları). Önce `python bootstrap.py` çalıştırın.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
