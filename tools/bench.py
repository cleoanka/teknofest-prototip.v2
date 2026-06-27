#!/usr/bin/env python3
"""FPS / kare-süresi benchmark aracı — bir videoyu RoadGuard pipeline'ından geçirip
işleme hızını ölçer ve Markdown rapor yazar.

Üretilen ölçüler (yalnız ısınma sonrası, K-004 — uydurma yok):
  * ortalama FPS (işlenen kare / toplam süre)
  * kare-süresi p50 / p95 (ms) — kuyruk gecikmesi (p95) tek-kare ortalamadan önemli
  * ısınma karesi sayısı (ilk kareler model yüklemesi/derleme yüzünden yavaştır;
    istatistiğe katılmaz)

Çıktı: ``eval_results/bench_<device>.md`` (örn. ``bench_mps.md``, ``bench_cuda0.md``).

NOT (cihaz): Apple Silicon (MPS) sayıları **alt sınırdır**; sunucu CUDA'da FPS
belirgin yüksektir (bkz. docs/dagitim.md §5). Gerçek dağıtım FPS'i için bu aracı
CUDA sunucuda ``--device cuda`` ile koşun.

Örnekler:
    python tools/bench.py --source data/samples/ornek.mp4
    python tools/bench.py --source ~/video_1.mp4 --device cuda --profile server
    python tools/bench.py --source ~/video_1.mp4 --max-frames 300 --warmup 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

# Repo kökünden de, tools/ içinden de çalışsın (test_video.py ile aynı kalıp).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aura.config import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python tools/bench.py",
        description=(
            "Videoyu RoadGuard pipeline'ından geçir; ortalama FPS + p50/p95 kare-süresi "
            "ölç ve eval_results/bench_<device>.md raporu yaz."
        ),
        epilog=(
            "örnekler:\n"
            "  python tools/bench.py --source data/samples/ornek.mp4\n"
            "  python tools/bench.py --source ~/video_1.mp4 --device cuda --profile server\n"
            "  python tools/bench.py --source ~/video_1.mp4 --max-frames 300 --warmup 10\n"
            "\nNOT: MPS (Apple Silicon) sayıları alt sınırdır; sunucu CUDA'da FPS yüksektir.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        required=True,
        metavar="PATH",
        help="Benchmark edilecek video dosyası",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="İşlem birimi (vars: auto → CUDA→MPS→CPU otomatik)",
    )
    p.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="Config profili: server | laptop | v4-finetune (config/profiles/*.yaml)",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Config dosyası (vars: config/default.yaml)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="En fazla bu kadar kare işle (vars: tüm video)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        metavar="N",
        help="İlk N kare ısınma sayılır, istatistiğe katılmaz (vars: 5)",
    )
    p.add_argument(
        "--ai-mode",
        choices=["auto", "real", "mock"],
        default="real",
        help="AI modu (vars: real — benchmark gerçek inference hızını ölçer)",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Markdown rapor yolu (vars: eval_results/bench_<device>.md)",
    )
    return p


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Doğrusal-interpolasyonlu yüzdelik (numpy'siz; küçük örnek için yeterli)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    src = Path(args.source).expanduser()
    if not src.exists():
        print(f"HATA: kaynak bulunamadı: {src}", file=sys.stderr)
        return 1

    cfg = load_config(args.config, profile=args.profile)
    cfg.data.setdefault("runtime", {})["ai_mode"] = args.ai_mode
    cfg.data["runtime"]["source"] = str(src)
    cfg.data["runtime"]["device"] = args.device

    # Cihazı gerçek-çözülmüş adıyla raporla (auto → cuda0/mps/cpu; CUDA kullanılamazsa düşer).
    from aura.device import resolve_device  # ağır import argümanlar doğrulandıktan sonra

    resolved = resolve_device(args.device)
    device_tag = resolved.replace(":", "")  # cuda:0 → cuda0 (dosya adı dostu)

    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    out_md = Path(args.output) if args.output else out_dir / f"bench_{device_tag}.md"

    from aura.pipeline.pipeline import Pipeline  # ağır import

    pipe = Pipeline(cfg)

    frame_ms: list[float] = []  # ısınma SONRASI kare süreleri (ms)
    warmup_ms: list[float] = []
    frames_done = 0
    t_wall0 = time.time()
    last = time.perf_counter()

    for _frame, _anno, _events in pipe.frames(str(src), max_frames=args.max_frames):
        now = time.perf_counter()
        dt_ms = (now - last) * 1000.0
        last = now
        if frames_done < args.warmup:
            warmup_ms.append(dt_ms)
        else:
            frame_ms.append(dt_ms)
        frames_done += 1

    wall_s = time.time() - t_wall0

    if not frame_ms:
        print(
            "HATA: ısınma sonrası ölçülecek kare yok "
            f"(işlenen={frames_done}, warmup={args.warmup}). "
            "--warmup'ı düşürün veya daha uzun video verin.",
            file=sys.stderr,
        )
        return 1

    measured = len(frame_ms)
    avg_ms = mean(frame_ms)
    sorted_ms = sorted(frame_ms)
    p50 = _percentile(sorted_ms, 50)
    p95 = _percentile(sorted_ms, 95)
    # Ortalama FPS: ısınma-sonrası kare-süresi ortalamasından (kararlı-hal hızı).
    fps_steady = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    # Uçtan-uca FPS: tüm karelerin duvar-saati ortalaması (ısınma dahil; gözlemlenen).
    fps_wall = frames_done / wall_s if wall_s > 0 else 0.0

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    profile_str = args.profile or "default"
    lines = [
        f"# RoadGuard Benchmark — {device_tag}",
        "",
        f"- **Tarih:** {ts}",
        f"- **Kaynak:** `{src.name}`",
        f"- **Cihaz (istek → çözülen):** `{args.device}` → `{resolved}`",
        f"- **Profil:** `{profile_str}`",
        f"- **AI modu:** `{args.ai_mode}`",
        f"- **İşlenen kare:** {frames_done} (ısınma {len(warmup_ms)}, ölçülen {measured})",
        "",
        "## Sonuçlar (ısınma sonrası)",
        "",
        "| Metrik | Değer |",
        "|---|---|",
        f"| Ortalama FPS (kararlı-hal) | **{fps_steady:.2f}** |",
        f"| Uçtan-uca FPS (ısınma dahil) | {fps_wall:.2f} |",
        f"| Kare-süresi ortalama | {avg_ms:.2f} ms |",
        f"| Kare-süresi p50 | {p50:.2f} ms |",
        f"| Kare-süresi p95 | {p95:.2f} ms |",
        f"| Toplam süre | {wall_s:.2f} s |",
        "",
        "> Not: Apple Silicon (MPS) sayıları **alt sınırdır**; sunucu CUDA'da FPS",
        "> belirgin yüksektir. Gerçek dağıtım FPS'i için bu aracı CUDA sunucuda",
        "> `--device cuda --profile server` ile koşun (bkz. docs/dagitim.md §5).",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # İnsan için kısa özet + makine için tek-satır JSON (loglardan ayrıştırılabilir).
    print(f"FPS(kararlı-hal)={fps_steady:.2f}  p50={p50:.2f}ms  p95={p95:.2f}ms")
    print(
        "BENCH "
        + json.dumps(
            {
                "device": resolved,
                "profile": profile_str,
                "frames": frames_done,
                "measured": measured,
                "fps_steady": round(fps_steady, 2),
                "fps_wall": round(fps_wall, 2),
                "avg_ms": round(avg_ms, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
            }
        )
    )
    print(f"Rapor: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
