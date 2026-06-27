"""QoD A/B değerlendirme harness'i (plan.md §10) — şartname %40 QoD kanıtı.

Aynı video iki senaryoda koşulur:
- **QoD ON**  : tam çözünürlük (yüksek bant / HIGH_THROUGHPUT benzeri).
- **QoD OFF** : düşük çözünürlük simülasyonu (düşük bant) → küçük plaka ROI'leri
                min_pixel_height altına düşer, küçük/uzak araçlar kaçar.

Çıktı: metrik delta tablosu (`GET /eval/results` + dashboard Chart.js + report.md/json).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2

from roadguard.eval import metrics as M
from roadguard.pipeline import Pipeline

log = logging.getLogger("roadguard.eval")


def _run(cfg, source, scale: float = 1.0) -> dict:
    pipe = Pipeline(cfg)
    src = int(source) if isinstance(source, str) and source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Kaynak açılamadı: {source}")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipe.fps = fps_src
    pipe.speed.fps = fps_src
    detected: list[int] = []
    small: list[int] = []
    i, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        anno, _ = pipe.process_frame(frame, i)
        H, W = frame.shape[:2]
        # Küçük-nesne eşiği: kare alanının %2'si. Çarpımı comprehension dışında bir
        # kez hesapla (davranış aynı, per-track tekrar çarpımı önlenir).
        small_thr = 0.02 * max(1, W * H)
        detected.append(len(anno.tracks))
        small.append(
            sum(
                1
                for t in anno.tracks
                if (t["bbox"][2] - t["bbox"][0]) * (t["bbox"][3] - t["bbox"][1]) < small_thr
            )
        )
        i += 1
    cap.release()
    elapsed = time.time() - t0
    confirmed = {
        r.track_id: r.plate.value for r in pipe.acc.active_tracks() if r.plate.status == "confirmed"
    }
    return {
        "confirmed": confirmed,
        "detected": detected,
        "small": small,
        "fps": round(i / elapsed, 1) if elapsed > 0 else 0.0,
        "frames": i,
    }


def run_eval(
    cfg, source, ground_truth, qod_comparison: bool = True, output_dir: str = "eval_results"
) -> dict:
    gt_path = Path(ground_truth)
    gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {"frames": []}

    log.info("Eval: QoD ON (tam çözünürlük)…")
    on = _run(cfg, source, scale=1.0)
    if qod_comparison:
        log.info("Eval: QoD OFF (düşük çözünürlük)…")
        off = _run(cfg, source, scale=0.35)
    else:
        off = on

    pa_on = M.plate_accuracy(on["confirmed"], gt)
    pa_off = M.plate_accuracy(off["confirmed"], gt)
    metrics = [
        _m("Plaka doğruluğu (%)", pa_off["accuracy"], pa_on["accuracy"]),
        _m(
            "Küçük nesne tespiti (%)",
            M.small_object_rate(off["small"], gt),
            M.small_object_rate(on["small"], gt),
        ),
        _m(
            "Tespit oranı (%)",
            M.detection_rate(off["detected"], gt),
            M.detection_rate(on["detected"], gt),
        ),
    ]
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    report_md = _report_md(metrics, pa_on, pa_off, ts, source)
    result = {
        "timestamp": ts,
        "source": str(source),
        "qod_comparison": qod_comparison,
        "metrics": metrics,
        "plate_detail": {"qod_on": pa_on, "qod_off": pa_off},
        "fps": {"qod_on": on["fps"], "qod_off": off["fps"]},
        "report_md": report_md,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "report.md").write_text(report_md, encoding="utf-8")
    log.info("Eval raporu yazıldı: %s", out / "report.md")
    return result


def _m(name: str, off: float, on: float) -> dict:
    return {"name": name, "qod_off": off, "qod_on": on, "delta_pct": round(on - off, 1)}


def _report_md(metrics, pa_on, pa_off, ts, source) -> str:
    lines = [
        "# QoD A/B Değerlendirme Raporu",
        "",
        f"- **Tarih:** {ts}",
        f"- **Kaynak:** {source}",
        "- **Senaryo:** Aynı video, QoD OFF (düşük çözünürlük) vs QoD ON (tam çözünürlük)",
        "",
        "| Metrik | QoD OFF | QoD ON | Δ |",
        "|---|---|---|---|",
    ]
    for m in metrics:
        d = m["delta_pct"]
        lines.append(
            f"| {m['name']} | {m['qod_off']} | {m['qod_on']} | " f"{'+' if d >= 0 else ''}{d} |"
        )
    lines += [
        "",
        "## Plaka detayı",
        f"- **QoD ON:** {pa_on['correct']}/{pa_on['gt_total']} doğru "
        f"({pa_on['confirmed']} okuma), CER={pa_on['cer']}",
        f"- **QoD OFF:** {pa_off['correct']}/{pa_off['gt_total']} doğru "
        f"({pa_off['confirmed']} okuma), CER={pa_off['cer']}",
        "",
        "> QoD yalnızca kritik anda devreye girerek küçük/uzak plaka ROI'lerinin "
        "yeterli pikselle okunmasını sağlar; ölçülen delta bunun kanıtıdır.",
    ]
    return "\n".join(lines) + "\n"
