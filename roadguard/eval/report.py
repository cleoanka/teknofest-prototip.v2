"""FTR §4 "Çözümün Sınanması" metrik raporu — video-düzeyi başarım + dedektör A/B.

`tools/test_video.py` özet JSON'larını (tracks + driver_flag_frames + plate) ve
`data/samples/<stem>_gt.json` video-düzeyi ground-truth'unu birleştirir; sürücü-davranışı
ve swerving için Precision/Recall/F1, plaka exact-match doğruluğu + CER, araç-sınıfı
doğruluğu ve FPS üretir. Dedektöre göre gruplar (yolo26l vs v4-finetune) → A/B kıyas.

Saf sözlük işleme (cv2/torch GEREKTİRMEZ) → hızlı + test edilebilir. Markdown + CSV
çıktısı FTR §4 tablolarına doğrudan yapıştırılabilir.

NOT (dürüstlük): metrikler 3-videoluk küçük held-out set üzerindedir; bu, davranış
tespitinin *çalıştığının* kanıtıdır, istatistiksel mAP değil. Geniş etiketli set
geldiğinde aynı harness mAP/PR eğrisi üretir (bkz. docs/egitim.md, ftr.md).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from statistics import median

from roadguard.eval import metrics as M

log = logging.getLogger("roadguard.eval.report")

#: Video-düzeyi değerlendirilen davranış sınıfları (GT + tahmin aynı isimler).
BEHAVIORS = ("phone", "smoking", "no_seatbelt", "fatigue", "swerving")


def gt_label(gt: dict) -> dict:
    """Video-düzeyi GT etiketi: araç sınıfı, plaka, sürücü bayrakları, swerving."""
    obj: dict = {}
    for f in gt.get("frames", []):
        objs = f.get("objects", [])
        if objs:
            obj = objs[0]
            break
    drv = obj.get("driver", {}) or {}
    # Hız mutlak-GT (WP-A4): video-düzeyi opsiyonel gerçek hız (komite radar/GPS ölçümü,
    # km/h). Yoksa None → hız doğruluğu sessizce atlanır (sayı uydurma yok, K-004).
    rs = gt.get("real_speed_kmh")
    return {
        "vehicle_class": obj.get("vehicle_class"),
        "plate": obj.get("plate"),
        "phone": bool(drv.get("phone")),
        "smoking": bool(drv.get("smoking")),
        "no_seatbelt": bool(drv.get("no_seatbelt")),
        "fatigue": bool(drv.get("fatigue")),
        "swerving": bool(gt.get("swerving", False)),
        "real_speed_kmh": float(rs) if isinstance(rs, (int, float)) else None,
    }


def pred_from_summary(summary: dict, min_frames: int = 3) -> dict:
    """test_video özetinden video-düzeyi tahmin çıkar (track'ler birleştirilir).

    Bir davranış, herhangi bir track'te >= ``min_frames`` kararlı karede görülürse
    "tespit edildi" sayılır (Katman B oylaması zaten 16/8 süzdü). Plaka: önce
    confirmed, yoksa en güçlü partial.
    """
    pred = {b: False for b in BEHAVIORS}
    plate = None
    plate_status = "none"
    vehicle_class = None
    # Tahmini hız (WP-A4): track-başı kalibre km/h medyanlarının medyanı. Yalnız
    # is_calibrated track'ler sayılır; hiç yoksa None (hız doğruluğu hesaplanmaz, K-004).
    speed_vals: list[float] = []
    for t in summary.get("tracks", []):
        dff = t.get("driver_flag_frames", {}) or {}
        for b in ("phone", "smoking", "no_seatbelt", "fatigue"):
            if dff.get(b, 0) >= min_frames:
                pred[b] = True
        if t.get("swerving_frames", 0) >= min_frames:
            pred["swerving"] = True
        if t.get("speed_is_calibrated") and t.get("speed_kmh") is not None:
            speed_vals.append(float(t["speed_kmh"]))
        if t.get("plate") and plate_status != "confirmed":
            plate, plate_status, vehicle_class = t["plate"], "confirmed", t.get("vehicle_class")
        elif plate is None and t.get("plate_partial"):
            plate, plate_status, vehicle_class = (
                t["plate_partial"],
                "partial",
                t.get("vehicle_class"),
            )
        if vehicle_class is None and t.get("vehicle_class"):
            vehicle_class = t.get("vehicle_class")
    speed_kmh = round(median(speed_vals), 1) if speed_vals else None
    return {
        "plate": plate,
        "plate_status": plate_status,
        "vehicle_class": vehicle_class,
        "speed_kmh": speed_kmh,
        **pred,
    }


def behavior_metrics(pairs: list[tuple[dict, dict]]) -> dict:
    """(gt, pred) çiftlerinden sınıf-bazı + makro P/R/F1 üret."""
    per: dict[str, dict] = {}
    macro_f1 = []
    for b in BEHAVIORS:
        tp = fp = fn = tn = 0
        for gt, pred in pairs:
            g, p = bool(gt.get(b)), bool(pred.get(b))
            tp += g and p
            fp += (not g) and p
            fn += g and (not p)
            tn += (not g) and (not p)
        cm = M.prf1(tp, fp, fn)
        cm["accuracy"] = M.accuracy(tp, tn, fp, fn)
        cm["support"] = tp + fn
        per[b] = cm
        macro_f1.append(cm["f1"])
    per["_macro_f1"] = round(sum(macro_f1) / len(macro_f1), 3) if macro_f1 else 0.0
    return per


def plate_metrics(pairs: list[tuple[dict, dict]]) -> dict:
    """Plaka exact-match doğruluğu + ortalama CER + confirmed/partial dökümü."""
    total = correct = confirmed = partial = 0
    cers = []
    for gt, pred in pairs:
        truth = gt.get("plate")
        if not truth:
            continue
        total += 1
        got = pred.get("plate")
        if pred.get("plate_status") == "confirmed":
            confirmed += 1
        elif pred.get("plate_status") == "partial":
            partial += 1
        if got:
            cers.append(M.cer(got, truth))
            if got == truth:
                correct += 1
    return {
        "total": total,
        "correct": correct,
        "confirmed": confirmed,
        "partial": partial,
        "accuracy": round(100.0 * correct / total, 1) if total else 0.0,
        "mean_cer": round(sum(cers) / len(cers), 3) if cers else 1.0,
    }


def vehicle_class_accuracy(pairs: list[tuple[dict, dict]]) -> float:
    total = correct = 0
    for gt, pred in pairs:
        if gt.get("vehicle_class"):
            total += 1
            correct += pred.get("vehicle_class") == gt.get("vehicle_class")
    return round(100.0 * correct / total, 1) if total else 0.0


def speed_metrics(pairs: list[tuple[dict, dict]]) -> dict | None:
    """Hız mutlak-GT doğruluğu (WP-A4): MAE (km/h) + MAPE (%).

    Yalnız GT'de video-düzeyi ``real_speed_kmh`` BULUNAN ve tahmininde kalibre km/h
    ÜRETİLEN videolar eşleştirilir. Hiç eşleşme yoksa None → çağıran satırı sessizce
    atlar (komite gerçek hız verisi gelmeden hız doğruluğu iddia edilmez, K-004).
    """
    preds: list[float] = []
    truths: list[float] = []
    for gt, pred in pairs:
        truth = gt.get("real_speed_kmh")
        got = pred.get("speed_kmh")
        if truth is None or got is None:
            continue
        preds.append(float(got))
        truths.append(float(truth))
    if not truths:
        return None
    return {"mae_kmh": M.mae(preds, truths), "mape_pct": M.mape(preds, truths), "n": len(truths)}


def _detector_key(summary: dict) -> str:
    """Özeti dedektör grubuna ata: profil > ağırlık dosya kökü."""
    prof = summary.get("profile")
    if prof:
        return prof
    path = summary.get("detector_path") or "?"
    return Path(path).stem


def _load_map_report(summaries_dir: Path, output_dir: str | Path | None) -> dict | None:
    """Varsa `map_report.json`'u bul ve yükle (roadguard/eval/map_eval.py üretir).

    `--map` ayrı çalıştırıldığında çıktı tipik olarak `output_dir`'a (vars: eval_results)
    yazılır; metrik raporu ise farklı bir `--summaries` dizininden okunabilir. Bu yüzden
    birkaç olası konuma bakılır (output_dir > summaries_dir > summaries_dir.parent >
    varsayılan eval_results). İlk bulunan, geçerli JSON döner; yoksa None (kopuk değil,
    DÜRÜST 'henüz üretilmedi' notuna düşülür).
    """
    candidates: list[Path] = []
    if output_dir is not None:
        candidates.append(Path(output_dir) / "map_report.json")
    candidates += [
        summaries_dir / "map_report.json",
        summaries_dir.parent / "map_report.json",
        Path("eval_results") / "map_report.json",
    ]
    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            try:
                data = json.loads(c.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("map50_95") is not None:
                    log.info("mAP raporu bağlandı: %s", c)
                    return data
            except (ValueError, OSError) as e:
                log.warning("map_report.json okunamadı (%s): %s", c, e)
    return None


def build_report(
    summaries_dir: str | Path,
    gt_dir: str | Path = "data/samples",
    min_frames: int = 3,
    output_dir: str | Path | None = None,
) -> dict:
    """summaries_dir/*.json + GT → dedektöre göre gruplu metrik sözlüğü.

    Varsa `map_report.json` (roadguard/eval/map_eval.py çıktısı) bulunup `report["map"]`'e
    işlenir → render_markdown 'İstatistiksel mAP' bölümü gerçek sayıları gösterir.
    """
    summaries_dir = Path(summaries_dir)
    gt_dir = Path(gt_dir)
    groups: dict[str, list[tuple[dict, dict, str, float]]] = {}
    for sjson in sorted(summaries_dir.glob("*.json")):
        summary = json.loads(sjson.read_text(encoding="utf-8"))
        src = summary.get("source", "")
        stem = Path(src).stem
        gt_path = gt_dir / f"{stem}_gt.json"
        if not gt_path.exists():
            log.warning("GT bulunamadı (atlanıyor): %s", gt_path)
            continue
        gt = gt_label(json.loads(gt_path.read_text(encoding="utf-8")))
        pred = pred_from_summary(summary, min_frames=min_frames)
        key = _detector_key(summary)
        groups.setdefault(key, []).append(
            (gt, pred, stem, float(summary.get("processing_fps", 0.0)))
        )

    report: dict = {"detectors": {}, "min_frames": min_frames}
    for key, items in groups.items():
        pairs = [(gt, pred) for gt, pred, _, _ in items]
        fps_vals = [fps for _, _, _, fps in items if fps]
        report["detectors"][key] = {
            "videos": [stem for _, _, stem, _ in items],
            "behavior": behavior_metrics(pairs),
            "plate": plate_metrics(pairs),
            "speed": speed_metrics(pairs),  # None ise hız GT'si yok → rapor sessizce atlar
            "vehicle_class_accuracy": vehicle_class_accuracy(pairs),
            "mean_fps": round(sum(fps_vals) / len(fps_vals), 2) if fps_vals else 0.0,
            "per_video": [{"video": stem, "gt": gt, "pred": pred} for gt, pred, stem, _ in items],
        }
    # İstatistiksel mAP (geniş set) — varsa map_eval.py'nin gerçek sayılarını bağla.
    map_data = _load_map_report(summaries_dir, output_dir)
    if map_data is not None:
        report["map"] = map_data
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# RoadGuard — Başarım Metrikleri (FTR §4 Çözümün Sınanması)",
        "",
        f"- Davranış tespiti eşiği: bir sınıf >= **{report['min_frames']}** kararlı karede görülürse pozitif.",
        "- Set: 3-videoluk gerçek held-out (kapalı otopark, TOGG). Küçük örnek → davranış",
        "  tespitinin *çalıştığının* kanıtı; istatistiksel mAP için geniş etiketli set gerekir",
        "  (bkz. `docs/egitim.md`, `ftr.md`). Tüm eşikler oran-bazlı (videoya-özel sabit yok).",
        "",
    ]
    for key, d in report["detectors"].items():
        lines += [
            f"## Dedektör: `{key}`  (videolar: {', '.join(d['videos'])})",
            "",
            f"- **Ortalama işleme hızı:** {d['mean_fps']} FPS",
            f"- **Araç sınıfı doğruluğu:** {d['vehicle_class_accuracy']}%",
            f"- **Plaka:** {d['plate']['correct']}/{d['plate']['total']} exact-match "
            f"({d['plate']['accuracy']}%), CER={d['plate']['mean_cer']}, "
            f"confirmed={d['plate']['confirmed']}, partial={d['plate']['partial']}",
        ]
        sp = d.get("speed")
        if sp:  # GT'de real_speed_kmh yoksa speed_metrics None → satır eklenmez (sessiz atla)
            lines.append(
                f"- **Hız doğruluğu (MAE/MAPE):** MAE={sp['mae_kmh']} km/h, "
                f"MAPE={sp['mape_pct']}% (n={sp['n']} video)"
            )
        lines += [
            "",
            "| Davranış | TP | FP | FN | Precision | Recall | F1 | Accuracy | Destek |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for b in BEHAVIORS:
            m = d["behavior"][b]
            lines.append(
                f"| {b} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']} | "
                f"{m['recall']} | {m['f1']} | {m['accuracy']} | {m['support']} |"
            )
        lines += [f"| **Makro F1** | | | | | | **{d['behavior']['_macro_f1']}** | | |", ""]
        lines += ["### Video-düzeyi karar matrisi (GT → tahmin)", ""]
        lines += ["| Video | GT davranış | Tahmin davranış | GT plaka | Tahmin plaka (durum) |"]
        lines += ["|---|---|---|---|---|"]
        for pv in d["per_video"]:
            gt_b = ", ".join(b for b in BEHAVIORS if pv["gt"].get(b)) or "temiz"
            pr_b = ", ".join(b for b in BEHAVIORS if pv["pred"].get(b)) or "temiz"
            lines.append(
                f"| {pv['video']} | {gt_b} | {pr_b} | {pv['gt'].get('plate')} | "
                f"{pv['pred'].get('plate')} ({pv['pred'].get('plate_status')}) |"
            )
        lines.append("")

    # --- İstatistiksel mAP (geniş set) — opsiyonel, roadguard/eval/map_eval.py üretir --- #
    lines += ["## İstatistiksel mAP (geniş set)", ""]
    mp = report.get("map")
    if mp:
        lines += [
            "- Kaynak: `map_report.json` (ultralytics `YOLO.val()` çıktısı; "
            "ayrıntılı tablo + PR eğrisi `map_report.md`'de).",
            "",
            "| Metrik | Değer |",
            "|---|---|",
            f"| mAP@50-95 | {mp.get('map50_95')} |",
            f"| mAP@50 | {mp.get('map50')} |",
            f"| Precision (ort.) | {mp.get('precision')} |",
            f"| Recall (ort.) | {mp.get('recall')} |",
            "",
            f"- Ağırlık: `{mp.get('weights')}`, data: `{mp.get('data')}`",
        ]
        per_class = mp.get("per_class") or []
        if per_class:
            lines += [
                "",
                "### Sınıf-bazlı mAP@50-95",
                "",
                "| Sınıf ID | Sınıf | mAP@50-95 |",
                "|---|---|---|",
            ]
            for row in per_class:
                lines.append(
                    f"| {row.get('class_id')} | {row.get('class_name')} | {row.get('map50_95')} |"
                )
        if mp.get("pr_curve"):
            lines.append(f"\n- PR eğrisi: `{mp.get('pr_curve')}`")
        lines.append("")
    else:
        lines += [
            "- Henüz üretilmedi (`map_report.json` bulunamadı). Geniş etiketli set + "
            "ultralytics ile `python -m roadguard.eval --map --weights <w.pt> --data <data.yaml>`",
            "  çalıştırın → ayrı `map_report.md` (+ `.json`) üretilir ve bir sonraki metrik "
            "raporunda bu bölüm gerçek sayılarla DOLAR (bkz. `roadguard/eval/map_eval.py`).",
            "  Yukarıdaki tablolar küçük held-out set kanıtıdır; istatistiksel mAP DEĞİLDİR.",
            "",
        ]
    return "\n".join(lines) + "\n"


def write_csv(report: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["detector", "behavior", "tp", "fp", "fn", "precision", "recall", "f1", "accuracy"]
        )
        for key, d in report["detectors"].items():
            for b in BEHAVIORS:
                m = d["behavior"][b]
                w.writerow(
                    [
                        key,
                        b,
                        m["tp"],
                        m["fp"],
                        m["fn"],
                        m["precision"],
                        m["recall"],
                        m["f1"],
                        m["accuracy"],
                    ]
                )


def run_metrics_report(
    summaries_dir: str | Path,
    gt_dir: str | Path = "data/samples",
    output_dir: str | Path = "eval_results",
    min_frames: int = 3,
) -> dict:
    report = build_report(summaries_dir, gt_dir, min_frames=min_frames, output_dir=output_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(report)
    (out / "metrics_report.md").write_text(md, encoding="utf-8")
    (out / "metrics_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(report, out / "metrics_report.csv")
    log.info("Metrik raporu: %s", out / "metrics_report.md")
    return report
