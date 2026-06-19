#!/usr/bin/env python3
"""Stok vs custom-profil A/B karşılaştırıcı (K-004: ölçülen sayı, gizleme yok).

İki tarafın test_video.py JSON özetlerini okur; plaka (exact/CER + P/R/F1 proxy)
ve sigara (kare sayısı + risk) metriklerini tablolar. GT plaka = 34TC8532.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GT_PLATE = "34TC8532"
ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval_results"


def cer(pred: str, gt: str) -> float:
    """Karakter Hata Oranı = Levenshtein(pred, gt) / len(gt)."""
    if not gt:
        return 0.0 if not pred else 1.0
    m, n = len(pred), len(gt)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if pred[i - 1] == gt[j - 1] else 1),
            )
            prev = cur
    return dp[n] / n


def load(stem: str, side: str) -> dict | None:
    if side == "stok":
        p = EVAL / f"{stem}_stok_summary.json"
    else:
        p = EVAL / f"{stem}_custom_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def plate_metrics(summary: dict) -> dict:
    """Onaylı plaka EVENTS akışından (PLATE_CONFIRMED) okunur — tracks listesi koşu
    sonunda yalnız HÂLÂ SAHNEDE olan araçları içerir, plakalı araç çoktan çıkmış olur.

    Onay yoksa en iyi partial kanıt izi (tracks'teki plate_partial) raporlanır.
    """
    confirmed_vals = [
        e["payload"].get("value")
        for e in summary.get("events", [])
        if e.get("type") == "PLATE_CONFIRMED" and e.get("payload", {}).get("value")
    ]
    val = confirmed_vals[0] if confirmed_vals else ""
    # Birden çok onay varsa hepsini takip et (GT-dışı ikincil onay = yanlış-onay riski)
    partials = [t.get("plate_partial") for t in summary.get("tracks", []) if t.get("plate_partial")]
    partial = partials[0] if partials else ""
    text = val or partial
    c = cer(text, GT_PLATE) if text else 1.0
    wrong_confirm = any(v != GT_PLATE for v in confirmed_vals)
    return {
        "value": val or None,
        "partial": partial or None,
        "cer": round(c, 3),
        "exact": val == GT_PLATE,
        "status": "confirmed" if val else ("pending" if partial else "none"),
        "n_confirmed": len(confirmed_vals),
        "wrong_confirm": wrong_confirm,
    }


def smoking_metrics(summary: dict) -> dict:
    """Sürücü bayrak kareleri + risk: EVENTS akışından (DRIVER_STATE flags + RISK_ALERT).

    DRIVER_STATE her aktif-bayrak karesinde bir event üretir → flag başına event sayısı
    o davranışın görüldüğü kare-sayısının ölçüsüdür (16/8 oylamasından geçmiş KARARLI).
    """
    smk = phone = 0
    risk_rules: dict[str, int] = {}
    for e in summary.get("events", []):
        if e.get("type") == "DRIVER_STATE":
            for f in e.get("payload", {}).get("flags", []):
                if f == "smoking":
                    smk += 1
                elif f == "phone":
                    phone += 1
        elif e.get("type") == "RISK_ALERT":
            rule = e.get("payload", {}).get("rule", "?")
            risk_rules[rule] = risk_rules.get(rule, 0) + 1
    ev = summary.get("event_counts", {})
    return {
        "smoking_events": smk,
        "phone_events": phone,
        "DRIVER_STATE": ev.get("DRIVER_STATE", 0),
        "RISK_ALERT": ev.get("RISK_ALERT", 0),
        "risk_rules": risk_rules,
    }


def main() -> int:
    videos = sys.argv[1:] or ["video_1", "video_2", "video_3"]
    print(
        f"\n{'='*78}\nA/B: STOK (lp_yolo11n + pose-hibrit) vs CUSTOM (custom_lp + custom_smoking)"
    )
    print(f"GT plaka = {GT_PLATE}\n{'='*78}")

    plate_rows = []
    smk_rows = []
    exact_stok = exact_custom = 0
    wrong_stok = wrong_custom = 0
    for v in videos:
        s = load(v, "stok")
        c = load(v, "custom")
        if s is None or c is None:
            print(f"[{v}] EKSİK: stok={s is not None} custom={c is not None}")
            continue
        ps, pc = plate_metrics(s), plate_metrics(c)
        ss, sc = smoking_metrics(s), smoking_metrics(c)
        exact_stok += int(ps["exact"])
        exact_custom += int(pc["exact"])
        wrong_stok += int(ps["wrong_confirm"])
        wrong_custom += int(pc["wrong_confirm"])
        plate_rows.append((v, ps, pc))
        smk_rows.append((v, ss, sc))

    print("\n--- PLAKA (PLATE_CONFIRMED event akışı; ana TOGG aracı) ---")
    print(
        f"{'video':9} | {'STOK plaka':12} {'st':9} CER  ex | {'CUSTOM plaka':12} {'st':9} CER  ex"
    )
    for v, ps, pc in plate_rows:
        print(
            f"{v:9} | {str(ps['value'] or ps['partial']):12} {str(ps['status']):9} "
            f"{ps['cer']:.2f} {'✓' if ps['exact'] else '·'}  | "
            f"{str(pc['value'] or pc['partial']):12} {str(pc['status']):9} "
            f"{pc['cer']:.2f} {'✓' if pc['exact'] else '·'}"
        )
    n = len(plate_rows)
    print(f"\nEXACT (3/3 hedefi):  STOK {exact_stok}/{n}  |  CUSTOM {exact_custom}/{n}")
    print(f"YANLIŞ ONAY:         STOK {wrong_stok}        |  CUSTOM {wrong_custom}")
    avg_cer_s = sum(p[1]["cer"] for p in plate_rows) / max(n, 1)
    avg_cer_c = sum(p[2]["cer"] for p in plate_rows) / max(n, 1)
    print(f"ORTALAMA CER:        STOK {avg_cer_s:.3f}        |  CUSTOM {avg_cer_c:.3f}")

    print("\n--- SİGARA / SÜRÜCÜ (DRIVER_STATE flag-event + RISK_ALERT kural) ---")
    print(f"{'video':9} | {'STOK smk/phone ev':18} RA-kural | {'CUSTOM smk/phone ev':18} RA-kural")
    for v, ss, sc in smk_rows:
        sx = f"{ss['smoking_events']}/{ss['phone_events']}"
        cx = f"{sc['smoking_events']}/{sc['phone_events']}"
        sr = ",".join(f"{k}:{n}" for k, n in ss["risk_rules"].items()) or "-"
        cr = ",".join(f"{k}:{n}" for k, n in sc["risk_rules"].items()) or "-"
        print(f"{v:9} | {sx:18} {sr:18} | {cx:18} {cr}")

    print(f"\n{'='*78}")
    no_regression = (
        exact_custom >= exact_stok and avg_cer_c <= avg_cer_s + 1e-9 and wrong_custom <= wrong_stok
    )
    verdict = (
        "REGRESYON YOK (custom >= stok; 3/3 exact + yanlış-onay korunur) → default'a terfi UYGUN"
        if no_regression
        else "REGRESYON VAR → profile-only bırak, nedeni belgele"
    )
    print(f"KARAR: {verdict}")
    print(f"{'='*78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
