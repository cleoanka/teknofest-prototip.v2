"""Değerlendirme metrikleri — plaka doğruluğu/CER, tespit oranı, sürücü F1."""

from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    """Character Error Rate."""
    if not truth:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, truth) / len(truth)


def gt_plates(gt: dict) -> set[str]:
    plates = set()
    for f in gt.get("frames", []):
        for o in f.get("objects", []):
            if o.get("plate"):
                plates.add(o["plate"])
    return plates


def plate_accuracy(confirmed: dict[int, str], gt: dict) -> dict:
    """confirmed: {track_id: plate_value}. GT plakalarına karşı exact-match + CER."""
    truth = gt_plates(gt)
    preds = set(confirmed.values())
    correct = preds & truth
    acc = 100.0 * len(correct) / len(truth) if truth else 0.0
    # CER: en iyi eşleşmeye göre ortalama
    cers = []
    for p in preds:
        best = min((cer(p, t) for t in truth), default=1.0)
        cers.append(best)
    mean_cer = round(sum(cers) / len(cers), 3) if cers else 1.0
    return {
        "accuracy": round(acc, 1),
        "cer": mean_cer,
        "confirmed": len(preds),
        "correct": len(correct),
        "gt_total": len(truth),
    }


def detection_rate(per_frame_detected: list[int], gt: dict) -> float:
    """Kare-bazlı tespit oranı: Σ min(detected, gt) / Σ gt (%)."""
    frames = gt.get("frames", [])
    num = den = 0
    for i, f in enumerate(frames):
        gt_n = len(f.get("objects", []))
        det_n = per_frame_detected[i] if i < len(per_frame_detected) else 0
        num += min(det_n, gt_n)
        den += gt_n
    return round(100.0 * num / den, 1) if den else 0.0


def small_object_rate(
    per_frame_small_detected: list[int], gt: dict, area_frac: float = 0.02
) -> float:
    """Küçük (uzak) nesneler için tespit oranı — GT bbox alanı kare alanının %2'sinden küçük."""
    frames = gt.get("frames", [])
    W, H = gt.get("width", 640), gt.get("height", 360)
    thr = area_frac * W * H
    num = den = 0
    for i, f in enumerate(frames):
        gt_small = 0
        for o in f.get("objects", []):
            x1, y1, x2, y2 = o["bbox"]
            if (x2 - x1) * (y2 - y1) < thr:
                gt_small += 1
        det = per_frame_small_detected[i] if i < len(per_frame_small_detected) else 0
        num += min(det, gt_small)
        den += gt_small
    return round(100.0 * num / den, 1) if den else 0.0
