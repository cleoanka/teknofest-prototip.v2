"""FTR §4 metrik raporu (aura/eval/report.py) birim testleri."""

from __future__ import annotations

from aura.eval import metrics as M
from aura.eval.report import (
    behavior_metrics,
    gt_label,
    plate_metrics,
    pred_from_summary,
    vehicle_class_accuracy,
)


def test_prf1_basic():
    m = M.prf1(tp=3, fp=1, fn=0)
    assert m["precision"] == 0.75
    assert m["recall"] == 1.0
    assert round(m["f1"], 3) == 0.857


def test_prf1_no_positives_no_fp_is_perfect_avoidance():
    # GT'de pozitif yok ve FP de yok → recall 1.0 (mükemmel kaçınma konvansiyonu)
    m = M.prf1(tp=0, fp=0, fn=0)
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0


def test_gt_label_reads_driver_and_swerving():
    gt = {
        "swerving": True,
        "frames": [
            {
                "objects": [
                    {
                        "vehicle_class": "car",
                        "plate": "34TC8532",
                        "driver": {"phone": True, "smoking": False},
                    }
                ]
            }
        ],
    }
    lbl = gt_label(gt)
    assert lbl["vehicle_class"] == "car"
    assert lbl["plate"] == "34TC8532"
    assert lbl["phone"] is True
    assert lbl["smoking"] is False
    assert lbl["swerving"] is True


def test_pred_from_summary_thresholds_and_plate():
    summary = {
        "tracks": [
            {
                "vehicle_class": "car",
                "plate": "34TC8532",
                "driver_flag_frames": {"phone": 110, "smoking": 1},
                "swerving_frames": 0,
            }
        ]
    }
    pred = pred_from_summary(summary, min_frames=3)
    assert pred["phone"] is True  # 110 >= 3
    assert pred["smoking"] is False  # 1 < 3 (cross-FP elenir)
    assert pred["swerving"] is False
    assert pred["plate"] == "34TC8532"
    assert pred["plate_status"] == "confirmed"


def test_pred_partial_plate_when_not_confirmed():
    summary = {
        "tracks": [{"vehicle_class": "car", "plate_partial": "8532", "driver_flag_frames": {}}]
    }
    pred = pred_from_summary(summary)
    assert pred["plate"] == "8532"
    assert pred["plate_status"] == "partial"


def _matrix_pairs():
    """3-video beklenen matrisi (cross-FP sıfır): v1 sigara, v2 telefon, v3 swerving."""
    g1 = {
        "smoking": True,
        "phone": False,
        "swerving": False,
        "no_seatbelt": False,
        "fatigue": False,
    }
    g2 = {
        "phone": True,
        "smoking": False,
        "swerving": False,
        "no_seatbelt": False,
        "fatigue": False,
    }
    g3 = {
        "swerving": True,
        "phone": False,
        "smoking": False,
        "no_seatbelt": False,
        "fatigue": False,
    }
    return [(g1, dict(g1)), (g2, dict(g2)), (g3, dict(g3))]


def test_behavior_metrics_perfect_matrix():
    bm = behavior_metrics(_matrix_pairs())
    assert bm["phone"]["f1"] == 1.0
    assert bm["smoking"]["f1"] == 1.0
    assert bm["swerving"]["f1"] == 1.0
    assert bm["_macro_f1"] == 1.0


def test_behavior_metrics_cross_fp_penalised():
    pairs = _matrix_pairs()
    # video_1'de yanlışlıkla phone=True (cross-FP) → phone precision düşer
    g1, p1 = pairs[0]
    p1 = dict(p1)
    p1["phone"] = True
    pairs[0] = (g1, p1)
    bm = behavior_metrics(pairs)
    assert bm["phone"]["fp"] == 1
    assert bm["phone"]["precision"] < 1.0


def test_plate_and_vehicle_metrics():
    pairs = [
        (
            {"plate": "34TC8532", "vehicle_class": "car"},
            {"plate": "34TC8532", "plate_status": "confirmed", "vehicle_class": "car"},
        ),
        (
            {"plate": "34TC8532", "vehicle_class": "car"},
            {"plate": "8532", "plate_status": "partial", "vehicle_class": "car"},
        ),
    ]
    pm = plate_metrics(pairs)
    assert pm["total"] == 2
    assert pm["correct"] == 1  # biri tam eşleşme
    assert pm["confirmed"] == 1
    assert pm["partial"] == 1
    assert vehicle_class_accuracy(pairs) == 100.0
