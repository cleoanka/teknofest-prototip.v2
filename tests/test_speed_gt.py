"""Hız mutlak-GT doğrulaması (WP-A4): MAE/MAPE metrikleri + raporda
video-düzeyi ``real_speed_kmh`` okuma/sessiz-atlama davranışı.

gercek_hiz_plani.md §8.2: komite gerçek hız (radar/GPS) verdiğinde tahmin edilen
km/h ile karşılaştırılır. GT'de real_speed_kmh yoksa hız doğruluğu SESSİZCE atlanır
(sayı uydurma yok, K-004).
"""

from __future__ import annotations

from roadguard.eval import metrics as M
from roadguard.eval.report import (
    gt_label,
    pred_from_summary,
    render_markdown,
    speed_metrics,
)


# --- metrics.mae / mape (sentetik) ------------------------------------------ #
def test_mae_basic():
    # |50-48| + |60-63| + |40-40| = 2+3+0 → 5/3 ≈ 1.67
    assert M.mae([48.0, 63.0, 40.0], [50.0, 60.0, 40.0]) == round(5 / 3, 2)


def test_mape_basic():
    # (2/50 + 3/60 + 0/40)/3 *100 = (0.04+0.05+0)/3*100 ≈ 3.0
    assert M.mape([48.0, 63.0, 40.0], [50.0, 60.0, 40.0]) == 3.0


def test_mae_perfect_is_zero():
    assert M.mae([50.0, 60.0], [50.0, 60.0]) == 0.0
    assert M.mape([50.0, 60.0], [50.0, 60.0]) == 0.0


def test_mae_mape_empty_is_none():
    # Hiç örnek yok → None (sessiz atla, sayı uydurma yok)
    assert M.mae([], []) is None
    assert M.mape([], []) is None


def test_mape_skips_zero_truth():
    # Gerçek hız 0 olan örnek (sıfıra bölme) atlanır; geçerli kalan: |60-63|/60 → 5%
    assert M.mape([48.0, 63.0], [0.0, 60.0]) == 5.0


def test_mae_skips_none_pred():
    # Tahmin None (kalibre olmamış) örnek atlanır; kalan |60-63| → 3.0
    assert M.mae([None, 63.0], [50.0, 60.0]) == 3.0


# --- gt_label: real_speed_kmh oku / yoksa None ------------------------------ #
def test_gt_label_reads_real_speed():
    gt = {
        "real_speed_kmh": 52.5,
        "frames": [{"objects": [{"vehicle_class": "car", "plate": "34TC8532"}]}],
    }
    assert gt_label(gt)["real_speed_kmh"] == 52.5


def test_gt_label_no_real_speed_is_none():
    gt = {"frames": [{"objects": [{"vehicle_class": "car"}]}]}
    assert gt_label(gt)["real_speed_kmh"] is None


# --- pred_from_summary: kalibre hız medyanı --------------------------------- #
def test_pred_speed_uses_calibrated_median():
    summary = {
        "tracks": [
            {"speed_kmh": 50.0, "speed_is_calibrated": True, "driver_flag_frames": {}},
            {"speed_kmh": 54.0, "speed_is_calibrated": True, "driver_flag_frames": {}},
        ]
    }
    assert pred_from_summary(summary)["speed_kmh"] == 52.0  # medyan(50,54)


def test_pred_speed_ignores_uncalibrated():
    # is_calibrated False → metrik iddia yok; speed_kmh None (K-004)
    summary = {
        "tracks": [{"speed_kmh": 99.0, "speed_is_calibrated": False, "driver_flag_frames": {}}]
    }
    assert pred_from_summary(summary)["speed_kmh"] is None


# --- speed_metrics: eşleşme varsa MAE/MAPE, yoksa None ---------------------- #
def test_speed_metrics_pairs():
    pairs = [
        ({"real_speed_kmh": 50.0}, {"speed_kmh": 48.0}),
        ({"real_speed_kmh": 60.0}, {"speed_kmh": 63.0}),
    ]
    sm = speed_metrics(pairs)
    assert sm is not None
    assert sm["n"] == 2
    assert sm["mae_kmh"] == 2.5  # (2+3)/2
    assert sm["mape_pct"] == 4.5  # (4%+5%)/2


def test_speed_metrics_none_when_no_gt():
    # GT'de real_speed_kmh yok → None (rapor satırı sessizce atlanır)
    pairs = [({"real_speed_kmh": None}, {"speed_kmh": 48.0})]
    assert speed_metrics(pairs) is None


def test_speed_metrics_none_when_pred_uncalibrated():
    # GT var ama tahmin yok (kalibre olmadı) → eşleşme yok → None
    pairs = [({"real_speed_kmh": 50.0}, {"speed_kmh": None})]
    assert speed_metrics(pairs) is None


# --- render_markdown: satır var/sessiz atla --------------------------------- #
def _report_with_speed(speed):
    behavior = {
        b: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "accuracy": 1.0,
            "support": 0,
        }
        for b in ("phone", "smoking", "no_seatbelt", "fatigue", "swerving")
    }
    behavior["_macro_f1"] = 1.0
    return {
        "min_frames": 3,
        "detectors": {
            "yolo26l": {
                "videos": ["video_1"],
                "behavior": behavior,
                "plate": {
                    "correct": 1,
                    "total": 1,
                    "accuracy": 100.0,
                    "mean_cer": 0.0,
                    "confirmed": 1,
                    "partial": 0,
                },
                "speed": speed,
                "vehicle_class_accuracy": 100.0,
                "mean_fps": 20.0,
                "per_video": [],
            }
        },
    }


def test_render_includes_speed_line_when_present():
    md = render_markdown(_report_with_speed({"mae_kmh": 2.5, "mape_pct": 4.5, "n": 1}))
    assert "Hız doğruluğu (MAE/MAPE)" in md
    assert "MAE=2.5 km/h" in md


def test_render_omits_speed_line_when_absent():
    md = render_markdown(_report_with_speed(None))
    assert "Hız doğruluğu" not in md
