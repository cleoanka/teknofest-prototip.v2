"""QoD A/B eval harness + metrikler (model gerektirmez)."""

from __future__ import annotations

from pathlib import Path

from aura.eval import metrics as M
from aura.eval.harness import run_eval


def _ensure_sample() -> Path:
    s = Path("data/samples/ornek.mp4")
    if not s.exists():
        from aura.synthetic import generate

        generate(s.parent, 90, 30, 640, 360)
    return s


def test_levenshtein_and_cer():
    assert M.levenshtein("ABC", "ABC") == 0
    assert M.levenshtein("34ABC123", "34ABX123") == 1
    assert M.cer("34ABC123", "34ABC123") == 0.0
    assert M.cer("", "ABC") == 1.0


def test_plate_accuracy_metric():
    gt = {"frames": [{"objects": [{"plate": "34ABC123"}, {"plate": "06FY4571"}]}]}
    res = M.plate_accuracy({1: "34ABC123"}, gt)
    assert res["correct"] == 1 and res["gt_total"] == 2 and res["accuracy"] == 50.0


def test_run_eval_qod_delta_positive(cfg):
    src = _ensure_sample()
    res = run_eval(
        cfg, str(src), "data/samples/ornek_gt.json", qod_comparison=True, output_dir="eval_results"
    )
    assert "metrics" in res and len(res["metrics"]) >= 3
    plate = next(m for m in res["metrics"] if m["name"].startswith("Plaka"))
    assert plate["qod_on"] >= plate["qod_off"]  # QoD ON ≥ OFF (şartname kanıtı)
    assert Path("eval_results/report.md").exists()
    assert "QoD A/B" in Path("eval_results/report.md").read_text()


def test_run_eval_no_comparison_zero_delta(cfg):
    src = _ensure_sample()
    res = run_eval(cfg, str(src), "data/samples/ornek_gt.json", qod_comparison=False)
    assert all(m["delta_pct"] == 0 for m in res["metrics"])
