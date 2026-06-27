"""roadguard/eval/metrics.py kenar-durum + hata-yolu birim testleri (model gerektirmez).

İnceleme test_gaps'i kapatır: mae/mape None-dalları + tutarlı None-eleme,
detection_rate/small_object_rate kısa per_frame + den==0 + area_frac dalları.
Tüm testler saf-sözlük; cv2/torch/model çağrılmaz.
"""

from __future__ import annotations

from roadguard.eval import metrics as M


# --------------------------------------------------------------------------- #
# mae() — Mean Absolute Error
# --------------------------------------------------------------------------- #
def test_mae_basic():
    # |10-12| + |20-18| = 2 + 2 = 4; ort = 2.0
    assert M.mae([10.0, 20.0], [12.0, 18.0]) == 2.0


def test_mae_empty_returns_none():
    assert M.mae([], []) is None


def test_mae_all_preds_none_returns_none():
    # tahminlerin tümü None → eşleşme yok → None (sayı uydurma yok, K-004)
    assert M.mae([None, None], [10.0, 20.0]) is None


def test_mae_skips_none_pred():
    # ilk tahmin None elenir; kalan |30-30|=0 → 0.0
    assert M.mae([None, 30.0], [99.0, 30.0]) == 0.0


def test_mae_skips_none_truth_no_typeerror():
    # GT'de None varsa elenmeli — eskiden float(None) TypeError fırlatırdı (latent bug fix).
    assert M.mae([10.0, 20.0], [None, 22.0]) == 2.0


def test_mae_all_truth_none_returns_none():
    assert M.mae([10.0, 20.0], [None, None]) is None


def test_mae_strict_false_length_mismatch():
    # zip strict=False: kısa olana göre kesilir, hata atmaz
    assert M.mae([10.0, 20.0, 30.0], [10.0]) == 0.0


# --------------------------------------------------------------------------- #
# mape() — Mean Absolute Percentage Error
# --------------------------------------------------------------------------- #
def test_mape_basic():
    # |12-10|/10 = %20; |18-20|/20 = %10 → ort %15.0
    assert M.mape([12.0, 18.0], [10.0, 20.0]) == 15.0


def test_mape_empty_returns_none():
    assert M.mape([], []) is None


def test_mape_skips_zero_truth():
    # gerçek hız 0 → sıfıra bölme; o örnek ATLANIR. Kalan |12-10|/10 = %20
    assert M.mape([99.0, 12.0], [0.0, 10.0]) == 20.0


def test_mape_all_zero_truth_returns_none():
    assert M.mape([99.0], [0.0]) is None


def test_mape_skips_none_pred_and_truth():
    assert M.mape([None, 12.0], [10.0, 10.0]) == 20.0
    assert M.mape([12.0, 99.0], [10.0, None]) == 20.0


def test_mape_length_mismatch_strict_false():
    assert M.mape([12.0, 18.0, 50.0], [10.0]) == 20.0


# --------------------------------------------------------------------------- #
# detection_rate()
# --------------------------------------------------------------------------- #
def test_detection_rate_full():
    gt = {"frames": [{"objects": [{}, {}]}, {"objects": [{}]}]}
    # kare0: 2 GT, 2 tespit → min 2; kare1: 1 GT, 1 tespit → min 1 ⇒ 3/3 = %100
    assert M.detection_rate([2, 1], gt) == 100.0


def test_detection_rate_per_frame_shorter_than_gt():
    # per_frame GT kare sayısından KISA → eksik kareler 0 tespit sayılır
    gt = {"frames": [{"objects": [{}, {}]}, {"objects": [{}, {}]}]}
    # kare0: min(2,2)=2; kare1: per_frame yok → 0 ⇒ 2/4 = %50
    assert M.detection_rate([2], gt) == 50.0


def test_detection_rate_over_detection_capped():
    gt = {"frames": [{"objects": [{}]}]}
    # 5 tespit ama 1 GT → min(5,1)=1 ⇒ %100 (aşırı tespit oranı şişirmez)
    assert M.detection_rate([5], gt) == 100.0


def test_detection_rate_no_gt_objects_zero():
    # den==0 (hiç GT nesnesi yok) → 0.0 dalı
    gt = {"frames": [{"objects": []}]}
    assert M.detection_rate([3], gt) == 0.0


def test_detection_rate_no_frames_zero():
    assert M.detection_rate([], {"frames": []}) == 0.0


# --------------------------------------------------------------------------- #
# small_object_rate()
# --------------------------------------------------------------------------- #
def test_small_object_rate_counts_small_gt():
    # 640x360 → alan 230400; eşik %2 = 4608.
    # küçük bbox: 10x10=100 < eşik (küçük); büyük bbox: 200x200=40000 >= eşik (sayılmaz)
    gt = {
        "width": 640,
        "height": 360,
        "frames": [
            {
                "objects": [
                    {"bbox": [0, 0, 10, 10]},  # küçük
                    {"bbox": [0, 0, 200, 200]},  # büyük
                ]
            }
        ],
    }
    # 1 küçük GT, 1 küçük tespit → %100
    assert M.small_object_rate([1], gt) == 100.0


def test_small_object_rate_no_small_objects_zero():
    # hiç küçük nesne yok → den==0 → 0.0
    gt = {
        "width": 640,
        "height": 360,
        "frames": [{"objects": [{"bbox": [0, 0, 300, 300]}]}],
    }
    assert M.small_object_rate([0], gt) == 0.0


def test_small_object_rate_per_frame_shorter():
    gt = {
        "width": 640,
        "height": 360,
        "frames": [
            {"objects": [{"bbox": [0, 0, 10, 10]}]},
            {"objects": [{"bbox": [0, 0, 10, 10]}]},
        ],
    }
    # kare0: min(1,1)=1; kare1: per_frame yok → 0 ⇒ 1/2 = %50
    assert M.small_object_rate([1], gt) == 50.0


def test_small_object_rate_custom_area_frac():
    # area_frac büyütülünce 200x200=40000 da "küçük" sayılır (eşik %20 = 46080)
    gt = {
        "width": 640,
        "height": 360,
        "frames": [{"objects": [{"bbox": [0, 0, 200, 200]}]}],
    }
    assert M.small_object_rate([1], gt, area_frac=0.20) == 100.0


def test_small_object_rate_default_dimensions():
    # width/height verilmezse 640x360 varsayılır (eşik 4608)
    gt = {"frames": [{"objects": [{"bbox": [0, 0, 10, 10]}]}]}
    assert M.small_object_rate([1], gt) == 100.0


# --------------------------------------------------------------------------- #
# accuracy() — den==0 dalı (rapor dolaylı kullanır)
# --------------------------------------------------------------------------- #
def test_accuracy_zero_total():
    assert M.accuracy(0, 0, 0, 0) == 0.0


def test_accuracy_basic():
    assert M.accuracy(tp=3, tn=5, fp=1, fn=1) == 0.8
