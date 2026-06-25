#!/usr/bin/env python3
"""FTR §2/§4 grafiklerini GERÇEK ölçülen verilerden üretir (docs/figures/*.png).

Onur zırhı (K-004): doğruluk sayıları elle yazılmaz — `weights/custom_*_s.metrics.json`
ve `eval_results/map_yolo26l.json` dosyalarından OKUNUR. Veri-seti sayıları ve plaka A/B
gibi belgeli sabitler kaynak yorumuyla işaretlenir (docs/veri_seti.md, docs/degerlendirme.md).
Jüri figürleri tek komutla yeniden üretebilir:  python tools/make_ftr_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Renk paleti (TEKNOFEST/AURA — koyu kırmızı vurgulu, profesyonel)
C_PRIMARY = "#1b5e20"   # yeşil — production/gerçek
C_ACCENT = "#0d47a1"    # mavi
C_BASE = "#9e9e9e"      # gri — baseline/eski
C_WARN = "#e65100"      # turuncu — dürüstlük/zırh
C_OK = "#2e7d32"

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True, "figure.dpi": 140})


def _load(p: str) -> dict:
    return json.loads((ROOT / p).read_text())


def _bars(ax, labels, values, colors, fmt="{:.3f}", rot=0):
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                fmt.format(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    if rot:
        plt.setp(ax.get_xticklabels(), rotation=rot, ha="right")
    return bars


def _save(fig, name, caption):
    fig.text(0.5, 0.005, caption, ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = FIG / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.relative_to(ROOT)}")


# === Gerçek ölçülen değerler (JSON'dan) ===
lp = _load("weights/custom_license_plate_s.metrics.json")
sm = _load("weights/custom_smoking_s.metrics.json")
sb = _load("weights/custom_seatbelt_s.metrics.json")
stock = _load("eval_results/map_yolo26l.json")["coco_val2017_heldout"]

# === Belgeli sabitler (kaynak: docs/veri_seti.md, docs/degerlendirme.md, config ölçüm notu) ===
DATASETS = {"license_plate": 8823, "seatbelt": 3104, "phone": 659, "smoking": 557}  # CC BY 4.0
LP_SPLIT = {"train": 6176, "val": 1765, "test": 882}  # %80/%10/%10 (docs/veri_seti.md)
# Plaka exact-match A/B (3 gerçek video, GT=34TC8532; docs/degerlendirme.md, config ölçüm notu):
PLATE_AB = {"Baseline\n(EasyOCR · stok/v4)": (2 / 3 * 100, 0.083),
            "Production\n(custom_LP + fast-plate-ocr)": (3 / 3 * 100, 0.0)}
FPS = {"yolo26l (stok)": 5.89, "v4-finetune (A/B tabanı)": 5.34}  # MPS, M4 Pro (metrics_report)

print(f"Figürler → {FIG.relative_to(ROOT)}/")

# --- §2.3 Veri dengeleme: sınıf dağılımı ---
fig, ax = plt.subplots(figsize=(7, 4))
ks = list(DATASETS)
_bars(ax, ks, [DATASETS[k] for k in ks],
      [C_PRIMARY, C_ACCENT, C_WARN, C_BASE], fmt="{:,}")
ax.set_ylabel("Görüntü sayısı")
ax.set_title("§2.3 Açık-kaynak veri seti dağılımı (CC BY 4.0)")
_save(fig, "fig_veri_dengesi.png",
      "Kaynak: docs/veri_seti.md — PIL-doğrulanmış, AURA taksonomisine eşlenmiş. seatbelt dengesizlik oranı 1,27 (<3).")

# --- §2.5 Train/Val/Test split (license_plate örneği) ---
fig, ax = plt.subplots(figsize=(6.5, 3.2))
parts = list(LP_SPLIT)
vals = [LP_SPLIT[p] for p in parts]
left = 0
colors = [C_OK, C_ACCENT, C_WARN]
for p, v, c in zip(parts, vals, colors):
    ax.barh(0, v, left=left, color=c, edgecolor="black", linewidth=0.5,
            label=f"{p} ({v:,} · {v/sum(vals)*100:.0f}%)")
    ax.text(left + v / 2, 0, f"{v:,}", ha="center", va="center", color="white", fontweight="bold")
    left += v
ax.set_yticks([])
ax.set_xlabel("Görüntü sayısı")
ax.set_title("§2.5 Train/Val/Test dağılımı — %80/%10/%10 (license_plate, 8.823 görsel)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
_save(fig, "fig_split.png",
      "Gerekçe: küçük özel setlerde val+test'in istatistiksel anlamı için her birine %10; seed 42.")

# --- §4.2 Tespit doğruluğu: held-out mAP ---
fig, ax = plt.subplots(figsize=(8, 4.5))
models = ["license_plate\n(YOLO26s)", "seatbelt\n(YOLO26s)", "smoking\n(YOLO26s)", "yolo26l\n(COCO genel)"]
map50 = [lp["mAP50"], sb["mAP50"], sm["mAP50"], stock["map50"]]
map5095 = [lp["mAP50_95"], sb["mAP50_95"], sm["mAP50_95"], stock["map50_95"]]
x = range(len(models))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], map50, w, label="mAP@0.50", color=C_PRIMARY, edgecolor="black", linewidth=0.5)
b2 = ax.bar([i + w / 2 for i in x], map5095, w, label="mAP@0.50:0.95", color=C_ACCENT, edgecolor="black", linewidth=0.5)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylim(0, 1.08)
ax.set_ylabel("mAP")
ax.set_title("§4.2 Held-out tespit doğruluğu (kesinleşmiş, 19 Haz 2026)")
ax.legend(loc="upper right")
_save(fig, "fig_tespit_map.png",
      "Custom: weights/custom_*_s.metrics.json (domain held-out test bölmesi). yolo26l: COCO val2017 5.000 görsel — FARKLI/genel set.")

# --- §4.2/§4.3 Custom P/R/F1 ---
fig, ax = plt.subplots(figsize=(8, 4.5))
metr = ["Precision", "Recall", "F1"]
data = {"license_plate": [lp["precision"], lp["recall"], lp["f1"]],
        "seatbelt": [sb["precision"], sb["recall"], sb["f1"]],
        "smoking": [sm["precision"], sm["recall"], sm["f1"]]}
x = range(len(metr))
w = 0.26
cols = [C_PRIMARY, C_ACCENT, C_WARN]
for i, (name, vals) in enumerate(data.items()):
    bars = ax.bar([xi + (i - 1) * w for xi in x], vals, w, label=name, color=cols[i], edgecolor="black", linewidth=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(list(x))
ax.set_xticklabels(metr)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Skor")
ax.set_title("§4.2 Özel model held-out Precision / Recall / F1 (YOLO26s)")
ax.legend(loc="lower right")
_save(fig, "fig_custom_prf1.png", "Kaynak: weights/custom_*_s.metrics.json (Ultralytics model.val ayrılmış test bölmesi).")

# --- §4.4 Plaka okuma A/B (neden custom+fastplate) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = list(PLATE_AB)
acc = [PLATE_AB[k][0] for k in labels]
cer = [PLATE_AB[k][1] for k in labels]
bars = _bars(ax, labels, acc, [C_BASE, C_PRIMARY], fmt="{:.1f}%")
for b, c in zip(bars, cer):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 8, f"CER {c:.3f}",
            ha="center", va="top", fontsize=9, color="white", fontweight="bold")
ax.set_ylim(0, 112)
ax.set_ylabel("Exact-match doğruluğu (%)")
ax.set_title("§4.4 Plaka okuma A/B — 3 gerçek video (GT=34TC8532)")
_save(fig, "fig_plaka_ab.png",
      "fast-plate-ocr + custom_license_plate, video_3 misread'ini (24IC8532) düzeltip 3/3 CER 0.0; sıfır yanlış-onay. Kaynak: docs/degerlendirme.md.")

# --- §4.6 İşleme hızı (FPS) ---
fig, ax = plt.subplots(figsize=(6.5, 4))
_bars(ax, list(FPS), list(FPS.values()), [C_PRIMARY, C_BASE], fmt="{:.1f}")
ax.set_ylabel("Ortalama FPS")
ax.set_title("§4.6 İşleme hızı — MPS (Apple M4 Pro), alt-sınır")
_save(fig, "fig_fps.png", "MPS alt-sınırdır; CUDA sunucuda belirgin daha yüksektir. Kaynak: eval_results/metrics_report.md.")

# --- §3.2 Çözüm mimarisi (kuşbakışı boru hattı) ---
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

fig, ax = plt.subplots(figsize=(9.5, 6.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("§3.2 Çözüm Mimarisi — kuşbakışı kaskad boru hattı", fontsize=13, fontweight="bold")


def box(x, y, w, h, text, color, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                linewidth=1.4, edgecolor=color, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, color="#111", wrap=True)


def arrow(x1, y1, x2, y2, color="#444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.3, color=color))


box(3.2, 9.0, 3.6, 0.8, "Kamera / Video / RTSP", C_ACCENT, "#e3f2fd")
box(3.2, 7.9, 3.6, 0.8, "Ön-İşleme\n(far-glare · blur · occlusion)", C_PRIMARY, "#e8f5e9")
box(2.7, 6.8, 4.6, 0.8, "YOLO26l + ByteTrack\n+ alan-ağırlıklı sınıf-oyu", C_PRIMARY, "#e8f5e9")
box(0.3, 5.0, 4.3, 1.3, "Sürücü ROI (Aşama 2a)\nKatman A: pose-hibrit / YOLO26\nKatman B: per-ID 16/8 oylama", C_PRIMARY, "#e8f5e9")
box(5.4, 5.0, 4.3, 1.3, "Plaka ROI (Aşama 2b)\ncustom YOLO26s LP + oy havuzu\n+ fast-plate-ocr (+ veto/zemin)", C_PRIMARY, "#e8f5e9")
box(3.2, 3.4, 3.6, 0.85, "Hız + Swerving\n(Kalman+EMA · ZigZag)", C_PRIMARY, "#e8f5e9")
box(2.7, 2.1, 4.6, 0.85, "ID-merkezli Accumulator\n+ risk kuralları", C_PRIMARY, "#e8f5e9")
box(2.4, 0.7, 5.2, 0.85, "Event / Annotation → Dashboard · Mobil · JSONL kanıt", "#4a148c", "#f3e5f5")
box(7.7, 7.6, 2.1, 1.4, "QoD tetik\nyaklaşma /\nkalite / anomali\n→ CAMARA QoD", C_WARN, "#fff3e0")

arrow(5.0, 9.0, 5.0, 8.7)
arrow(5.0, 7.9, 5.0, 7.6)
arrow(4.6, 6.8, 2.6, 6.3)
arrow(5.4, 6.8, 7.4, 6.3)
arrow(2.4, 5.0, 4.0, 4.25)
arrow(7.6, 5.0, 6.0, 4.25)
arrow(5.0, 3.4, 5.0, 2.95)
arrow(5.0, 2.1, 5.0, 1.55)
arrow(7.6, 7.6, 7.3, 6.0, C_WARN)
ax.text(0.2, 4.6, "Kaskad: ağır model yalnız 2 ROI'de çalışır (kabin + plaka)", ha="left",
        fontsize=7.2, style="italic", color="#666")
_save(fig, "fig_mimari.png",
      "Gerçek↔mock sınırı: YZ çekirdeği gerçek; QoD/NV telekom katmanı CAMARA sözleşmesini taklit eder. Kaynak: docs/diagrams/.")

print("Tamamlandı.")
