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
C_PRIMARY = "#1b5e20"  # yeşil — production/gerçek
C_ACCENT = "#0d47a1"  # mavi
C_BASE = "#9e9e9e"  # gri — baseline/eski
C_WARN = "#e65100"  # turuncu — dürüstlük/zırh
C_OK = "#2e7d32"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "figure.dpi": 140,
    }
)


def _load(p: str) -> dict:
    return json.loads((ROOT / p).read_text())


def _bars(ax, labels, values, colors, fmt="{:.3f}", rot=0):
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, values, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
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
PLATE_AB = {
    "Baseline\n(EasyOCR · stok/v4)": (2 / 3 * 100, 0.083),
    "Production\n(custom_LP + fast-plate-ocr)": (3 / 3 * 100, 0.0),
}
# FPS — MPS geliştirme alt-sınırı vs gerçek CUDA ölçümü (RTX 5070 Laptop, 4.608 çekirdek,
# 2026-06-26; eval_results/bench_cuda0_{server,laptop}.md). Belgeli ölçüm sabitleri.
FPS_CMP = [
    ("yolo26l · server\nMPS (M4 Pro)", 5.89, C_BASE),
    ("yolo26l · server\nCUDA (RTX 5070)", 12.31, C_PRIMARY),
    ("yolo26s · laptop\nCUDA (RTX 5070)", 14.72, C_OK),
]

print(f"Figürler → {FIG.relative_to(ROOT)}/")

# --- §2.3 Veri dengeleme: sınıf dağılımı ---
fig, ax = plt.subplots(figsize=(7, 4))
ks = list(DATASETS)
_bars(ax, ks, [DATASETS[k] for k in ks], [C_PRIMARY, C_ACCENT, C_WARN, C_BASE], fmt="{:,}")
ax.set_ylabel("Görüntü sayısı")
ax.set_title("§2.3 Açık-kaynak veri seti dağılımı (CC BY 4.0)")
_save(
    fig,
    "fig_veri_dengesi.png",
    "Kaynak: docs/veri_seti.md — PIL-doğrulanmış, AURA taksonomisine eşlenmiş. seatbelt dengesizlik oranı 1,27 (<3).",
)

# --- §2.5 Train/Val/Test split (license_plate örneği) ---
fig, ax = plt.subplots(figsize=(6.5, 3.2))
parts = list(LP_SPLIT)
vals = [LP_SPLIT[p] for p in parts]
left = 0
colors = [C_OK, C_ACCENT, C_WARN]
for p, v, c in zip(parts, vals, colors, strict=True):
    ax.barh(
        0,
        v,
        left=left,
        color=c,
        edgecolor="black",
        linewidth=0.5,
        label=f"{p} ({v:,} · {v/sum(vals)*100:.0f}%)",
    )
    ax.text(left + v / 2, 0, f"{v:,}", ha="center", va="center", color="white", fontweight="bold")
    left += v
ax.set_yticks([])
ax.set_xlabel("Görüntü sayısı")
ax.set_title("§2.5 Train/Val/Test dağılımı — %80/%10/%10 (license_plate, 8.823 görsel)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
_save(
    fig,
    "fig_split.png",
    "Gerekçe: küçük özel setlerde val+test'in istatistiksel anlamı için her birine %10; seed 42.",
)

# --- §4.2 Tespit doğruluğu: held-out mAP ---
fig, ax = plt.subplots(figsize=(8, 4.5))
models = [
    "license_plate\n(YOLO26s)",
    "seatbelt\n(YOLO26s)",
    "smoking\n(YOLO26s)",
    "yolo26l\n(COCO genel)",
]
map50 = [lp["mAP50"], sb["mAP50"], sm["mAP50"], stock["map50"]]
map5095 = [lp["mAP50_95"], sb["mAP50_95"], sm["mAP50_95"], stock["map50_95"]]
x = range(len(models))
w = 0.38
b1 = ax.bar(
    [i - w / 2 for i in x],
    map50,
    w,
    label="mAP@0.50",
    color=C_PRIMARY,
    edgecolor="black",
    linewidth=0.5,
)
b2 = ax.bar(
    [i + w / 2 for i in x],
    map5095,
    w,
    label="mAP@0.50:0.95",
    color=C_ACCENT,
    edgecolor="black",
    linewidth=0.5,
)
for bars in (b1, b2):
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{b.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylim(0, 1.08)
ax.set_ylabel("mAP")
ax.set_title("§4.2 Held-out tespit doğruluğu (kesinleşmiş, 19 Haz 2026)")
ax.legend(loc="upper right")
_save(
    fig,
    "fig_tespit_map.png",
    "Custom: weights/custom_*_s.metrics.json (domain held-out test bölmesi). yolo26l: COCO val2017 5.000 görsel — FARKLI/genel set.",
)

# --- §4.2/§4.3 Custom P/R/F1 ---
fig, ax = plt.subplots(figsize=(8, 4.5))
metr = ["Precision", "Recall", "F1"]
data = {
    "license_plate": [lp["precision"], lp["recall"], lp["f1"]],
    "seatbelt": [sb["precision"], sb["recall"], sb["f1"]],
    "smoking": [sm["precision"], sm["recall"], sm["f1"]],
}
x = range(len(metr))
w = 0.26
cols = [C_PRIMARY, C_ACCENT, C_WARN]
for i, (name, vals) in enumerate(data.items()):
    bars = ax.bar(
        [xi + (i - 1) * w for xi in x],
        vals,
        w,
        label=name,
        color=cols[i],
        edgecolor="black",
        linewidth=0.5,
    )
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{b.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
ax.set_xticks(list(x))
ax.set_xticklabels(metr)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Skor")
ax.set_title("§4.2 Özel model held-out Precision / Recall / F1 (YOLO26s)")
ax.legend(loc="lower right")
_save(
    fig,
    "fig_custom_prf1.png",
    "Kaynak: weights/custom_*_s.metrics.json (Ultralytics model.val ayrılmış test bölmesi).",
)

# --- §4.4 Plaka okuma A/B (neden custom+fastplate) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = list(PLATE_AB)
acc = [PLATE_AB[k][0] for k in labels]
cer = [PLATE_AB[k][1] for k in labels]
bars = _bars(ax, labels, acc, [C_BASE, C_PRIMARY], fmt="{:.1f}%")
for b, c in zip(bars, cer, strict=True):
    ax.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() - 8,
        f"CER {c:.3f}",
        ha="center",
        va="top",
        fontsize=9,
        color="white",
        fontweight="bold",
    )
ax.set_ylim(0, 112)
ax.set_ylabel("Exact-match doğruluğu (%)")
ax.set_title("§4.4 Plaka okuma A/B — 3 gerçek video (GT=34TC8532)")
_save(
    fig,
    "fig_plaka_ab.png",
    "fast-plate-ocr + custom_license_plate, video_3 misread'ini (24IC8532) düzeltip 3/3 CER 0.0; sıfır yanlış-onay. Kaynak: docs/degerlendirme.md.",
)

# --- §4.6 İşleme hızı (FPS) — MPS alt-sınırı vs gerçek CUDA ---
fig, ax = plt.subplots(figsize=(7, 4.2))
labels = [c[0] for c in FPS_CMP]
vals = [c[1] for c in FPS_CMP]
cols = [c[2] for c in FPS_CMP]
_bars(ax, labels, vals, cols, fmt="{:.2f}")
ax.set_ylabel("Ortalama FPS (kararlı-hal)")
ax.set_ylim(0, 17)
ax.set_title("§4.6 İşleme hızı — MPS geliştirme alt-sınırı vs gerçek CUDA ölçümü")
ax.annotate(
    "CUDA ≈ 2× MPS",
    xy=(1, 12.31),
    xytext=(0.45, 15.3),
    fontsize=10,
    fontweight="bold",
    color=C_PRIMARY,
    arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=1.3),
)
_save(
    fig,
    "fig_fps.png",
    "Gerçek ölçüm: RTX 5070 Laptop GPU (4.608 CUDA çekirdeği, 8 GB, Compute 12.0), 2026-06-26. "
    "Kaynak: eval_results/bench_cuda0_{server,laptop}.md.",
)

# --- §3.2 Çözüm mimarisi (graphviz dot → profesyonel diyagram) ---
import shutil  # noqa: E402
import subprocess  # noqa: E402

dot_src = ROOT / "docs" / "diagrams" / "pipeline_mimari.dot"
if shutil.which("dot") and dot_src.exists():
    subprocess.run(
        ["dot", "-Tpng", "-Gdpi=200", str(dot_src), "-o", str(FIG / "fig_mimari.png")], check=True
    )
    print(f"  ✓ {(FIG / 'fig_mimari.png').relative_to(ROOT)} (graphviz)")
else:
    print(
        "  ⚠ graphviz 'dot' bulunamadı — fig_mimari.png atlandı "
        "(docs/diagrams/pipeline_mimari.dot'tan elle render edin)"
    )

print("Tamamlandı.")
