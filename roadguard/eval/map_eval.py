"""İstatistiksel mAP/PR değerlendirme harness'ı — FTR §4 "geniş etiketli set".

`report.py` davranış-tespitinin *çalıştığının* kanıtını (3-videoluk held-out)
üretirken, bu modül geniş etiketli set geldiğinde ultralytics `val()` ile
istatistiksel mAP50-95 / mAP50 / Precision / Recall ve (varsa) sınıf-bazlı
tablo çıkarır. Markdown + JSON çıktısı FTR §4 tablolarına yapıştırılabilir.

DÜRÜSTLÜK (K-004): metrikler ultralytics'in döndürdüğü ölçümlerden birebir
alınır; hiçbir sayı uydurulmaz. ultralytics import edilemiyorsa veya ağırlık/
data dosyası yoksa exception fırlatmak yerine LOGLU uyarıp ``None`` döneriz —
böylece eval boru hattı (davranış raporu) kesilmez.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("roadguard.eval.map_eval")


def _safe_float(val) -> float | None:
    """ultralytics metriğini güvenli float'a çevir; çevirilemezse None."""
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def _class_table(metrics, names: dict | None) -> list[dict]:
    """Varsa sınıf-bazlı mAP tablosu üret (ultralytics box.maps + ap_class_index).

    ultralytics sürümleri arasında API oynar; her erişim try ile korunur ki
    eksik alan tüm raporu düşürmesin (boş tablo döneriz).
    """
    table: list[dict] = []
    box = getattr(metrics, "box", None)
    if box is None:
        return table
    # sınıf-başı mAP50-95 dizisi
    maps = getattr(box, "maps", None)
    if maps is None:
        return table
    idx = getattr(box, "ap_class_index", None)
    names = names or {}
    try:
        if idx is not None:
            # ap_class_index: değerlendirilen sınıfların indeksleri
            for ci in idx:
                ci = int(ci)
                table.append(
                    {
                        "class_id": ci,
                        "class_name": names.get(ci, str(ci)),
                        "map50_95": _safe_float(maps[ci]),
                    }
                )
        else:
            for ci, m in enumerate(maps):
                table.append(
                    {
                        "class_id": ci,
                        "class_name": names.get(ci, str(ci)),
                        "map50_95": _safe_float(m),
                    }
                )
    except (TypeError, IndexError, KeyError) as e:  # pragma: no cover - sürüm farkı koruması
        log.warning("Sınıf-bazlı mAP tablosu çıkarılamadı (atlanıyor): %s", e)
        return []
    return table


def _find_pr_curve(save_dir) -> str | None:
    """ultralytics'in ürettiği PR-curve PNG'sini bul (varsa mutlak yolu döner)."""
    if not save_dir:
        return None
    d = Path(save_dir)
    if not d.exists():
        return None
    # ultralytics tipik adlar: PR_curve.png, BoxPR_curve.png
    for name in ("PR_curve.png", "BoxPR_curve.png"):
        p = d / name
        if p.exists():
            return str(p.resolve())
    for p in sorted(d.glob("*PR_curve.png")):
        return str(p.resolve())
    return None


def _render_markdown(data: dict) -> str:
    """mAP sonuç sözlüğünden FTR §4 Markdown raporu üret."""
    lines = [
        "# RoadGuard — İstatistiksel mAP (geniş set) (FTR §4)",
        "",
        f"- **Ağırlık:** `{data['weights']}`",
        f"- **Veri tanımı (data.yaml):** `{data['data']}`",
        "- Metrikler ultralytics `YOLO.val()` çıktısından birebir alınmıştır "
        "(videoya-özel sabit yok).",
        "",
        "| Metrik | Değer |",
        "|---|---|",
        f"| mAP@50-95 | {data['map50_95']} |",
        f"| mAP@50 | {data['map50']} |",
        f"| Precision (ort.) | {data['precision']} |",
        f"| Recall (ort.) | {data['recall']} |",
        "",
    ]
    table = data.get("per_class") or []
    if table:
        lines += [
            "## Sınıf-bazlı mAP@50-95",
            "",
            "| Sınıf ID | Sınıf | mAP@50-95 |",
            "|---|---|---|",
        ]
        for row in table:
            lines.append(f"| {row['class_id']} | {row['class_name']} | {row['map50_95']} |")
        lines.append("")
    pr = data.get("pr_curve")
    if pr:
        lines += [f"- **PR eğrisi:** `{pr}`", ""]
    return "\n".join(lines) + "\n"


def run_map(
    weights: str | Path,
    data_yaml: str | Path,
    out_dir: str | Path = "eval_results",
) -> dict | None:
    """ultralytics `YOLO(weights).val(data=...)` → mAP raporu (md + json).

    Args:
        weights: YOLO ağırlık dosyası (.pt).
        data_yaml: ultralytics data tanım YAML'ı (val seti yolları + sınıflar).
        out_dir: rapor çıktı dizini (vars: ``eval_results``).

    Returns:
        Metrik sözlüğü ya da değerlendirme yapılamadıysa ``None`` (asla exception).
    """
    weights_p = Path(weights)
    data_p = Path(data_yaml)
    if not weights_p.exists():
        log.warning("Ağırlık bulunamadı, mAP atlanıyor: %s", weights_p)
        return None
    if not data_p.exists():
        log.warning("Veri tanımı (data.yaml) bulunamadı, mAP atlanıyor: %s", data_p)
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        log.warning(
            "ultralytics import edilemedi; mAP değerlendirmesi atlanıyor "
            "(pip install ultralytics). Davranış raporu etkilenmez."
        )
        return None

    try:
        model = YOLO(str(weights_p))
        metrics = model.val(data=str(data_p), verbose=False)
    except Exception as e:  # noqa: BLE001 - dış kütüphane; boru hattını kesme
        log.warning("ultralytics val() başarısız, mAP atlanıyor: %s", e)
        return None

    box = getattr(metrics, "box", None)
    if box is None:
        log.warning("val() sonucunda box metrikleri yok; mAP atlanıyor.")
        return None

    names = getattr(model, "names", None) or getattr(metrics, "names", None)
    data: dict = {
        "weights": str(weights_p),
        "data": str(data_p),
        "map50_95": _safe_float(getattr(box, "map", None)),
        "map50": _safe_float(getattr(box, "map50", None)),
        "precision": _safe_float(getattr(box, "mp", None)),
        "recall": _safe_float(getattr(box, "mr", None)),
        "per_class": _class_table(metrics, names if isinstance(names, dict) else None),
        "pr_curve": _find_pr_curve(getattr(metrics, "save_dir", None)),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = _render_markdown(data)
    (out / "map_report.md").write_text(md, encoding="utf-8")
    (out / "map_report.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("mAP raporu: %s", out / "map_report.md")
    return data
