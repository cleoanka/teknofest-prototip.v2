"""Eksik-sınıf veri seti manifestini oku → indirme PLANI bas (varsayılan kuru).

Bu modül `train/datasets.yaml` bildirimsel manifestini okur ve her hedef sınıf
için hangi açık veri setinin nereden, hangi lisansla çekileceğinin PLANINI basar.
**Varsayılan davranış AĞ KULLANMAZ** (`--dry`): yalnız planı ve sınıf-eşlemesini
gösterir. Gerçek indirme yalnız `--run` ile yapılır (roboflow kaynakları için
`train.roboflow_pull` çağrılır; kaggle/url kaynakları MANUEL — yalnız talimat basılır).

Sınıf-remap'i `aura.taxonomy` ile tutarlıdır: manifestteki `class_map` değerleri
(hedef kanonik adlar) taksonomi sözlüğüyle çapraz-kontrol edilir; tutarsızlık
PLANDA uyarı olarak işaretlenir (ONUR: belirsizlikte sessizce düzeltmez).

    python -m train fetch                  # tüm planı bas (kuru)
    python -m train fetch --class minibus  # tek sınıf
    python -m train fetch --run            # gerçek indirme (ROBOFLOW_API_KEY)
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from aura.taxonomy import canonical

log = logging.getLogger("aura.train.fetch")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "datasets.yaml"


def load_manifest(path: Path) -> dict:
    """Manifest YAML'ını oku ve temel yapı doğrulamasını yap."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest bulunamadı: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "targets" not in data or not isinstance(data["targets"], dict):
        raise ValueError(f"Manifestte 'targets' sözlüğü yok: {path}")
    return data


def _canonical_for(source: dict, aura_class: str) -> dict[str, str]:
    """Kaynak class_map'ini çöz: verilmişse onu, yoksa aura.taxonomy ile türet.

    Dönen sözlük {kaynak_sınıf_adı: kanonik_ad}. class_map boşsa hedef sınıfın
    kendi adı taksonomiden geçirilir (örn. cigarette -> smoking).
    """
    cmap = source.get("class_map") or {}
    if cmap:
        return {str(k): str(v) for k, v in cmap.items()}
    # class_map yok → hedef sınıf adından taksonomi ile türet (bilgilendirici).
    return {aura_class: canonical(aura_class)}


def _taxonomy_warnings(class_map: dict[str, str], aura_class: str) -> list[str]:
    """class_map'i aura.taxonomy ile çapraz-kontrol et; tutarsızlıkları topla.

    Kaynak sınıf adının taksonomideki kanonik karşılığı, manifestteki hedefle
    çelişiyorsa uyarı üretir (ONUR: sessiz düzeltme yok, planda görünür kılınır).

    İstisna: hedefin aura_class'ı taksonomi kanoniğine zaten eşitse (ör. seatbelt
    NESNESİ -> no_seatbelt_evidence; ihlal Katman B'de türetilir), kaynak sınıfın
    HAM adını eğitim sınıfı olarak tutmak BİLİNÇLİ bir karardır; uyarı bastırılır.
    """
    warns: list[str] = []
    for src_name, mapped in class_map.items():
        canon = canonical(src_name)
        if canon == src_name or canon == mapped:
            continue  # taksonomi src_name'i tanımıyor ya da hedefle uyumlu → sorun yok
        # Hedef sınıf, src_name'in kanoniğini zaten ÜSTLENMİŞ → ham adı tutmak bilinçli.
        if aura_class == canon:
            continue
        warns.append(
            f"taksonomi '{src_name}' -> '{canon}' diyor ama manifest '{mapped}' "
            f"(aura/taxonomy.py ile teyit edin)"
        )
    return warns


def build_plan(manifest: dict, only_class: str | None = None) -> list[dict]:
    """Manifestten indirme planı üret (saf veri; ağ yok).

    Her kaynak için bir adım: hedef sınıf, kaynak tipi/koordinatları, lisans,
    çözülmüş sınıf-eşlemesi ve taksonomi uyarıları. only_class verilirse filtreler.
    """
    targets: dict = manifest.get("targets", {})
    if only_class is not None and only_class not in targets:
        raise KeyError(f"Manifestte hedef sınıf yok: {only_class} (mevcut: {sorted(targets)})")

    plan: list[dict] = []
    for tgt_name, tgt in targets.items():
        if only_class is not None and tgt_name != only_class:
            continue
        aura_class = str((tgt or {}).get("aura_class", tgt_name))
        sources = (tgt or {}).get("sources") or []
        for src in sources:
            class_map = _canonical_for(src, aura_class)
            plan.append(
                {
                    "target": tgt_name,
                    "aura_class": aura_class,
                    "kind": src.get("kind", "?"),
                    "name": src.get("name", "?"),
                    "license": src.get("license", "?"),
                    "images": src.get("images"),
                    "source": src,
                    "class_map": class_map,
                    "warnings": _taxonomy_warnings(class_map, aura_class),
                }
            )
    return plan


def _output_dir(step: dict, base: Path) -> Path:
    """Bir kaynak için varsayılan indirme dizini (data/raw/<sınıf>/<kaynak-tipi-adı>)."""
    src = step["source"]
    if step["kind"] == "roboflow":
        leaf = f"{src.get('workspace', 'rf')}__{src.get('project', step['name'])}"
    else:
        leaf = str(src.get("ref") or src.get("name") or step["name"]).replace("/", "__")
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in leaf)
    return base / step["target"] / safe


def print_plan(plan: list[dict], base: Path) -> None:
    """Planı okunaklı bas (FTR §2/§5 için kopyalanabilir)."""
    if not plan:
        print("\n=== Veri Seti Çekme Planı: (boş) ===")
        print("  Hiç teyitli kaynak yok (manifestte sources: [] olan sınıflar atlandı).")
        return
    print("\n=== Veri Seti Çekme Planı (KURU — ağ kullanılmadı) ===")
    by_target: dict[str, list[dict]] = {}
    for step in plan:
        by_target.setdefault(step["target"], []).append(step)
    for tgt, steps in by_target.items():
        print(f"\n  hedef sınıf: {tgt}  (AURA: {steps[0]['aura_class']})")
        for step in steps:
            out = _output_dir(step, base)
            n = step["images"]
            n_str = f"~{n} görüntü" if n else "? görüntü"
            print(f"    - [{step['kind']}] {step['name']}  ({n_str}, lisans: {step['license']})")
            print(f"        eşleme: {step['class_map']}")
            print(f"        çıktı : {out}")
            for w in step["warnings"]:
                print(f"        ⚠ {w}")
    licenses = sorted({s["license"] for s in plan})
    print(f"\n  Kaynakça (FTR §5) lisansları: {', '.join(licenses)}")
    print("  Gerçek indirme için: python -m train fetch --run  (ROBOFLOW_API_KEY gerekir)")


def _run_step(step: dict, base: Path) -> int:
    """Tek bir kaynağı GERÇEKTEN indir (yalnız --run). roboflow → roboflow_pull."""
    import types

    out = _output_dir(step, base)
    if step["kind"] == "roboflow":
        from train.roboflow_pull import roboflow_pull

        src = step["source"]
        rf_args = types.SimpleNamespace(
            workspace=src["workspace"],
            project=src["project"],
            version=int(src.get("version", 1)),
            format=src.get("format", "yolov8"),
            output=str(out),
        )
        log.info("[%s] roboflow indirme → %s", step["name"], out)
        return roboflow_pull(rf_args)
    # kaggle / url: otomatik indirme yok (kimlik/lisans onayı gerekir) → talimat bas.
    ref = step["source"].get("ref") or step["source"].get("url") or "?"
    log.warning(
        "[%s] '%s' kaynağı MANUEL: '%s' adresinden indirip %s altına çıkarın "
        "(lisans: %s), sonra: python -m train dataset --input %s --output ... --report",
        step["name"],
        step["kind"],
        ref,
        out,
        step["license"],
        out,
    )
    return 0


def fetch(args) -> int:
    """fetch alt-komutu girişi: manifesti oku, planı bas, --run ise indir."""
    manifest_path = Path(getattr(args, "manifest", None) or DEFAULT_MANIFEST)
    base = Path(getattr(args, "output", None) or (ROOT / "data" / "raw"))
    try:
        manifest = load_manifest(manifest_path)
        plan = build_plan(manifest, only_class=getattr(args, "klass", None))
    except (FileNotFoundError, ValueError, KeyError) as e:
        log.error("%s", e)
        return 1

    print_plan(plan, base)

    if not getattr(args, "run", False):
        log.info("Kuru çalıştırma (--dry varsayılan): hiçbir şey indirilmedi.")
        return 0

    # --run: gerçek indirme.
    rc = 0
    for step in plan:
        rc |= _run_step(step, base)
    return 0 if rc == 0 else 1
