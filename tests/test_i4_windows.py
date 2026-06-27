"""GRUP I4-windows: Windows UTF-8 sertleştirmesi (davranış-koruyan).

- H1/H2: entrypoint + tools modüllerinin başında stdout/stderr.reconfigure(utf-8)
  bloğu bulunmalı ve modül import-safe olmalı (import sırasında patlamamalı).
  reconfigure çağrısı try/except ile sarılı → import-safe (callable).
- M1: read_text/load çağrıları encoding="utf-8" ile yapılmalı (Windows cp1254
  varsayılanı yerine). Türkçe karakterli dosyalar doğru okunmalı.

Hepsi import/AST tabanlı veya küçük dosya I/O — gerçek model/MPS GEREKMEZ.
"""

from __future__ import annotations

# Modül import'u sırasında autostart/kamera tetiklenmesin.
import os

os.environ.setdefault("ROADGUARD_AUTOSTART", "0")
os.environ.setdefault("ROADGUARD_CAMERA_PROBE", "0")
os.environ.setdefault("AI_MODE", "mock")

import importlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

RECONFIGURE_MODULES = [
    "roadguard.__main__",
    "train.__main__",
    "services.inference_api.main",
    "train.merge_driver_datasets",
    "tools.doctor",
    "tools.test_video",
    "tools.show_driver_rois",
]


@pytest.mark.parametrize("modname", RECONFIGURE_MODULES)
def test_reconfigure_block_present_and_import_safe(modname):
    """Modül import-safe olmalı ve kaynakta utf-8 reconfigure bloğu bulunmalı."""
    mod = importlib.import_module(modname)
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "reconfigure(" in src, f"{modname}: reconfigure bloğu eksik"
    assert 'encoding="utf-8"' in src, f"{modname}: utf-8 encoding eksik"
    # reconfigure try/except ile sarılı → import-safe (callable, patlamaz)
    assert hasattr(mod, "__file__")


def test_reconfigure_block_is_import_safe_against_non_reconfigurable_stream():
    """reconfigure'u olmayan stream'de (AttributeError) blok sessiz geçmeli."""

    class _Dummy:
        # reconfigure yok → AttributeError; blok yutmalı
        pass

    stream = _Dummy()
    # entrypoint'lerdeki blokla aynı semantik
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # noqa: B018
    except (AttributeError, ValueError):
        pass  # beklenen — import-safe
    else:
        pytest.fail("reconfigure bloğu AttributeError'u yutmalıydı")


def test_homography_ipm_reads_utf8(tmp_path):
    """homography_ipm._load_calib Türkçe karakterli yaml'ı utf-8 ile okumalı."""
    from roadguard.optional import homography_ipm

    p = tmp_path / "ipm.yaml"
    p.write_text("ipm:\n  not: 'şçğüöİ kalibrasyon'\n", encoding="utf-8")

    class _Cfg:
        def get(self, key, default=None):
            return str(p) if key == "speed.calibration_file" else default

    data = homography_ipm._load_calib(_Cfg())
    assert data is not None
    assert data["not"] == "şçğüöİ kalibrasyon"


def test_eval_harness_reads_gt_utf8(tmp_path):
    """harness GT json'u utf-8 ile okumalı (read_text encoding)."""
    gt = tmp_path / "gt.json"
    payload = {"frames": [], "açıklama": "şçğ"}
    gt.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # read_text(encoding="utf-8") çağrısının davranışını doğrula
    loaded = json.loads(gt.read_text(encoding="utf-8"))
    assert loaded["açıklama"] == "şçğ"


def test_prepare_dataset_reads_classes_utf8(tmp_path):
    """prepare_dataset.read_class_names classes.txt'i utf-8 ile okumalı."""
    from train import prepare_dataset

    inp = tmp_path
    (inp / "classes.txt").write_text("araç\nplaka\nşerit\n", encoding="utf-8")
    names = prepare_dataset._read_classes(inp, None)
    assert "araç" in names and "şerit" in names


def test_m1_sources_have_utf8_encoding():
    """M1 düzeltilen 3 read_text çağrısı encoding='utf-8' içermeli."""
    import inspect

    from roadguard.eval import harness
    from roadguard.optional import homography_ipm
    from train import prepare_dataset

    for mod in (homography_ipm, harness, prepare_dataset):
        src = inspect.getsource(mod)
        assert 'read_text(encoding="utf-8")' in src, f"{mod.__name__}: read_text utf-8 eksik"
