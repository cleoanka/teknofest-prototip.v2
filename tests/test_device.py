"""aura/device.py cihaz çözümü — torch'u mock'layarak CUDA/MPS/CPU dalları.

CI cihazı CPU-only ya da MPS-only olabilir; gerçek GPU'ya bağımlı kalmamak için
``torch.cuda`` / ``torch.backends.mps`` monkeypatch'lenir. Hiçbir gerçek model
çalışmaz; yalnız device-seçim mantığı (D4 auto→mps düzeltmesi dahil) çitlenir.
"""

from __future__ import annotations

import sys
import types

import pytest

import aura.device as dev


@pytest.fixture(autouse=True)
def _clear_cache():
    # Her test öncesi/sonrası önbelleği temizle (resolve sonuçları sızmasın).
    dev.reset_cache()
    yield
    dev.reset_cache()


def _fake_torch(*, cuda_available=False, mps_available=False, smoke_ok=True):
    """Minik sahte torch modülü (cuda + backends.mps)."""
    t = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def get_device_capability(_i):
            return (12, 0)

        @staticmethod
        def get_arch_list():
            return ["sm_90"]

    class _MpsBackend:
        @staticmethod
        def is_available():
            return mps_available

    backends = types.SimpleNamespace(mps=_MpsBackend())

    def _zeros(_shape, device="cpu"):
        if device == "cuda" and not smoke_ok:
            raise RuntimeError("no kernel image is available")

        class _T:
            def __matmul__(self, other):
                return self

        return _T()

    t.cuda = _Cuda()
    t.backends = backends
    t.zeros = _zeros
    return t


def test_resolve_cpu_always_cpu():
    assert dev.resolve_device("cpu") == "cpu"


def test_resolve_mps_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=True))
    assert dev.resolve_device("mps") == "mps"


def test_resolve_mps_unavailable_falls_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=False))
    assert dev.resolve_device("mps") == "cpu"


def test_resolve_cuda_smoke_ok(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True, smoke_ok=True))
    assert dev.resolve_device("cuda") == "cuda:0"


def test_resolve_cuda_index_preserved(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True, smoke_ok=True))
    assert dev.resolve_device("cuda:1") == "cuda:1"


def test_resolve_cuda_smoke_fails_explicit_cuda_to_cpu(monkeypatch):
    # CUDA istendi ama kernel çalışmadı (capability-mesaj dalı + CPU'ya düş)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True, smoke_ok=False))
    assert dev.resolve_device("cuda") == "cpu"


def test_resolve_auto_falls_to_mps_when_no_cuda(monkeypatch):
    # D4 düzeltmesi: auto, CUDA yoksa MPS'i dener (macOS'ta artık CPU'ya takılmaz)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False, mps_available=True))
    assert dev.resolve_device("auto") == "mps"


def test_resolve_auto_falls_to_cpu_when_nothing(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda_available=False, mps_available=False)
    )
    assert dev.resolve_device("auto") == "cpu"


def test_resolve_none_defaults_to_auto(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=True))
    assert dev.resolve_device(None) == "mps"


def test_resolve_no_torch_falls_to_cpu(monkeypatch):
    # torch import edilemezse (ImportError) → CPU
    monkeypatch.setitem(sys.modules, "torch", None)
    assert dev.resolve_device("auto") == "cpu"
    assert dev.resolve_device("mps") == "cpu"


def test_resolve_cache_avoids_reprobe(monkeypatch):
    fake = _fake_torch(mps_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    first = dev.resolve_device("mps")
    # torch'u kaldırsak bile önbellekten döner (yeniden probe yok)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert dev.resolve_device("mps") == first


def test_reset_cache_clears(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=True))
    dev.resolve_device("mps")
    assert dev._resolved_cache  # dolu
    dev.reset_cache()
    assert dev._resolved_cache == {}


def test_cuda_is_usable_true(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True, smoke_ok=True))
    assert dev.cuda_is_usable() is True


def test_cuda_is_usable_false(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda_available=False, mps_available=False)
    )
    assert dev.cuda_is_usable() is False
