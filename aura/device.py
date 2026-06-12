"""Merkezi çalışma-zamanı cihaz çözümleyici.

Config/CLI `cuda` ya da `auto` istese bile, kurulu torch derlemesi GPU'nun compute
capability'sini gerçekten **çalıştırabiliyor mu** diye doğrularız. Doğrulayamıyorsa
(ör. yeni Blackwell `sm_120` GPU + yalnız `sm_90`'a kadar kernel taşıyan eski
`cu121` torch) sessizce CPU'ya düşeriz — "no kernel image is available" ile çöken
bir pipeline yerine çalışan bir pipeline.

Tek doğrulama noktası: tüm backend'ler (`detection/yolo.py`,
`driver_state/yolo.py`, `plate/ocr.py`) buradan geçer; cihaz seçim mantığı tek
yerde toplanır ve uyarı yalnızca bir kez loglanır.
"""

from __future__ import annotations

import logging

log = logging.getLogger("aura.device")

# (istek → çözülmüş cihaz) önbelleği: pahalı CUDA probe'unu ve tekrar eden
# uyarıları engeller.
_resolved_cache: dict[str, str] = {}


def _cuda_smoke_ok() -> bool:
    """GPU'da minik bir gerçek op çalıştır; 'no kernel image' vb. hataları yakalar.

    `torch.cuda.is_available()` True dönse bile kurulu derleme GPU mimarisini
    desteklemeyebilir; tek kesin test, fiilen bir kernel çalıştırmaktır.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        x = torch.zeros((8, 8), device="cuda")
        _ = x @ x  # gerçek bir CUDA kernel'i tetikler
        torch.cuda.synchronize()
        return True
    except Exception as e:  # noqa: BLE001 - her türlü CUDA/torch hatası → CPU'ya düş
        cap = ""
        try:
            import torch

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                arches = ", ".join(torch.cuda.get_arch_list())
                cap = (
                    f" (GPU sm_{major}{minor}; kurulu torch yalnızca: {arches}; "
                    "doğru sürüm için bootstrap.py'yi yeniden çalıştırın)"
                )
        except Exception:  # noqa: BLE001
            pass
        log.warning("CUDA kullanılamıyor, CPU'ya düşülüyor: %s%s", e, cap)
        return False


def cuda_is_usable() -> bool:
    """GPU bu torch derlemesiyle gerçekten kullanılabilir mi? (önbellekli)."""
    return resolve_device("cuda").startswith("cuda")


def resolve_device(requested: str | None) -> str:
    """İstenen cihazı çalıştırılabilir bir cihaz dizesine çöz.

    Dönüş: ``"cuda:0"`` | ``"cpu"`` | ``"mps"`` (ultralytics + EasyOCR uyumlu).

    - ``cpu``            → her zaman ``"cpu"``
    - ``cuda``/``auto``  → GPU gerçekten çalışıyorsa ``"cuda:0"``, aksi halde ``"cpu"``
    - ``mps``            → Apple MPS varsa ``"mps"``, aksi halde ``"cpu"``
    - ``cuda:N``         → indeks korunur (GPU çalışıyorsa)
    """
    key = (requested or "auto").strip().lower()
    if key in _resolved_cache:
        return _resolved_cache[key]

    resolved = _resolve_uncached(key)
    _resolved_cache[key] = resolved
    return resolved


def _resolve_uncached(key: str) -> str:
    if key == "cpu":
        return "cpu"

    if key == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        log.warning("MPS istendi ama kullanılamıyor → CPU")
        return "cpu"

    # cuda | auto | cuda:N
    if _cuda_smoke_ok():
        if key.startswith("cuda:"):
            return key
        return "cuda:0"

    if key in ("cuda",) or key.startswith("cuda:"):
        # Kullanıcı açıkça CUDA istedi ama çalışmıyor — uyarı _cuda_smoke_ok'ta verildi.
        return "cpu"
    # auto: CUDA yoksa Apple Silicon'da MPS'i dene (4K videoda CPU'ya göre kat kat
    # hızlı); o da yoksa CPU. Önceden auto yalnız CUDA'yı deniyordu ve macOS'ta
    # her zaman CPU'ya düşüyordu (D4 düzeltmesi).
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - torch yok/MPS probe hatası → CPU
        pass
    return "cpu"


def reset_cache() -> None:
    """Önbelleği temizle (test/yeniden yapılandırma için)."""
    _resolved_cache.clear()
