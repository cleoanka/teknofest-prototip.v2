#!/usr/bin/env python3
"""AURA kurulum bootstrap'i — saf Python stdlib.

Tek komutla sıfırdan çalışır hale getirir: sanal ortam, donanıma uygun torch
backend'i, paket bağımlılıkları, model ağırlıkları (SHA256 doğrulamalı), örnek
veri ve smoke test. İdempotenttir; ikinci çalıştırma tamamlanmış adımları atlar.

Bu dosya hiçbir üçüncü-parti modüle bağımlı DEĞİLDİR (sistem python'u ile çalışır).
OpenCV/numpy gerektiren işler (örnek video üretimi, smoke) venv python'una
subprocess ile devredilir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn

# Windows konsolu cp1254/cp437 gibi olabilir; UTF-8'e geç ki kutu çizimi ve
# sembol çıktısı (╔ ▶ ✓ ⚠ █) UnicodeEncodeError ile çökmesin.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
WEIGHTS_DIR = ROOT / "weights"
SAMPLES_DIR = ROOT / "data" / "samples"
MIN_PY = (3, 10)

# Model ağırlıkları — bootstrap tarafından indirilir, .gitignore'lu.
# sha256 None => "trust on first use": ilk indirmede hash hesaplanır ve
# weights/weights.lock.json'a yazılır; sonraki çalıştırmalar buna karşı doğrular.
WEIGHTS: dict[str, dict] = {
    "yolo26s.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v9.0.0/yolo26s.pt",
        "sha256": None,
    },
    "yolo26l.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v9.0.0/yolo26l.pt",
        "sha256": None,
    },
}

# --------------------------------------------------------------------------- #
# Çıktı yardımcıları (ANSI; NO_COLOR veya tty değilse düz)
# --------------------------------------------------------------------------- #
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, msg: str) -> str:
    return f"\033[{code}m{msg}\033[0m" if _COLOR else msg


def step(msg: str) -> None:
    print(_c("1;36", f"▶ {msg}"))


def ok(msg: str) -> None:
    print(_c("1;32", f"  ✓ {msg}"))


def warn(msg: str) -> None:
    print(_c("1;33", f"  ⚠ {msg}"))


def fail(msg: str) -> None:
    print(_c("1;31", f"  ✗ {msg}"))


def die(msg: str, hint: str = "") -> NoReturn:
    fail(msg)
    if hint:
        print(f"    → {hint}")
    sys.exit(1)


# --------------------------------------------------------------------------- #
def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Komutu çalıştır; çıktıyı akıt. check=True ile hata fırlatır."""
    kw.setdefault("cwd", ROOT)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


# --- 3.1 Sistem doğrulama ------------------------------------------------- #
def check_system() -> None:
    step("Sistem doğrulama")
    if sys.version_info < MIN_PY:
        die(
            f"Python {MIN_PY[0]}.{MIN_PY[1]}+ gerekli (mevcut {sys.version.split()[0]}).",
            "Kurulum rehberi: docs/kurulum.md",
        )
    ok(f"Python {sys.version.split()[0]}")
    if shutil.which("git") is None:
        warn("git bulunamadı — klonlama/versiyonlama sınırlı olabilir.")
    else:
        ok("git mevcut")
    ok(f"Platform: {platform.system()} {platform.machine()}")


# --- 3.2 Sanal ortam ------------------------------------------------------- #
def ensure_venv(force: bool) -> None:
    step("Sanal ortam (.venv)")
    if force and VENV.exists():
        warn(".venv siliniyor (--force)")
        shutil.rmtree(VENV)
    if VENV.exists():
        ok(".venv mevcut (atlandı)")
        return
    run([sys.executable, "-m", "venv", str(VENV)])
    run([venv_python(), "-m", "pip", "install", "--quiet", "--upgrade", "pip", "wheel"])
    ok(".venv oluşturuldu")


# --- 3.3 Torch backend tespiti + kurulum ----------------------------------- #
def detect_backend() -> tuple[str, list[str]]:
    """(backend_adı, pip_index_args) döndür.

    CUDA index'i cu128: Turing(sm_75)'ten Blackwell(sm_120)'e kadar tüm modern
    NVIDIA GPU'larını kapsar. Eski cu121, yeni GPU'larda "no kernel image" verir.
    """
    system, machine = platform.system(), platform.machine()
    if system == "Darwin" and machine == "arm64":
        return "mps", []  # Apple Silicon — varsayılan index, MPS otomatik
    if shutil.which("nvidia-smi"):
        return "cuda", ["--index-url", "https://download.pytorch.org/whl/cu128"]
    return "cpu", ["--index-url", "https://download.pytorch.org/whl/cpu"]


# Blackwell (sm_120) dahil tüm modern NVIDIA GPU'ları destekleyen, test edilmiş
# sabit sürüm çifti (cu128). Şartname donanımı RTX 5070 (sm_120) bunu gerektirir;
# eski cu121 derlemesi "no kernel image is available" ile çöker.
TORCH_PIN_CUDA = ["torch==2.8.0", "torchvision==0.23.0"]

# torch import edilebiliyor olması yetmez: CUDA backend'inde GPU'da gerçek bir
# kernel çalışabilmeli. Exit kodları: 0=GPU çalışıyor, 3=GPU yok (CPU derlemesi),
# 4=kernel yok (yanlış/eski derleme).
_TORCH_GPU_PROBE = (
    "import sys, torch, torchvision\n"
    "if not torch.cuda.is_available():\n"
    "    sys.exit(3)\n"
    "try:\n"
    "    x = torch.zeros((8, 8), device='cuda'); _ = x @ x; torch.cuda.synchronize()\n"
    "except Exception:\n"
    "    sys.exit(4)\n"
    "sys.exit(0)\n"
)


def install_torch(skip: bool) -> str:
    step("Torch backend")
    backend, index_args = detect_backend()
    if skip:
        warn(f"torch kurulumu atlandı (--skip-deps); tespit edilen backend: {backend}")
        return backend

    imported = (
        subprocess.run(
            [str(venv_python()), "-c", "import torch, torchvision"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )
    if imported:
        if backend != "cuda":
            ok(f"torch zaten kurulu (backend: {backend})")
            return backend
        # CUDA: sadece import değil, bu GPU'da fiilen çalıştığını da doğrula.
        probe = subprocess.run(
            [str(venv_python()), "-c", _TORCH_GPU_PROBE], cwd=ROOT, capture_output=True
        )
        if probe.returncode == 0:
            ok("torch zaten kurulu ve GPU çalışıyor (backend: cuda)")
            return backend
        if probe.returncode == 3:
            warn("torch kurulu ama GPU görünmüyor (CPU derlemesi?) — CUDA (cu128) derlemesi kuruluyor")
        else:
            warn(
                "torch kurulu ama bu GPU'da kernel yok (eski/yanlış derleme) — "
                "doğru CUDA (cu128) derlemesi kuruluyor"
            )

    ok(f"Tespit edilen backend: {backend}")
    pkgs = TORCH_PIN_CUDA if backend == "cuda" else ["torch", "torchvision"]
    run([venv_python(), "-m", "pip", "install", "--upgrade", *pkgs, *index_args])
    ok("torch + torchvision kuruldu")
    return backend


# --- 3.4 Paket kurulumu ---------------------------------------------------- #
def install_package(dev: bool, skip: bool) -> None:
    step("AURA paketi + bağımlılıklar")
    if skip:
        warn("paket kurulumu atlandı (--skip-deps)")
        return
    spec = ".[dev]" if dev else "."
    run([venv_python(), "-m", "pip", "install", "-e", spec])
    ok(f"pip install -e {spec} tamamlandı")


# --- 3.5 Model ağırlıkları ------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> bool:
    """urllib ile indir, basit ilerleme çubuğu. Başarı/başarısızlık döndürür."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "aura-bootstrap"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length", 0))
            tmp = dest.with_suffix(dest.suffix + ".part")
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        bar = "█" * (pct // 4) + "·" * (25 - pct // 4)
                        print(f"\r    {dest.name} [{bar}] {pct:3d}%", end="", flush=True)
            print()
            tmp.replace(dest)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print()
        warn(f"{dest.name} indirilemedi: {e}")
        return False


def _load_lock() -> dict:
    lock = WEIGHTS_DIR / "weights.lock.json"
    if lock.exists():
        return json.loads(lock.read_text())
    return {}


def _save_lock(data: dict) -> None:
    (WEIGHTS_DIR / "weights.lock.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def fetch_weights(skip: bool) -> dict[str, str]:
    step("Model ağırlıkları (weights/)")
    WEIGHTS_DIR.mkdir(exist_ok=True)
    lock = _load_lock()
    status: dict[str, str] = {}
    if skip:
        warn("ağırlık indirme atlandı (--skip-weights)")
        for name in WEIGHTS:
            status[name] = "present" if (WEIGHTS_DIR / name).exists() else "missing"
        return status

    for name, meta in WEIGHTS.items():
        dest = WEIGHTS_DIR / name
        expected = meta["sha256"] or lock.get(name, {}).get("sha256")

        if dest.exists():
            actual = _sha256(dest)
            if expected and actual != expected:
                warn(f"{name} hash uyuşmuyor — yeniden indiriliyor")
                dest.unlink()
            else:
                if not expected:  # trust on first use → kilitle
                    lock[name] = {"sha256": actual, "url": meta["url"]}
                ok(f"{name} mevcut ve doğrulandı ({actual[:12]}…)")
                status[name] = "present"
                continue

        if _download(meta["url"], dest):
            actual = _sha256(dest)
            if expected and actual != expected:
                fail(f"{name} indirildi ama hash uyuşmadı (beklenen {expected[:12]}…)")
                dest.unlink(missing_ok=True)
                status[name] = "corrupt"
                continue
            lock[name] = {"sha256": actual, "url": meta["url"]}
            ok(f"{name} indirildi ve kilitlendi ({actual[:12]}…)")
            status[name] = "present"
        else:
            warn(
                f"{name} indirilemedi — pipeline ağırlık olmadan 'mock' modda çalışır. "
                "Manuel yerleştirme için bkz. weights/README.md"
            )
            status[name] = "missing"

    _save_lock(lock)
    _write_weights_readme(status, lock)
    return status


def _write_weights_readme(status: dict[str, str], lock: dict) -> None:
    backend, _ = detect_backend()
    lines = [
        "# Model Ağırlıkları",
        "",
        "Bu dizin `bootstrap.py` tarafından doldurulur ve `.gitignore`'ludur.",
        "",
        f"- **Tespit edilen torch backend:** `{backend}`",
        f"- **Son kurulum platformu:** {platform.system()} {platform.machine()}",
        "",
        "## Ağırlıklar",
        "",
        "| Dosya | Durum | SHA256 (ilk 16) | Kaynak |",
        "|---|---|---|---|",
    ]
    for name, meta in WEIGHTS.items():
        h = lock.get(name, {}).get("sha256", "—")
        lines.append(
            f"| `{name}` | {status.get(name, '?')} | `{h[:16] if h != '—' else '—'}` | {meta['url']} |"
        )
    lines += [
        "",
        "## Trust-on-first-use",
        "",
        "Şartname için resmi SHA256 yayımlandığında `bootstrap.py` içindeki `WEIGHTS` "
        "sözlüğüne yazın; bozuk indirmeler otomatik yeniden indirilir. İlk indirmede "
        "hesaplanan hash `weights.lock.json`'a yazılır ve sonraki çalıştırmalarda doğrulanır.",
        "",
        "## Custom ağırlık swap",
        "",
        "Fine-tune sonrası `weights/custom_detector.pt` üretip `config/default.yaml` →",
        "`models.detector.path` değerini güncelleyin. Inference yeniden başladığında yeni",
        "ağırlık yüklenir. Detay: `docs/egitim.md`.",
    ]
    (WEIGHTS_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- 3.6 Config + env ------------------------------------------------------ #
def ensure_config_env() -> None:
    step("Config ve ortam değişkenleri")
    default = ROOT / "config" / "default.yaml"
    template = ROOT / "config" / "default.yaml.template"
    if not default.exists() and template.exists():
        shutil.copy(template, default)
        ok("config/default.yaml template'ten oluşturuldu")
    else:
        ok("config/default.yaml mevcut")
    env, env_example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists() and env_example.exists():
        shutil.copy(env_example, env)
        ok(".env oluşturuldu (.env.example'dan)")
    else:
        ok(".env mevcut")


# --- 3.7 Örnek veri -------------------------------------------------------- #
def ensure_sample_data(skip_deps: bool) -> None:
    step("Örnek veri (data/samples/)")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    video = SAMPLES_DIR / "ornek.mp4"
    if video.exists():
        ok("örnek video mevcut (atlandı)")
        return
    if skip_deps:
        warn("bağımlılıklar atlandığından örnek video üretilemedi")
        return
    try:
        run([venv_python(), "-m", "aura.synthetic", "--out", str(SAMPLES_DIR)])
        ok("sentetik örnek video + ground-truth üretildi")
    except subprocess.CalledProcessError:
        warn("örnek video üretilemedi (aura.synthetic henüz mevcut değil?)")


# --- 3.8 Node.js (opsiyonel) ---------------------------------------------- #
def ensure_node(skip: bool) -> None:
    step("Node.js / mobil (opsiyonel)")
    if skip:
        warn("node kurulumu atlandı (--skip-node)")
        return
    if shutil.which("node") is None:
        warn(
            "node bulunamadı — Python servisleri yine de tam çalışır. "
            "Mobil için: https://nodejs.org"
        )
        return
    mobile = ROOT / "mobile"
    if (mobile / "package.json").exists():
        try:
            run(["npm", "install"], cwd=mobile)
            ok("mobil bağımlılıkları kuruldu")
        except (subprocess.CalledProcessError, FileNotFoundError):
            warn("npm install başarısız — mobil opsiyonel, devam ediliyor")
    else:
        ok("mobile/package.json yok (iskelet sonra eklenecek)")


# --- 3.9 Smoke test -------------------------------------------------------- #
def smoke_test(skip_deps: bool) -> None:
    step("Smoke test")
    if skip_deps:
        warn("bağımlılıklar atlandı — smoke test çalıştırılamadı")
        return
    try:
        run([venv_python(), "-m", "aura.smoke", "--frames", "10"])
        ok("smoke test geçti")
    except subprocess.CalledProcessError:
        die("Smoke test başarısız.", "Ayrıntı için yukarıdaki çıktıya bakın.")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python bootstrap.py",
        description="AURA kurulum bootstrap'i — tek komutla sıfırdan kurulum.",
        epilog=(
            "örnekler:\n"
            "  python bootstrap.py                 # tam kurulum\n"
            "  python bootstrap.py --dev           # dev bağımlılıklarıyla\n"
            "  python bootstrap.py --skip-weights  # ağırlık indirmeden\n"
            "  python bootstrap.py --force         # .venv'i sıfırdan kur\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--skip-weights", action="store_true", help="Model ağırlığı indirmeyi atla")
    p.add_argument("--skip-node", action="store_true", help="Node.js/mobil kurulumunu atla")
    p.add_argument(
        "--skip-deps",
        action="store_true",
        help="pip kurulumlarını atla (yalnızca yapı/config/ağırlık)",
    )
    p.add_argument("--force", action="store_true", help="Mevcut .venv'i sil ve sıfırdan kur")
    p.add_argument(
        "--dev", action="store_true", help="Dev bağımlılıklarını da kur (pytest, ruff, black)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(_c("1;35", "\n╔══════════════════════════════════════════╗"))
    print(_c("1;35", "║   AURA — Bootstrap / Tek Komutla Kurulum ║"))
    print(_c("1;35", "╚══════════════════════════════════════════╝\n"))

    check_system()
    ensure_venv(args.force)
    install_torch(args.skip_deps)
    install_package(args.dev, args.skip_deps)
    weights_status = fetch_weights(args.skip_weights)
    ensure_config_env()
    ensure_sample_data(args.skip_deps)
    ensure_node(args.skip_node)
    smoke_test(args.skip_deps)

    print()
    if any(v in ("missing", "corrupt") for v in weights_status.values()):
        warn("Bazı ağırlıklar eksik — pipeline 'mock' modda çalışacak. weights/README.md'ye bakın.")
    ok("Kurulum tamamlandı.")
    print(_c("1;36", "\n  Sonraki adım:  ./run.sh   (macOS/Linux)   |   .\\run.ps1   (Windows)\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
