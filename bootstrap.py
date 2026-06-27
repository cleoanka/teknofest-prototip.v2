#!/usr/bin/env python3
"""RoadGuard kurulum bootstrap'i — saf Python stdlib.

Tek komutla sıfırdan çalışır hale getirir: sanal ortam, donanıma uygun torch
backend'i, paket bağımlılıkları, model ağırlıkları (SHA256 doğrulamalı), örnek
veri ve smoke test. İdempotenttir; ikinci çalıştırma tamamlanmış adımları atlar.

Bu dosya hiçbir üçüncü-parti modüle bağımlı DEĞİLDİR (sistem python'u ile çalışır).
OpenCV/numpy gerektiren işler (örnek video üretimi, smoke) venv python'una
subprocess ile devredilir.
"""

# Python 3.10 öncesi sürümlerde bile "str | None" gibi tip ipuçlarının çalışması
# için anotasyonları string olarak ertele (PEP 563).
from __future__ import annotations

import argparse  # Komut satırı argümanlarını (--force, --dev, vb.) ayrıştırmak için
import hashlib  # İndirilen ağırlıkların SHA256 bütünlük doğrulaması için
import json  # weights.lock.json okuma/yazma için
import os  # Ortam değişkenleri (NO_COLOR) ve işletim sistemi erişimi için
import platform  # İşletim sistemi / mimari tespiti (Windows, Darwin, arm64...) için
import shutil  # Dosya kopyalama, dizin silme, PATH'te program arama (which) için
import subprocess  # venv python'unu ve harici komutları (pip, git, nvidia-smi) çağırmak için
import sys  # Python sürümü, stdout/stderr, çıkış kodu erişimi için
import time  # İndirme yeniden-denemeleri arasındaki bekleme için
import urllib.error  # İndirme sırasında ağ/HTTP hatalarını yakalamak için
import urllib.request  # Ağırlık dosyalarını HTTP üzerinden indirmek için
from pathlib import Path  # Platformdan bağımsız dosya yolu işlemleri için
from typing import NoReturn  # die() gibi asla geri dönmeyen fonksiyonları işaretlemek için

# Windows konsolu cp1254/cp437 gibi olabilir; UTF-8'e geç ki kutu çizimi ve
# sembol çıktısı (╔ ▶ ✓ ⚠ █) UnicodeEncodeError ile çökmesin.
for _stream in (sys.stdout, sys.stderr):
    try:
        # reconfigure stdout/stderr'ı UTF-8'e zorlar; kodlanamayan karakteri kırpma yerine değiştirir.
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        # Eski Python veya yönlendirilmiş çıktıda reconfigure olmayabilir — sessizce geç.
        pass

# --- Proje genelinde kullanılan sabit dosya yolları ----------------------- #
ROOT = Path(__file__).resolve().parent  # Bu dosyanın bulunduğu proje kök dizini
VENV = ROOT / ".venv"  # Oluşturulacak/kullanılacak sanal ortam dizini
WEIGHTS_DIR = ROOT / "weights"  # Model ağırlıklarının indirileceği dizin
SAMPLES_DIR = ROOT / "data" / "samples"  # Örnek video ve ground-truth dizini
MIN_PY = (3, 10)  # Desteklenen en düşük Python sürümü

# Model ağırlıkları — bootstrap tarafından indirilir, .gitignore'lu.
# sha256 None => "trust on first use": ilk indirmede hash hesaplanır ve
# weights/weights.lock.json'a yazılır; sonraki çalıştırmalar buna karşı doğrular.
WEIGHTS: dict[str, dict] = {
    "yolo26s.pt": {  # Küçük (small) YOLO modeli — hızlı, düşük doğruluk
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
        "sha256": None,  # Resmi hash yok → ilk indirmede kilitle (trust on first use)
    },
    "yolo26l.pt": {  # Büyük (large) YOLO modeli — yavaş, yüksek doğruluk
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
        "sha256": None,  # Resmi hash yok → ilk indirmede kilitle
    },
    "yolo26s-pose.pt": {  # Pose modeli (small) — l-pose inmezse loglu fallback
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt",
        "sha256": None,  # Resmi hash yok → ilk indirmede kilitle
    },
    "yolo26l-pose.pt": {  # Pose modeli (large) — sürücü davranışı keypoint geometrisi
        # Sürücü-içi sıkı kırpma ile modele giden alan minimum olduğundan büyük
        # model affordable (kullanıcı kararı: minimum alana maksimum model).
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-pose.pt",
        "sha256": None,  # Resmi hash yok → ilk indirmede kilitle
    },
    "lp_yolo11n.pt": {  # Plaka dedektörü (YOLOv11n fine-tune) — sıkı plaka kırpma
        "url": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt",
        "sha256": None,  # Resmi hash yok → ilk indirmede kilitle
    },
}

# Fine-tune dedektör (11 sınıf; held-out mAP50 .788). İndirme URL'si yok — Git LFS ile
# teknofest-prototip (v1) reposunda yaşar. Bootstrap önce weights/ altına bakar,
# yoksa bilinen komşu klon yollarından kopyalamayı dener (yerel geliştirme kolaylığı).
FINETUNE_WEIGHT = "yolguvenligi_types_v4.pt"
FINETUNE_SIBLINGS = [
    Path.home() / "teknofest-prototip" / "models" / FINETUNE_WEIGHT,
    ROOT.parent / "teknofest-prototip" / "models" / FINETUNE_WEIGHT,
]

# --------------------------------------------------------------------------- #
# Çıktı yardımcıları (ANSI; NO_COLOR veya tty değilse düz)
# --------------------------------------------------------------------------- #
# Renkli çıktı yalnızca gerçek bir terminale yazıyorsak ve NO_COLOR ayarlı değilse açık.
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, msg: str) -> str:
    # Mesajı ANSI renk koduyla sarmalar; renk kapalıysa düz metni döndürür.
    return f"\033[{code}m{msg}\033[0m" if _COLOR else msg


def step(msg: str) -> None:
    # Ana kurulum adımı başlığı (mavi ▶).
    print(_c("1;36", f"▶ {msg}"))


def ok(msg: str) -> None:
    # Başarılı alt-adım (yeşil ✓).
    print(_c("1;32", f"  ✓ {msg}"))


def warn(msg: str) -> None:
    # Kritik olmayan uyarı (sarı ⚠) — kurulum devam eder.
    print(_c("1;33", f"  ⚠ {msg}"))


def fail(msg: str) -> None:
    # Hata mesajı (kırmızı ✗) — tek başına program akışını durdurmaz.
    print(_c("1;31", f"  ✗ {msg}"))


def die(msg: str, hint: str = "") -> NoReturn:
    # Ölümcül hata: mesajı bas, varsa ipucu göster ve programı 1 koduyla sonlandır.
    fail(msg)
    if hint:
        print(f"    → {hint}")
    sys.exit(1)


# --------------------------------------------------------------------------- #
def venv_python() -> Path:
    # Sanal ortamdaki python yorumlayıcısının yolunu döndürür (OS'a göre değişir).
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"  # Windows: .venv\Scripts\python.exe
    return VENV / "bin" / "python"  # macOS/Linux: .venv/bin/python


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Komutu çalıştır; çıktıyı akıt. check=True ile hata fırlatır."""
    kw.setdefault("cwd", ROOT)  # Çalışma dizini verilmediyse proje kökünü kullan
    # Tüm argümanları str'e çevir (Path nesneleri olabilir) ve hata durumunda istisna fırlat.
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


# --- 3.1 Sistem doğrulama ------------------------------------------------- #
def check_system() -> None:
    step("Sistem doğrulama")
    # Python sürümü minimumun altındaysa kurulumu hemen durdur.
    if sys.version_info < MIN_PY:
        die(
            f"Python {MIN_PY[0]}.{MIN_PY[1]}+ gerekli (mevcut {sys.version.split()[0]}).",
            "Kurulum rehberi: docs/kurulum.md",
        )
    ok(f"Python {sys.version.split()[0]}")
    # git zorunlu değil ama yoksa bazı işlevler sınırlanabilir — sadece uyar.
    if shutil.which("git") is None:
        warn("git bulunamadı — klonlama/versiyonlama sınırlı olabilir.")
    else:
        ok("git mevcut")
    ok(f"Platform: {platform.system()} {platform.machine()}")


# --- 3.2 Sanal ortam ------------------------------------------------------- #
def ensure_venv(force: bool) -> None:
    step("Sanal ortam (.venv)")
    # --force verildiyse ve mevcut bir .venv varsa onu tamamen sil (sıfırdan kurulum).
    if force and VENV.exists():
        warn(".venv siliniyor (--force)")
        shutil.rmtree(VENV)
    # .venv zaten varsa (ve silinmediyse) bu adımı atla — idempotentlik.
    if VENV.exists():
        ok(".venv mevcut (atlandı)")
        return
    # Sistem python'u ile yeni bir sanal ortam oluştur.
    run([sys.executable, "-m", "venv", str(VENV)])
    # Yeni ortamda pip ve wheel'i güncelle (sessizce) — sonraki kurulumlar için temel.
    run([venv_python(), "-m", "pip", "install", "--quiet", "--upgrade", "pip", "wheel"])
    ok(".venv oluşturuldu")


# --- 3.3 Torch backend tespiti + kurulum ----------------------------------- #
def _cuda_index() -> str:
    """GPU compute capability'sine göre uygun PyTorch CUDA tekerlek index'ini seç.

    Blackwell (sm_100/sm_120, ör. RTX 50xx) yalnızca cu128+ tekerleklerinde derlenmiş
    kernel'lere sahiptir; eski mimariler (<= sm_90) için cu121 yeterlidir. nvidia-smi
    compute_cap okunamazsa güvenli varsayılan cu121'dir.
    """
    try:
        # nvidia-smi'den GPU'nun compute capability değerini (ör. "12.0") sorgula.
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        # Çıktıdaki tüm GPU yeteneklerini float listesine çevir.
        caps = [float(line.strip()) for line in out.splitlines() if line.strip()]
        if caps and max(caps) >= 10.0:  # sm_100+ (Blackwell) → cu128
            return "https://download.pytorch.org/whl/cu128"
    except (OSError, ValueError, subprocess.SubprocessError):
        # nvidia-smi yok / çıktı bozuk / zaman aşımı → güvenli varsayılana düş.
        pass
    return "https://download.pytorch.org/whl/cu121"


def detect_backend() -> tuple[str, list[str]]:
    """(backend_adı, pip_index_args) döndür.

    CUDA index'i cu128: Turing(sm_75)'ten Blackwell(sm_120)'e kadar tüm modern
    NVIDIA GPU'larını kapsar. Eski cu121, yeni GPU'larda "no kernel image" verir.
    """
    system, machine = platform.system(), platform.machine()
    # Apple Silicon (M serisi) → Metal (MPS) hızlandırması, özel index gerekmez.
    if system == "Darwin" and machine == "arm64":
        return "mps", []  # Apple Silicon — varsayılan index, MPS otomatik
    # nvidia-smi varsa NVIDIA GPU mevcut demektir → CUDA tekerleğini kullan.
    if shutil.which("nvidia-smi"):
        return "cuda", ["--index-url", "https://download.pytorch.org/whl/cu128"]
    # Hiçbir GPU yoksa CPU-only tekerlekle kur.
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
    "if not torch.cuda.is_available():\n"  # GPU hiç görünmüyorsa
    "    sys.exit(3)\n"  # → kod 3 (muhtemelen CPU derlemesi)
    "try:\n"
    "    x = torch.zeros((8, 8), device='cuda'); _ = x @ x; torch.cuda.synchronize()\n"  # GPU'da gerçek matris çarpımı dene
    "except Exception:\n"
    "    sys.exit(4)\n"  # Çalışmadıysa → kod 4 (yanlış/eski kernel derlemesi)
    "sys.exit(0)\n"  # Sorunsuz → kod 0 (GPU çalışıyor)
)


def install_torch(skip: bool) -> str:
    step("Torch backend")
    backend, index_args = detect_backend()  # Donanıma uygun backend ve pip index'ini tespit et
    # --skip-deps ile kurulum atlanırsa sadece tespit edilen backend'i bildir ve dön.
    if skip:
        warn(f"torch kurulumu atlandı (--skip-deps); tespit edilen backend: {backend}")
        return backend

    # torch + torchvision hâlihazırda import edilebiliyor mu? (returncode 0 ise evet)
    imported = (
        subprocess.run(
            [str(venv_python()), "-c", "import torch, torchvision"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )
    if imported:
        # CUDA dışı backend'lerde (cpu/mps) import başarısı yeterli kabul edilir.
        if backend != "cuda":
            ok(f"torch zaten kurulu (backend: {backend})")
            return backend
        # CUDA: sadece import değil, bu GPU'da fiilen çalıştığını da doğrula.
        probe = subprocess.run(
            [str(venv_python()), "-c", _TORCH_GPU_PROBE], cwd=ROOT, capture_output=True
        )
        if probe.returncode == 0:
            # GPU'da kernel çalıştı → mevcut kurulum sağlam, yeniden kurmaya gerek yok.
            ok("torch zaten kurulu ve GPU çalışıyor (backend: cuda)")
            return backend
        if probe.returncode == 3:
            # GPU görünmüyor → büyük olasılıkla yanlışlıkla CPU derlemesi kurulu.
            warn(
                "torch kurulu ama GPU görünmüyor (CPU derlemesi?) — CUDA (cu128) derlemesi kuruluyor"
            )
        else:
            # Kod 4: GPU var ama bu mimariye uygun kernel yok → doğru derlemeyi kur.
            warn(
                "torch kurulu ama bu GPU'da kernel yok (eski/yanlış derleme) — "
                "doğru CUDA (cu128) derlemesi kuruluyor"
            )

    ok(f"Tespit edilen backend: {backend}")
    # CUDA için sabitlenmiş sürümleri, diğer backend'lerde en güncel sürümü kur.
    pkgs = TORCH_PIN_CUDA if backend == "cuda" else ["torch", "torchvision"]
    run([venv_python(), "-m", "pip", "install", "--upgrade", *pkgs, *index_args])
    ok("torch + torchvision kuruldu")
    return backend


# --- 3.4 Paket kurulumu ---------------------------------------------------- #
def install_package(dev: bool, skip: bool) -> None:
    step("RoadGuard paketi + bağımlılıklar")
    # --skip-deps verildiyse pip kurulumunu tümüyle atla.
    if skip:
        warn("paket kurulumu atlandı (--skip-deps)")
        return
    # --dev ile test/lint araçlarını da içeren "extra"yı seç, aksi halde temel paket.
    spec = ".[dev]" if dev else "."
    # RoadGuard paketini düzenlenebilir (editable, -e) modda kur — kaynak değişiklikleri anında yansır.
    run([venv_python(), "-m", "pip", "install", "-e", spec])
    ok(f"pip install -e {spec} tamamlandı")
    # Plakaya-özel OCR motoru (varsayılan plate.ocr_engine=fastplate) için fast-plate-ocr +
    # onnxruntime — best-effort; kurulamazsa pipeline EasyOCR'a graceful düşer, çökmez.
    try:
        run([venv_python(), "-m", "pip", "install", "--quiet", "fast-plate-ocr", "onnxruntime"])
        ok("fast-plate-ocr + onnxruntime kuruldu (plaka OCR motoru, fastplate)")
    except Exception:
        warn("fast-plate-ocr kurulamadı — plaka EasyOCR'a düşer (opsiyonel)")


# --- 3.5 Model ağırlıkları ------------------------------------------------- #
def _sha256(path: Path) -> str:
    # Dosyanın SHA256 özetini bellek dostu şekilde (1 MB'lık parçalarla) hesapla.
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # iter(..., b"") dosya sonuna kadar 1 MB'lık bloklar okur.
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    """urllib ile indir (yeniden-denemeli), basit ilerleme çubuğu.

    Ağ kesintileri (connection reset / timeout) tek denemede kurulumu mock moda
    düşürüyordu (D2): artık `retries` deneme yapılır, denemeler arası bekleme
    katlanarak artar (2s, 4s, 8s) ve yarım kalan `.part` dosyası temizlenir.
    """
    for attempt in range(1, retries + 1):
        if _download_once(url, dest):
            return True
        if attempt < retries:
            wait = 2**attempt
            warn(f"{dest.name} deneme {attempt}/{retries} başarısız — {wait}s sonra yeniden")
            time.sleep(wait)
    dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
    return False


def _download_once(url: str, dest: Path) -> bool:
    try:
        # Bazı sunucular User-Agent ister; özel bir başlıkla istek oluştur.
        req = urllib.request.Request(url, headers={"User-Agent": "roadguard-bootstrap"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            # İlerleme yüzdesi için toplam boyutu başlıktan al (yoksa 0).
            total = int(resp.headers.get("Content-Length", 0))
            # Önce .part geçici dosyasına yaz; tamamlanınca atomik olarak yerine koy.
            tmp = dest.with_suffix(dest.suffix + ".part")
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB'lık parçalar halinde indir
                    if not chunk:
                        break  # Veri bitti → döngüden çık
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        # Basit metin tabanlı ilerleme çubuğu çiz (aynı satırda günceller).
                        pct = done * 100 // total
                        bar = "█" * (pct // 4) + "·" * (25 - pct // 4)
                        print(f"\r    {dest.name} [{bar}] {pct:3d}%", end="", flush=True)
            print()  # İlerleme çubuğundan sonra satır sonu
            tmp.replace(dest)  # Geçici dosyayı nihai konuma taşı (atomik)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        # Ağ/dosya hatası → uyar ve başarısız döndür (kurulum mock modda devam edebilir).
        print()
        warn(f"{dest.name} indirilemedi: {e}")
        return False


def _load_lock() -> dict:
    # weights.lock.json varsa içeriğini (ad→{sha256,url}) yükle, yoksa boş sözlük.
    lock = WEIGHTS_DIR / "weights.lock.json"
    if lock.exists():
        return json.loads(lock.read_text())
    return {}


def _save_lock(data: dict) -> None:
    # Kilit verisini okunabilir (girintili) JSON olarak diske yaz.
    (WEIGHTS_DIR / "weights.lock.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def fetch_weights(skip: bool) -> dict[str, str]:
    step("Model ağırlıkları (weights/)")
    WEIGHTS_DIR.mkdir(exist_ok=True)  # weights/ dizini yoksa oluştur
    lock = _load_lock()  # Daha önce kilitlenmiş hash'leri yükle
    status: dict[str, str] = {}  # Her ağırlık için sonuç durumu (present/missing/corrupt)
    # --skip-weights: indirme yapma, yalnızca dosyaların var olup olmadığını raporla.
    if skip:
        warn("ağırlık indirme atlandı (--skip-weights)")
        for name in WEIGHTS:
            status[name] = "present" if (WEIGHTS_DIR / name).exists() else "missing"
        return status

    for name, meta in WEIGHTS.items():
        dest = WEIGHTS_DIR / name
        # Beklenen hash: önce WEIGHTS'teki sabit değer, yoksa kilit dosyasındaki değer.
        expected = meta["sha256"] or lock.get(name, {}).get("sha256")

        if dest.exists():
            # Dosya zaten varsa hash'ini hesapla ve doğrula.
            actual = _sha256(dest)
            if expected and actual != expected:
                # Hash beklenenle uyuşmuyor → bozuk kabul et, sil ve aşağıda yeniden indir.
                warn(f"{name} hash uyuşmuyor — yeniden indiriliyor")
                dest.unlink()
            else:
                if not expected:  # trust on first use → kilitle
                    # İlk kez görülüyor: hesaplanan hash'i kilit dosyasına yaz.
                    lock[name] = {"sha256": actual, "url": meta["url"]}
                ok(f"{name} mevcut ve doğrulandı ({actual[:12]}…)")
                status[name] = "present"
                continue  # Bu ağırlık tamam, sonrakine geç

        # Dosya yok veya bozuk olduğu için silindi → indir.
        if _download(meta["url"], dest):
            actual = _sha256(dest)
            if expected and actual != expected:
                # İndirildi ama hash hâlâ tutmuyor → bozuk, sil ve corrupt olarak işaretle.
                fail(f"{name} indirildi ama hash uyuşmadı (beklenen {expected[:12]}…)")
                dest.unlink(missing_ok=True)
                status[name] = "corrupt"
                continue
            # Başarılı indirme → hash'i kilitle.
            lock[name] = {"sha256": actual, "url": meta["url"]}
            ok(f"{name} indirildi ve kilitlendi ({actual[:12]}…)")
            status[name] = "present"
        else:
            # İndirme başarısız → eksik; pipeline ağırlıksız mock modda çalışabilir.
            warn(
                f"{name} indirilemedi — pipeline ağırlık olmadan 'mock' modda çalışır. "
                "Manuel yerleştirme için bkz. weights/README.md"
            )
            status[name] = "missing"

    # Fine-tune dedektör: indirme URL'si yok (Git LFS, teknofest-prototip v1 reposu).
    # Önce weights/ altına bak; yoksa bilinen komşu klon yollarından kopyalamayı dene.
    ft = WEIGHTS_DIR / FINETUNE_WEIGHT
    if ft.exists() and ft.stat().st_size > 1024:
        ok(f"{FINETUNE_WEIGHT} mevcut")
        status[FINETUNE_WEIGHT] = "present"
    else:
        for cand in FINETUNE_SIBLINGS:
            # 1 KB üstü kontrolü: gerçek ağırlık mı, çekilmemiş LFS pointer'ı mı ayırır.
            if cand.exists() and cand.stat().st_size > 1024:
                shutil.copy2(cand, ft)
                ok(f"{FINETUNE_WEIGHT} komşu repodan kopyalandı: {cand}")
                status[FINETUNE_WEIGHT] = "present"
                break
        else:
            # İpucu platforma göre: '~' Windows cmd/git'te genişlemez (%USERPROFILE% gerekir).
            v1_path = (
                "%USERPROFILE%\\teknofest-prototip" if os.name == "nt" else "~/teknofest-prototip"
            )
            warn(
                f"{FINETUNE_WEIGHT} bulunamadı — detector stok yolo26s.pt'ye düşer (loglu). "
                f'Edinmek için: git lfs install && git -C "{v1_path}" lfs pull '
                "(git-lfs kurulu olmalı: https://git-lfs.com)"
            )
            status[FINETUNE_WEIGHT] = "missing"

    _save_lock(lock)  # Güncellenen kilit verisini diske yaz
    _write_weights_readme(status, lock)  # weights/README.md'yi durum tablosuyla güncelle
    return status


def _write_weights_readme(status: dict[str, str], lock: dict) -> None:
    # weights/ dizinine, ağırlık durumunu özetleyen bir README üret.
    backend, _ = detect_backend()  # README'de göstermek için tespit edilen backend
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
    # Her ağırlık için kilitteki hash ve güncel durumu içeren tablo satırı oluştur.
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
    # Tüm satırları birleştirip README.md'ye yaz.
    (WEIGHTS_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- 3.6 Config + env ------------------------------------------------------ #
def ensure_config_env() -> None:
    step("Config ve ortam değişkenleri")
    default = ROOT / "config" / "default.yaml"  # Asıl kullanılacak config
    template = ROOT / "config" / "default.yaml.template"  # Şablon (depoya dahil)
    # Asıl config yoksa ama şablon varsa → şablondan kopyala.
    if not default.exists() and template.exists():
        shutil.copy(template, default)
        ok("config/default.yaml template'ten oluşturuldu")
    else:
        ok("config/default.yaml mevcut")
    env, env_example = ROOT / ".env", ROOT / ".env.example"  # Gizli/ortam değişkenleri ve örneği
    # .env yoksa ama .env.example varsa → örnekten kopyala.
    if not env.exists() and env_example.exists():
        shutil.copy(env_example, env)
        ok(".env oluşturuldu (.env.example'dan)")
    else:
        ok(".env mevcut")


# --- 3.7 Örnek veri -------------------------------------------------------- #
def ensure_sample_data(skip_deps: bool) -> None:
    step("Örnek veri (data/samples/)")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)  # data/samples/ dizinini (gerekirse) oluştur
    video = SAMPLES_DIR / "ornek.mp4"
    # Örnek video zaten varsa yeniden üretme — idempotentlik.
    if video.exists():
        ok("örnek video mevcut (atlandı)")
        return
    # Bağımlılıklar kurulmadıysa (OpenCV/numpy yok) video üretilemez.
    if skip_deps:
        warn("bağımlılıklar atlandığından örnek video üretilemedi")
        return
    try:
        # venv python'una roadguard.synthetic modülünü çağırarak sentetik video ürettir.
        run([venv_python(), "-m", "roadguard.synthetic", "--out", str(SAMPLES_DIR)])
        ok("sentetik örnek video + ground-truth üretildi")
    except subprocess.CalledProcessError:
        # Modül henüz yoksa/başarısızsa kritik değil — uyar ve devam et.
        warn("örnek video üretilemedi (roadguard.synthetic henüz mevcut değil?)")


# --- 3.8 Node.js (opsiyonel) ---------------------------------------------- #
def ensure_node(skip: bool) -> None:
    step("Node.js / mobil (opsiyonel)")
    # --skip-node ile bu opsiyonel adım atlanır.
    if skip:
        warn("node kurulumu atlandı (--skip-node)")
        return
    # node yoksa Python tarafı yine de tam çalışır; sadece mobil kurulamaz.
    if shutil.which("node") is None:
        warn(
            "node bulunamadı — Python servisleri yine de tam çalışır. "
            "Mobil için: https://nodejs.org"
        )
        return
    mobile = ROOT / "mobile"
    # mobile/package.json varsa npm bağımlılıklarını kur.
    if (mobile / "package.json").exists():
        try:
            run(["npm", "install"], cwd=mobile)
            ok("mobil bağımlılıkları kuruldu")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # npm yoksa/başarısızsa mobil opsiyonel olduğundan devam et.
            warn("npm install başarısız — mobil opsiyonel, devam ediliyor")
    else:
        # Mobil iskelet henüz eklenmemiş — sorun değil.
        ok("mobile/package.json yok (iskelet sonra eklenecek)")


# --- 3.9 Smoke test -------------------------------------------------------- #
def smoke_test(skip_deps: bool) -> None:
    step("Smoke test")
    # Bağımlılıklar yoksa pipeline çalıştırılamaz → smoke testi atla.
    if skip_deps:
        warn("bağımlılıklar atlandı — smoke test çalıştırılamadı")
        return
    try:
        # venv python'unda 10 kare üzerinde hızlı bir uçtan uca çalışırlık testi yap.
        run([venv_python(), "-m", "roadguard.smoke", "--frames", "10"])
        ok("smoke test geçti")
    except subprocess.CalledProcessError:
        # Smoke test başarısızsa kurulum sağlıklı değildir → ölümcül hata.
        die("Smoke test başarısız.", "Ayrıntı için yukarıdaki çıktıya bakın.")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    # Komut satırı argümanlarını ve --help çıktısını tanımlayan ayrıştırıcıyı kur.
    p = argparse.ArgumentParser(
        prog="python bootstrap.py",
        description="RoadGuard kurulum bootstrap'i — tek komutla sıfırdan kurulum.",
        epilog=(  # --help sonunda gösterilecek örnek kullanımlar
            "örnekler:\n"
            "  python bootstrap.py                 # tam kurulum\n"
            "  python bootstrap.py --dev           # dev bağımlılıklarıyla\n"
            "  python bootstrap.py --skip-weights  # ağırlık indirmeden\n"
            "  python bootstrap.py --force         # .venv'i sıfırdan kur\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,  # epilog'daki satır sonlarını koru
    )
    # Her bayrak bir kurulum adımını atlamaya/değiştirmeye yarar (store_true = varlık bayrağı).
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
    args = build_parser().parse_args(argv)  # Argümanları ayrıştır (None → sys.argv kullanılır)
    # Renkli başlık banner'ı bas.
    print(_c("1;35", "\n╔══════════════════════════════════════════╗"))
    print(_c("1;35", "║   RoadGuard — Bootstrap / Tek Komutla Kurulum ║"))
    print(_c("1;35", "╚══════════════════════════════════════════╝\n"))

    # Kurulum adımları sırayla — her biri idempotent ve bayraklarla yönetilebilir.
    check_system()  # 3.1 Python/git/platform doğrula
    ensure_venv(args.force)  # 3.2 Sanal ortamı kur
    install_torch(args.skip_deps)  # 3.3 Donanıma uygun torch'u kur
    install_package(args.dev, args.skip_deps)  # 3.4 RoadGuard paketini kur
    weights_status = fetch_weights(args.skip_weights)  # 3.5 Ağırlıkları indir/doğrula
    ensure_config_env()  # 3.6 Config ve .env hazırla
    ensure_sample_data(args.skip_deps)  # 3.7 Örnek veriyi üret
    ensure_node(args.skip_node)  # 3.8 (Opsiyonel) Node/mobil
    smoke_test(args.skip_deps)  # 3.9 Uçtan uca çalışırlık testi

    print()
    # Herhangi bir ağırlık eksik/bozuksa kullanıcıyı mock mod hakkında uyar.
    if any(v in ("missing", "corrupt") for v in weights_status.values()):
        warn("Bazı ağırlıklar eksik — pipeline 'mock' modda çalışacak. weights/README.md'ye bakın.")
    ok("Kurulum tamamlandı.")
    # Bir sonraki adım için çalıştırma komutlarını hatırlat.
    print(_c("1;36", "\n  Sonraki adım:  ./run.sh   (macOS/Linux)   |   .\\run.ps1   (Windows)\n"))
    return 0  # Başarı çıkış kodu


# Dosya doğrudan çalıştırıldığında main()'i koştur ve dönüş kodunu süreç çıkış kodu yap.
if __name__ == "__main__":
    raise SystemExit(main())
