# Kurulum

## Ne yapar
AURA'yı sıfırdan, tek komutla, manuel adım olmadan kurar. İdempotenttir.

## Gereksinimler
- Python ≥ 3.10
- git
- (opsiyonel) Node.js — mobil için
- Donanım backend'i otomatik: Apple Silicon→MPS, NVIDIA→CUDA, diğer→CPU

## Kurulum

### macOS / Linux
```bash
./setup.sh           # = python3 bootstrap.py
./setup.sh --dev     # dev bağımlılıklarıyla (pytest, ruff, black)
```

### Windows (PowerShell 7+)
```powershell
.\setup.ps1
```

`bootstrap.py` adımları: sistem doğrulama → `.venv` → torch backend (otomatik) →
`pip install -e .` → model ağırlıkları (SHA256, trust-on-first-use) → config/.env →
örnek video → (opsiyonel) node → smoke test.

### Seçenekler
```bash
python bootstrap.py --skip-weights   # ağırlık indirmeden
python bootstrap.py --skip-node      # node atla
python bootstrap.py --skip-deps      # pip atla (yapı/config/ağırlık)
python bootstrap.py --force          # .venv'i sıfırdan
```

## Kurulumu doğrula
```bash
python tools/doctor.py    # bağımlılık, cihaz (MPS/CUDA/CPU), ağırlık, config, profil ✓
```
Tüm çekirdek satırlar ✓ ise sistem gerçek modda hazır. Ağırlık eksikse `python bootstrap.py`.

## Kullanım
```bash
./run.sh             # inference :8080, qod :8081, nv :8082
open http://localhost:8080/
```

## Örnekler
```bash
# CPU'ya zorla
AURA_DEVICE=cpu ./run.sh
# Farklı port
AURA_INFERENCE_PORT=9090 ./run.sh
```

## Sorun Giderme
| Belirti | Çözüm |
|---|---|
| `Python 3.10+ gerekli` | `python3 --version`; pyenv/Homebrew ile güncelleyin |
| Ağırlık indirilemiyor (404/ağ) | Kurulum durmaz; pipeline `mock` modda çalışır. Manuel: `.pt`'yi `weights/`'e koyun |
| `ultralytics yok` → mock mod | Normal; gerçek model için ağırlık + `ai_mode: real` |
| Port dolu | `AURA_INFERENCE_PORT`/`AURA_QOD_MOCK_PORT`/`AURA_NV_MOCK_PORT` ile değiştirin |
| `/cameras` izin istiyor (macOS) | Kamera izni verin veya `AURA_CAMERA_PROBE=0` |
| node yok | Sarı uyarı; Python servisleri tam çalışır, mobil opsiyonel |
