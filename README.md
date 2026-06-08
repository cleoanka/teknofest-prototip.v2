# AURA — 5G & Yapay Zekâ ile Akıllı Yol Güvenliği

> **TEKNOFEST 2026** "5G & Yapay Zekâ ile Akıllı Yol Güvenliği" yarışması prototipi.
> Üretim kalitesinde, cross-platform (macOS + Windows), tek komutla ayağa kalkan monorepo.

AURA; trafik kamerası görüntüsünden **araç / plaka / sürücü davranışı / hız** tespiti yapan
gerçek bir YZ çekirdeği ile, bu çekirdeği **5G QoD** (CAMARA Quality-on-Demand) ve
**Number Verification** gibi telekom yetenekleriyle birleştiren bir sistemdir.

**Gerçek / Mock sınırı:** YZ çekirdeğinin tamamı (CV/tracking/state-machine/OCR/speed/eval)
**gerçektir**. Ağ/telekom/mobil katmanları (QoD gateway, NV API, 5G şebekesi) gerçek API
sözleşmesini birebir taklit eden **mock**'lardır — final ortamında yalnızca endpoint/credential değişir.

**Öne çıkanlar:**
- **Ağırlıksız çalışır:** YOLO26 ağırlığı yoksa pipeline deterministik *mock* modda tüm
  hattı (tespit→plaka→sürücü→hız→QoD→event) uçtan uca koşturur; demo ve testler model olmadan geçer.
- **QoD kanıtı:** A/B harness ölçülebilir delta üretir (plaka doğruluğu **+33pp**, küçük
  nesne tespiti **+51pp**) — şartnamenin %40 QoD puanı için kanıt aracı.
- **Kalite:** 58 unit test, `ruff` + `black` temiz, GitHub Actions CI.

---

## Hızlı Başlangıç

### macOS / Linux
```bash
./setup.sh        # bağımlılıklar + model ağırlıkları + örnek veri + smoke (tek komut)
./run.sh          # inference :8080, QoD mock :8081, NV mock :8082
```

### Windows (PowerShell 7+)
```powershell
.\setup.ps1
.\run.ps1
```

Ardından tarayıcıda:
- **Dashboard:** http://localhost:8080/
- **OpenAPI:** http://localhost:8080/docs

`setup` idempotenttir; ikinci çalıştırma tamamlanmış adımları atlar. Donanım backend'i
otomatik seçilir (Apple Silicon→MPS, NVIDIA→CUDA, diğer→CPU).

---

## Mimari Özeti

```
[Kamera] → [Ön-İşleme] → [YOLO26s + ByteTrack] ──┬─→ [Sürücü Kabini ROI] → [YOLO26l Driver State]
                                ↑                 └─→ [Plaka ROI] → [Sweet Spot + Voting + OCR]
                          [16/8 State Machine]                              ↓
                                                                     [QoD Tetikleyici]
                                                                           ↓
                              [ID-Merkezli Accumulator] ← [Hız (kalibrasyon-bağımlı)]
                                          ↓
                              [Event / Annotation Stream] → Dashboard + Mobil
```

Mimari kararlar (değişmez): cascade pipeline (YOLO26s→YOLO26l), ID-merkezli birikim,
16/8 state machine, **MediaPipe yok** (yorgunluk dahil tüm sürücü durumları YOLO26l detection
sınıfı), kalibrasyon-bağımlı hız. Detay: [`docs/mimari.md`](docs/mimari.md).

---

## Komut Rehberi

| Komut | Açıklama |
|---|---|
| `python bootstrap.py --help` | Kurulum seçenekleri |
| `python -m aura --help` | Ana inference pipeline |
| `python -m aura.eval --help` | Değerlendirme + QoD A/B karşılaştırması |
| `python -m train --help` | Model eğitimi (detector / driver-state / dataset) |
| `python -m aura.synthetic` | Sentetik örnek video + ground-truth üret |
| `make help` | Geliştirme kısayolları |

Tüm `--help` çıktıları: [`docs/cli_referans.md`](docs/cli_referans.md).
Tüm API endpoint'leri: [`docs/api_referans.md`](docs/api_referans.md).

---

## Repo Haritası

| Dizin | İçerik |
|---|---|
| `aura/` | YZ çekirdeği (preprocessing, detection, stability, driver_state, plate, speed, accumulator, qod, events, pipeline, eval) |
| `services/` | `inference_api` (FastAPI) + `qod_mock` + `nv_mock` |
| `dashboard/` | Vanilla JS + Canvas profesyonel web arayüzü |
| `mobile/` | Expo (React Native) iskeleti |
| `train/` | YOLO26 fine-tune pipeline'ları |
| `config/` | `default.yaml` — tek config kaynağı |
| `weights/` | Model ağırlıkları (bootstrap doldurur, `.gitignore`'lu) |
| `data/samples/` | Örnek video + ground-truth |
| `docs/` | Mimari, kurulum, CLI/API referans, değerlendirme, izlenebilirlik |
| `tests/` | pytest (state machine, voting, risk, QoD, API sözleşmeleri) |

---

## Dokümantasyon

| Belge | İçerik |
|---|---|
| [`docs/mimari.md`](docs/mimari.md) | Tam sistem mimarisi v2.0 |
| [`docs/kurulum.md`](docs/kurulum.md) | Platform-bazlı kurulum + sorun giderme |
| [`docs/calistirma.md`](docs/calistirma.md) | Uçtan uca demo senaryosu |
| [`docs/cli_referans.md`](docs/cli_referans.md) | Tüm `--help` çıktıları |
| [`docs/api_referans.md`](docs/api_referans.md) | Tüm endpoint'ler (curl + response) |
| [`docs/egitim.md`](docs/egitim.md) | Eğitim akışı + hyperparameter rehberi |
| [`docs/veri_seti.md`](docs/veri_seti.md) | Veri toplama + sentetik augmentasyon |
| [`docs/kalibrasyon.md`](docs/kalibrasyon.md) | Hız kalibrasyonu (tripwire / IPM) |
| [`docs/degerlendirme.md`](docs/degerlendirme.md) | Metrikler + QoD A/B protokolü |
| [`docs/mimari_ek_moduller.md`](docs/mimari_ek_moduller.md) | §8 opsiyonel modüller (lazy) |
| [`docs/sartname_izlenebilirlik.md`](docs/sartname_izlenebilirlik.md) | Şartname ↔ modül eşlemesi |

Her dizin kendi `README.md`'sini taşır.

---

## Test & Kalite

```bash
pytest -m "not integration"        # 58 unit test (mock modda, ağırlık gerektirmez)
ruff check . && black --check .    # lint + format
```
Model gerektiren testler `@pytest.mark.integration` ile işaretli (CI'da skip edilir).
CI iskeleti: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — ruff + black + pytest.

---

## Lisans

MIT — bkz. [`LICENSE`](LICENSE).
