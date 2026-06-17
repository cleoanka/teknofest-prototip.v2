# AURA — 5G & Yapay Zekâ ile Akıllı Yol Güvenliği

> **TEKNOFEST 2026** "5G & Yapay Zekâ ile Akıllı Yol Güvenliği" yarışması prototipi.
> Üretim kalitesinde, cross-platform (macOS + Windows), tek komutla ayağa kalkan monorepo.

AURA; trafik kamerası görüntüsünden **araç / plaka / sürücü davranışı / hız** tespiti yapan
gerçek bir YZ çekirdeği ile, bu çekirdeği **5G QoD** (CAMARA Quality-on-Demand) ve
**Number Verification** gibi telekom yetenekleriyle birleştiren bir sistemdir.

**Gerçek / Mock sınırı:** YZ çekirdeğinin tamamı (CV/tracking/state-machine/OCR/speed/eval)
**gerçektir**. Ağ/telekom/mobil katmanları (QoD gateway, NV API, 5G şebekesi) gerçek API
sözleşmesini birebir taklit eden **mock**'lardır — final ortamında yalnızca endpoint/credential değişir.

**Öne çıkanlar (v2.3 — YOLO26 sunucu sürümü):**
- **YOLO26 omurga, konfigüre edilebilir:** varsayılan Stage-1 dedektör **stok `yolo26l`**
  (sunucu, doğruluk-önce); **config profilleri** (`--profile server|laptop|v4-finetune`)
  `default.yaml` üzerine derin-merge edilir. Sürücü davranışı **YOLO26-pose** geometrisi +
  hibrit nesne kanıtı; plaka **YOLO11n LP dedektörü + format-öncelikli güven-ağırlıklı oylama**.
- **ID-merkezli iki-katmanlı sürücü motoru:** Katman A (pose-hibrit model) + Katman B
  (`DriverStateEngine` per-track 16/8 zaman-oylaması) → tek-kare FP'leri eler, araç çıkınca tampon düşer.
- **FTR'ye hazır metrikler:** `python -m aura.eval --metrics-report` → video-düzeyi
  **P/R/F1 + plaka exact-match/CER + FPS** (dedektör A/B); `python -m aura.eval --map`
  doğrulama setinde **mAP**; hız mutlak-GT **MAE/MAPE** (kalibre kareler). `eval_results/`.
- **W1 plaka & hız sağlamlaştırma:** LP kırpığına OCR-öncesi **dewarp + enhance** (açılı/karanlık
  otopark); opsiyonel **PaddleOCR** motoru (`ocr_engine: paddleocr`, kuruluysa); hız varsayılanı
  **`metric` oto-kalibrasyon** (Kalman+EMA) + **swerving**; `tools/bench.py` ile FPS profilleme.
- **Eğitim boru hattı (YOLO26 fine-tune):** `python -m train` eğit→doğrula→metrik→best;
  `dataset --report` veri-dengeleme dağılımı (FTR §2). Komite verisi gelince tek komut.
- **Ağırlıksız da çalışır:** ağırlık yoksa pipeline deterministik *mock* modda tüm hattı
  (tespit→plaka→sürücü→hız→QoD→event) uçtan uca koşturur; demo ve testler model olmadan geçer.
- **QoD kanıtı:** A/B harness ölçülebilir delta üretir; **yaklaşma tetiği** (`vehicle_approach`)
  şartnamenin "TOGG yaklaşınca QoD" senaryosunu birebir karşılar — %40 QoD puanı için kanıt.
- **Denetim izi + sağlık:** `tools/test_video.py` annotated mp4 + JSON kanıt; `--save-events`
  JSONL iz (şartname 4.5); `python tools/doctor.py` tek-bakış ortam/hazırlık kontrolü.
- **Kalite:** 238 unit test, `ruff` + `black` temiz (sürüm-pinli), GitHub Actions CI.
- **FTR rehberi:** [`ftr.md`](ftr.md) — Final Tasarım Raporu'nu bu kanıtlarla doldurma kılavuzu.

---

## Hızlı Başlangıç

### macOS / Linux
```bash
./setup.sh                       # bağımlılıklar + model ağırlıkları + örnek veri + smoke (tek komut)
python tools/doctor.py           # ortam/hazırlık kontrolü (bağımlılık, cihaz, ağırlık, profil)
./run.sh                         # inference :8080, QoD mock :8081, NV mock :8082
AURA_PROFILE=server ./run.sh     # sunucu profili (yolo26l, CUDA, büyük imgsz)
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
[Kamera] → [Ön-İşleme] → [YOLO26 + ByteTrack] ─┬─→ [Sürücü ROI] → Katman A: YOLO26-pose geometri+hibrit nesne
                              ↑                 │                  Katman B: per-ID 16/8 zaman-oylaması (engine)
                       [Sınıf oyu]             └─→ [Plaka ROI] → [YOLO11n LP + Güven-Ağırlıklı Oylama + OCR]
                                                                          ↓
                                                          [QoD Tetik (yaklaşma/kalite/anomali)]
                              [ID-Merkezli Accumulator] ← [Hız + Swerving (yanal yörünge)]
                                          ↓
                              [Event / Annotation Stream] → Dashboard + Mobil + JSONL kanıt
```

Mimari kararlar (değişmez): cascade pipeline (Stage-1 YOLO26 dedektör → Stage-2 sürücü motoru:
**Katman A model + Katman B ID-merkezli zaman-oylaması**), ID-merkezli birikim, **landmark
kütüphanesi yok** (sürücü davranışı YOLO26-pose keypoint geometrisi veya YOLO26l detection ile —
`models.driver_state.backend`), kalibrasyon-bağımlı hız. Dedektör/cihaz/eşikler **config
profilleriyle** seçilir (`--profile`). Detay: [`docs/mimari.md`](docs/mimari.md).

**Yayın diyagramları (FTR §3.2):** yukarıdaki ASCII'nin yayın-kalite Mermaid karşılıkları
`docs/diagrams/`'tadır — [pipeline kuşbakışı](docs/diagrams/pipeline_kusbakisi.mmd) ·
[sistem topolojisi (gerçek↔mock)](docs/diagrams/sistem_topolojisi.mmd) ·
[plaka karar akışı](docs/diagrams/plaka_karar_akisi.mmd). Render: [`docs/diagrams/README.md`](docs/diagrams/README.md).

---

## Komut Rehberi

| Komut | Açıklama |
|---|---|
| `python bootstrap.py --help` | Kurulum seçenekleri |
| `python tools/doctor.py` | Ortam/hazırlık sağlık kontrolü (bağımlılık, cihaz, ağırlık, profil) |
| `python -m aura --help` | Ana inference pipeline (`--profile`, `--save-events` JSONL kanıt izi) |
| `python tools/test_video.py --help` | Gerçek video testi → annotated mp4 + JSON kanıt (`--profile` ile A/B) |
| `python -m aura.eval --metrics-report` | FTR §4 metrik raporu (P/R/F1 + plaka CER + FPS + dedektör A/B) |
| `python -m aura.eval --map` | Doğrulama setinde mAP (`eval_results/map_*.json`) |
| `python -m aura.eval --qod-comparison` | QoD A/B delta (şartname %40 QoD kanıtı) |
| `python tools/bench.py --help` | Video + profil → ortalama FPS + p50/p95 kare-süresi (`eval_results/bench_<device>.md`) |
| `python -m train --help` | YOLO26 eğitimi (detector / driver-state / dataset `--report`) |
| `python -m aura.synthetic` | Sentetik örnek video + ground-truth üret |
| `make help` | Geliştirme kısayolları (`make doctor` / `metrics` / `test`) |

**Config profilleri** (`config/profiles/*.yaml`, `default.yaml` üzerine derin-merge):
`--profile server` (yolo26l/CUDA/imgsz960) · `laptop` (yolo26s/MPS) · `v4-finetune`
(11-sınıf fine-tune). `AURA_PROFILE` env ile de seçilir.

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
| `tools/` | `test_video.py` (annotated mp4 + JSON kanıt), `doctor.py` (sağlık), `bench.py` (FPS profilleme) |
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
| [`docs/diagrams/`](docs/diagrams/README.md) | Yayın-kalite Mermaid mimari diyagramları (FTR §3.2) |
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
| [`docs/dagitim.md`](docs/dagitim.md) | Sunucu dağıtımı (CUDA, profil, servis, ölçeklenme) |
| [`docs/yol_haritasi.md`](docs/yol_haritasi.md) | Sıradaki işler (plaka dewarp+OCR, açık veri setleri, final hazırlığı) — Gemini destekli |
| [`ftr.md`](ftr.md) | Final Tasarım Raporu doldurma rehberi + doldurulabilir taslak |
| [`gemini.md`](gemini.md) | Gemini CLI kullanım rehberi (araştırma/ikinci-görüş) |

Her dizin kendi `README.md`'sini taşır.

---

## Test & Kalite

```bash
pytest -m "not integration"        # 238 unit test (mock modda, ağırlık gerektirmez)
ruff check . && black --check .    # lint + format
```
Model gerektiren testler `@pytest.mark.integration` ile işaretli (CI'da skip edilir).
CI iskeleti: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — ruff + black + pytest
(sürümler pinli: `ruff==0.15.17`, `black==26.5.1`).

---

## Lisans

MIT — bkz. [`LICENSE`](LICENSE).
