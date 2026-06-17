# AURA — Repo Detaylı Anlatım

**Repo:** `cleoanka/teknofest-prototip.v2`
**Proje:** AURA — 5G & Yapay Zekâ ile Akıllı Yol Güvenliği (TEKNOFEST 2026 prototipi)
**Lisans:** MIT · **Diller:** Python %81 · JavaScript %7 · TypeScript %4 · CSS/HTML/PowerShell

> **v2.3 güncel durum (2026-06-17):** dedektör omurgası varsayılan **stok YOLO26l** (sunucu),
> seçilebilir config **profilleri** (`--profile server|laptop|v4-finetune`); sürücü durumu
> **iki katmanlı** (`DriverStateEngine`); FTR metrik harness'ı (`aura.eval --metrics-report`);
> eğitim tool'u doğrulama+metrik export ile mükemmelleştirildi; `tools/doctor.py`. Güncel
> ayrıntı için: `README.md`, `CHANGELOG.md` (2.3.0), `docs/`, ve FTR rehberi `ftr.md`.

---

## 1. Proje Nedir, Ne Yapar?

AURA, bir trafik kamerası görüntüsünden **araç, plaka, sürücü davranışı ve hız** tespiti yapan bir yapay zekâ çekirdeğini, bu çekirdeği **5G telekom yetenekleriyle** (CAMARA QoD ve Number Verification) birleştiren uçtan uca bir sistemdir. TEKNOFEST 2026 "5G & Yapay Zekâ ile Akıllı Yol Güvenliği" yarışmasının şartnamesine göre tasarlanmıştır (şartname PDF'i de repoda mevcut).

Tek cümlede: Kameradan gelen görüntüyü işleyip "kim, hangi plakayla, ne hızda, dikkatsiz/yorgun/kemeresiz mi sürüyor?" sorularını yanıtlar ve yalnızca **kritik anlarda** 5G ağ kalitesini talep üzerine yükseltir.

### En kritik tasarım kararı: Gerçek / Mock sınırı

Projenin en önemli özelliği bu ayrımdır:

- **GERÇEK (gerçekten kod ile çalışan):** Tüm YZ çekirdeği — ön-işleme, tespit, takip, kararlılık, sürücü durumu, plaka/OCR, hız, accumulator, değerlendirme ve eğitim.
- **MOCK (gerçek API sözleşmesini birebir taklit eden):** Ağ/telekom katmanları — QoD gateway (`qod_mock`), Number Verification (`nv_mock`), 5G şebekesi ve TOGG video beslemesi.

Bu sayede final ortamına geçişte **yalnızca endpoint/credential değişir**; sözleşme ve YZ çekirdeği aynı kalır. Bu, "elimizde gerçek 5G şebekesi yokken bile sistemi uçtan uca çalıştırıp gösterebiliriz" demektir.

### Ağırlıksız (mock modda) çalışabilme

Eğer YOLO26 model ağırlıkları yoksa, sistem **deterministik mock modda** tüm hattı (tespit → plaka → sürücü → hız → QoD → event) baştan sona koşturur. Yani demo ve testler model olmadan da geçer. Bu, jüri/değerlendirici makinesinde gigabaytlık model indirmeden çalıştırma imkânı sağlar.

---

## 2. Mimari Akış (Pipeline)

Sistem **cascade (kademeli) pipeline** mantığıyla çalışır — hafif model önce, ağır model sonra ve yalnızca gerektiği yerde:

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

Aşağıda her aşama tek tek açıklanmıştır.

### Aşama 1 — Dinamik Ön-İşleme

Görüntü herhangi bir modele girmeden çevresel gürültüyü temizleyen ilk filtredir. Her filtre config'ten açılıp kapatılabilir:

- **Far patlaması maskeleme (headlight suppression):** Gece araç farlarının plaka etrafında oluşturduğu beyaz glare bölgesi maskelenir.
- **Motion blur düzeltme:** Yüksek hızlı araçların hareket bulanıklığını giderir.
- **Yansıma süpürme:** Ön cam ve ıslak yüzey yansımalarını bastırır.
- **Occlusion handling:** Direk/ağaç/araç çakışması gibi geçici kapanmaları yönetir.

### Aşama 2 — Tespit ve Takip (YOLO26s + ByteTrack)

Uç cihazda çalışan ana tespit motoru. Hedef araç sınıflarını (araba, kamyon, otobüs, minibüs, motosiklet) tespit eder. **Püf noktası:** Tam çözünürlüklü kareyi ağır bileşenlere göndermek yerine yalnızca **iki ROI kırpması** üretir:

| ROI | Nereye gider |
|-----|--------------|
| Sürücü Kabini | YOLO26l Driver State'e |
| Plaka Bölgesi | OCR Konsensüs Döngüsüne |

Böylece pahalı modüller hiçbir zaman tam kare üzerinde çalışmaz; hesap yükü ve gecikme minimumda kalır.

**ByteTrack** her araca benzersiz bir ID atar. Tüm sistem kararları (hız, sürücü durumu, plaka) bu ID üzerinde zaman içinde biriktirilir. Sistem **ID-merkezli** çalışır, kare-merkezli değil — bu da tutarlı karar üretiminin temelidir.

### Aşama 3 — Kararlılık: 16/8 Kuralı (State Machine)

Kamera kaynaklı anlık "hayalet" tespitlerin (flickering) sistemi yanıltmasını engelleyen state machine katmanıdır.

**Kural:** Bir ID'ye atanmış durumun güncellenebilmesi için, sistemin **16 ardışık karenin en az 8'inde** yeni durumu tutarlı biçimde tespit etmesi şarttır. Bu eşik sağlanmazsa önceki yüksek-güvenilirlikli veri korunur, override yapılmaz. Özellikle kötü ışık ve geçici kapanma senaryolarında yanlış alarmı önler.

### Aşama 4 — Sürücü Durumu (YOLO26l)

Aşama 1'in ürettiği Sürücü Kabini ROI'sini girdi alır ve şu durumları tespit eder: **telefon kullanımı, sigara içme, emniyet kemeri takmama, yorgunluk.** Birden fazla sınıf aynı anda aktif olabilir (bu bir *detection* problemi, *classification* değil).

**Önemli mimari karar — MediaPipe yok:** Yorgunluk dahil tüm sürücü durumları YOLO26l detection sınıfı olarak öğrenilir. Landmark/pose tabanlı yaklaşımlar (MediaPipe gibi) kullanılmaz. Gerekçe: Trafik kamerası montaj açıları ve değişken görüş mesafelerinde landmark sistemleri tutarsız ve kırılgan sonuç üretir (yüz çözünürlüğü düşük, açı uç, occlusion sık). Yorgunluk; kapalı göz, esneme, baş düşmesi sahnelerinin `fatigue` sınıfı olarak etiketlenmesiyle çözülür.

### Aşama 5 — Plaka Okuma ve Konsensüs Döngüsü

Hesap yükü en yüksek parça olduğu için katı kaynak yönetimiyle çalışır:

- **Sweet Spot (Sanal Okuma Bölgesi):** Araç uzaktayken OCR pasiftir. Araç, kameranın en yüksek optik netlik sağladığı önceden tanımlı sanal koordinata girince OCR etkinleşir (config'te `x1,y1,x2,y2` ile tanımlı).
- **Voting Buffer (Oy Havuzu):** Araç sweet-spot içindeyken ardışık OCR okumaları havuzlanır (varsayılan 7 okuma, %60 konsensüs oranı).
- **Karar:**
  - **Konsensüs varsa** → plaka ID'ye kalıcı yazılır, OCR kapatılır (erken çıkış), `PLATE_CONFIRMED` event'i üretilir.
  - **Konsensüs yoksa / piksel yetersizse** → `QOD_TRIGGER` (kalite tetiği) + yeniden okuma döngüsü.
- **Post-validasyon:** Türk plaka regex'i `^\d{2}[A-Z]{1,3}\d{2,4}$` ile doğrulanır.

### Aşama 6 — QoD: Dinamik Kaynak Yönetimi (CAMARA QoD)

Bu, projeyi gerçekten **5G-native** kılan parçadır. 5G ağını statik bir bant genişliği olarak değil, talep üzerine şekillenen **dinamik bir kaynak havuzu** olarak kullanır. Yalnızca gerektiğinde devreye girer:

- **Optimizasyon tetiği (LOW_LATENCY):** Hız/yörünge anomalisi veya tehlike sezildiğinde — amaç gecikmeyi düşürmek, FPS'i artırmak.
- **Kalite tetiği (HIGH_THROUGHPUT):** Voting buffer ret kararı verdiğinde veya piksel kalitesi yetersiz kaldığında — amaç yüksek çözünürlük talep etmek.
- **Histerezis:** Minimum aktif süre (3 sn) + cooldown (5 sn) ile "tetikle-bırak-tetikle" salınımı önlenir.

### Aşama 7 — Hız Tahmini (Kalibrasyon Bağımlı)

Hız ölçümü kamera kurulumuna bağlı olduğundan üç moddan biriyle çalışır:

| Mod | Şart | Yöntem |
|-----|------|--------|
| `tripwire` | Sabit kamera + bilinen mesafe | İki sanal çizgi arası ByteTrack frame-delta × gerçek mesafe |
| `ipm` | Kamera intrinsics + montaj verisi | Homography/IPM ile piksel → gerçek dünya dönüşümü |
| `disabled` | Kalibrasyon verisi yok | Hız üretilmez; bunun yerine `relative_velocity_flag` (göreli hız bayrağı) üretilir |

**Mimari felsefe:** Kalibrasyon şartları sağlanamıyorsa sistem hız iddiasında bulunmaz. Bu, sistemin kendi sınırlarını tanıyan ve aşmayan bir tasarımıdır — yarışma jürisine karşı dürüstlük ve güvenilirlik gösterir.

---

## 3. Sistem Katmanı (Servisler ve Akış)

YZ çekirdeğinin üzerine 3 servisli bir mikroservis mimarisi oturur:

```
                      ┌──────────── inference_api (:8080) ────────────┐
[Kamera/Video/RTSP] → │  Pipeline (gerçek YZ) → EventEmitter          │
                      │     │                    ├─ MJPEG GET /stream/video
                      │     │                    ├─ WS /stream/annotations
                      │     │                    └─ WS /stream/events
                      │     └─ QoDController ──► qod_mock (:8081)      │
                      └──────────┬────────────────────┬───────────────┘
                            [Dashboard]          [Mobil (Expo)] ──► nv_mock (:8082)
```

| Servis | Port | Görev |
|--------|------|-------|
| `inference_api` | 8080 | FastAPI; gerçek YZ pipeline'ı, video/event akışları, dashboard'u serve eder |
| `qod_mock` | 8081 | CAMARA QoD sözleşmesini taklit eder (session CRUD) |
| `nv_mock` | 8082 | Number Verification sessiz doğrulama (SMS/OTP yok, SIM/şebeke bağı) |

### İki kanallı akış tasarımı (önemli detay)

Sistem, ham video ile bbox çizimlerini **ayrı kanallardan** akıtır:

- **MJPEG video** (`GET /stream/video`) — ham görüntü kareleri.
- **`AnnotationFrame`** (`WS /stream/annotations`) — kare başına bbox koordinatları: `{frame_id, ts, tracks:[{track_id, bbox, cls, plate, driver, speed_kmh, risk_flags, qod_active}]}`.
- **`AuraEvent`** (`WS /stream/events`) — durum değişimi event'leri.

Bu sayede dashboard'da **bbox aç/kapa, sunucuya gidip gelmeden client tarafında** yapılır (MJPEG akışı kesilmez). Performans açısından zarif bir çözüm.

### Number Verification akışı

Mobil uygulama açılışta `POST /verify` (nv_mock) çağırır → sessiz doğrulama (kullanıcıya SMS/OTP sorulmaz, SIM/şebeke seviyesinde doğrulanır). Doğrulanırsa ana ekrana geçilir ve `WS /stream/events` ile tespitler canlı listelenir.

---

## 4. Repo Haritası (Dizin Dizin)

| Dizin | İçerik |
|-------|--------|
| `aura/` | YZ çekirdeği — preprocessing, detection, stability, driver_state, plate, speed, accumulator, qod, events, pipeline, eval, optional |
| `services/` | `inference_api` (FastAPI) + `qod_mock` + `nv_mock` |
| `dashboard/` | Vanilla JS + Canvas profesyonel web arayüzü (build/npm yok) |
| `mobile/` | Expo (React Native + TypeScript) iskeleti |
| `train/` | YOLO26 fine-tune pipeline'ları (detector / driver-state / dataset) |
| `config/` | `default.yaml` — tek config kaynağı |
| `weights/` | Model ağırlıkları (bootstrap doldurur, `.gitignore`'lu) |
| `data/samples/` | Örnek video + ground-truth |
| `docs/` | Mimari, kurulum, CLI/API referans, değerlendirme, izlenebilirlik belgeleri |
| `tests/` | pytest (state machine, voting, risk, QoD, API sözleşmeleri) |

### Kök dizindeki önemli dosyalar

- `bootstrap.py` — Saf stdlib kurulum scripti: venv kurar, torch backend'ini otomatik tespit eder (Apple Silicon→MPS, NVIDIA→CUDA, diğer→CPU), paketleri kurar, model ağırlıklarını otomatik indirir (SHA256 trust-on-first-use), örnek video üretir, smoke test koşar. **Idempotenttir** — ikinci çalıştırmada tamamlanmış adımları atlar.
- `setup.sh` / `setup.ps1` ve `run.sh` / `run.ps1` — cross-platform sarmalayıcılar (macOS/Linux + Windows).
- `config/default.yaml` — sistemin **tek doğruluk kaynağı**; hiçbir eşik/flag koda gömülmez.
- `AURA_YZ_Mimarisi_v1.1.md` — orijinal YZ mimari taslağı (yalnızca inference katmanı).
- `docs/mimari.md` — genişletilmiş v2.0 mimarisi (YZ + sistem katmanı).
- `2026_5G_..._SARTNAMESI_TR.pdf` — yarışma şartnamesi.

---

## 5. Yapılandırma Felsefesi (`config/default.yaml`)

Tüm çalışma zamanı davranışı tek YAML dosyasından yönetilir — kodda gömülü sihirli sayı yoktur. Öne çıkan ayar blokları:

- **`runtime`** — cihaz seçimi (`auto/cpu/cuda/mps`), kaynak, log seviyesi ve **`ai_mode`** (`real` = gerçek YOLO, `mock` = numpy deterministik, `auto` = ağırlık varsa gerçek).
- **`models`** — detector (YOLO26s, conf 0.35) ve driver_state (YOLO26l, conf 0.40, 4 sınıf) ayarları; mock dedektör eşikleri.
- **`stability`** — `window: 16`, `min_consistent: 8` (16/8 kuralı).
- **`plate`** — sweet spot koordinatları, voting buffer boyutu (7), konsensüs oranı (0.6), Türk plaka regex'i, minimum piksel yüksekliği.
- **`speed`** — mod (`disabled` varsayılan), tripwire çizgi konumları ve gerçek mesafe.
- **`qod`** — backend (`mock`), profiller (LOW_LATENCY / HIGH_THROUGHPUT), histerezis süreleri.
- **`risk`** — ID-merkezli risk kuralları; örneğin `distracted_speeding` (telefon + yüksek hız → high), `prolonged_fatigue` (yorgunluk + uzun ömürlü track → high), `unbelted` (kemersiz → medium). Genişletilebilir.
- **`optional_modules`** — sıfır-atık payload, süper çözünürlük, homography/IPM — hepsi varsayılan kapalı.

---

## 6. Değerlendirme ve QoD Kanıtı (Şartmanın Kalbi)

Yarışma şartnamesinde QoD kullanımı puanının %40 olması nedeniyle, projenin **A/B harness** aracı kritik öneme sahiptir. `aura/eval` modülü aynı videoyu iki kez koşar:

- **QoD ON** — tam çözünürlük (5G kaynak artırımı simüle edilmiş).
- **QoD OFF** — düşük çözünürlük.

Sonra her ikisini ground-truth'a karşı karşılaştırıp ölçülebilir delta üretir:

| Metrik | QoD OFF | QoD ON | Delta |
|--------|---------|--------|-------|
| Plaka doğruluğu | %33.3 | %66.7 | **+33pp** |
| Küçük nesne tespiti | %46.8 | %98.2 | **+51pp** |
| Tespit oranı | %74.5 | %100 | **+25pp** |

Bu sayılar `python -m aura.eval` veya `POST /eval/run` ile üretilir, dashboard'daki Chart.js paneli görselleştirir, `/eval/results/export` ile Markdown rapor indirilir. Yani "QoD gerçekten fark yaratıyor mu?" sorusuna **somut, tekrar üretilebilir kanıt** sunar. Metrikler Levenshtein/CER tabanlı hesaplanır.

---

## 7. API Yüzeyi (Özet)

`inference_api` (:8080) yaklaşık 15 endpoint sunar. Öne çıkanlar:

- **Sistem:** `GET /health`, `GET /info`
- **Kamera/kaynak:** `GET /cameras` (OpenCV ile kamera tarama + platform isim çözümleme; iPhone Continuity Camera ve RTSP/IP kamera desteği), `POST /stream/start`, `POST /stream/stop`, `PATCH /stream/config`, `GET /stream/status`
- **Video:** `GET /stream/video` (MJPEG), `WS /stream/annotations`, `WS /stream/events`
- **Track:** `GET /tracks`, `GET /tracks/{id}`, `GET /tracks/{id}/history`
- **Değerlendirme:** `POST /eval/run`, `GET /eval/results`, `GET /eval/results/export`
- **Config:** `GET /config`, `PATCH /config`

**Event tipleri:** `DETECTION_UPDATE, PLATE_CONFIRMED, PLATE_REJECTED, DRIVER_STATE, SPEED, QOD_TRIGGER, QOD_RELEASE, RISK_ALERT`.

`qod_mock` (:8081) CAMARA sözleşmesini taklit eder (session aç/sorgula/sil). `nv_mock` (:8082) `POST /verify` ile sessiz doğrulama döner.

---

## 8. Kalite, Test ve CI

- **58 unit test** (mock modda, model ağırlığı gerektirmez) — `pytest -m "not integration"`.
- Model gerektiren testler `@pytest.mark.integration` ile işaretli, CI'da skip edilir.
- **`ruff` + `black`** ile lint/format temiz.
- **GitHub Actions CI** (`.github/workflows/ci.yml`): ruff + black + pytest; torch/ultralytics olmadan hafif kurulumla koşar.
- Test kapsamı: state machine (7/16→ret, 8/16→kabul, flicker senaryoları), voting/konsensüs, risk kuralları, QoD histerezis, API sözleşmeleri, eval metrikleri, opsiyonel modüllerin lazy-import davranışı.

---

## 9. Geliştirme Yolculuğu (16 Milestone)

CHANGELOG, projenin disiplinli ve katmanlı kurulduğunu gösterir. Kısa özet:

1. **M1** — Repo iskeleti + bootstrap + config + ağırlıklar + smoke test.
2. **M2** — Pydantic v2 sözleşmeleri (`schema.py`) + pipeline iskeleti + accumulator + event emitter.
3. **M3** — Detection + ByteTrack + ROI crop (gerçek + mock dedektör).
4. **M4** — Stability (16/8) + driver_state.
5. **M5** — Plate (sweet spot + voting + OCR) + QoD kalite tetiği.
6. **M6** — Speed (disabled/tripwire/ipm) + speed anomalisi → QoD.
7. **M7** — Servisler: inference_api + qod_mock + nv_mock.
8. **M8** — Dashboard (kamera seçici + MJPEG+Canvas + bbox toggle + event log + track panel).
9. **M9** — QoD A/B paneli (eval harness + Chart.js).
10. **M10** — Train modülü + eğitim/veri seti dokümanları.
11. **M11** — Mobil Expo iskeleti (NV giriş + canlı event listesi).
12. **M12** — §8 opsiyonel modüller (toggle + lazy import).
13. **M13** — Her yerde CLI `--help` + CLI referansı.
14. **M14** — API referans dokümanı.
15. **M15** — Mimari v2.0 + doküman tamamlama.
16. **M16** — Testler + CI + izlenebilirlik (şartname ↔ modül eşlemesi).

---

## 10. Hızlı Başlangıç

**macOS / Linux:**
```bash
./setup.sh   # bağımlılıklar + ağırlıklar + örnek veri + smoke (tek komut)
./run.sh     # inference :8080, QoD mock :8081, NV mock :8082
```

**Windows (PowerShell 7+):**
```powershell
.\setup.ps1
.\run.ps1
```

Ardından: Dashboard → `http://localhost:8080/` · OpenAPI → `http://localhost:8080/docs`

---

## 11. Güçlü Yönler ve Genel Değerlendirme

**Güçlü yanları:**

- **Şartname izlenebilirliği:** Her şartname maddesi bir modüle eşlenmiş (`docs/sartname_izlenebilirlik.md`) — jüriye karşı net argüman.
- **Gerçek/mock ayrımı:** Gerçek 5G şebekesi olmadan uçtan uca demo; final'de sadece endpoint değişir.
- **Ağırlıksız çalışma:** Model olmadan tüm hat deterministik koşar — taşınabilir demo.
- **Ölçülebilir QoD kanıtı:** A/B harness somut delta üretir (%40 puan için kanıt aracı).
- **Mimari olgunluk:** Cascade pipeline, ID-merkezli birikim, 16/8 kararlılık, edge-first işlem, decoupled mikroservisler.
- **Mühendislik disiplini:** 58 test, lint/format temiz, CI, cross-platform, tek config kaynağı, kapsamlı dokümantasyon.

**Dikkat edilecek noktalar:**

- YOLO26s/YOLO26l ve OCR şu an **yer tutucu** — projenin kendi veri setiyle eğitilmiş custom modellerle değiştirilmesi planlanıyor (mimari dokümanda belirtilmiş).
- Telekom katmanı tamamen mock; gerçek CAMARA/5G entegrasyonu final aşamasında yapılacak.
- Hız modülü varsayılan `disabled` — kalibrasyon verisi olmadan mutlak hız üretilmiyor (bilinçli karar).

---

*Bu doküman repo README'si, v1.1/v2.0 mimari belgeleri, `config/default.yaml`, CHANGELOG (16 milestone) ve API referansı incelenerek hazırlanmıştır.*
