# AURA — Repo Detaylı Anlatım

**Repo:** `cleoanka/teknofest-prototip.v2`
**Proje:** AURA — 5G & Yapay Zekâ ile Akıllı Yol Güvenliği (TEKNOFEST 2026 prototipi)
**Lisans:** MIT · **Diller:** Python %81 · JavaScript %7 · TypeScript %4 · CSS/HTML/PowerShell

> **v2.3 + W1 güncel durum (2026-06-18):** dedektör omurgası varsayılan **stok YOLO26l**
> (sunucu, `conf 0.10`), seçilebilir config **profilleri** (`--profile server|laptop|v4-finetune`,
> `default.yaml` üzerine derin-merge); sürücü durumu **iki katmanlı** (`DriverStateEngine`);
> hız varsayılanı **`metric` oto-kalibrasyon** (plaka/araç-genişliği → ppm, Kalman+EMA) +
> **swerving** (yanal yörünge); FTR metrik harness'ı (`aura.eval --metrics-report` / `--map`);
> plaka OCR varsayılanı **`fast-plate-ocr`** (3 gerçek videoda 3/3 exact, CER 0.0) + W1
> dewarp/enhance + opsiyonel PaddleOCR; eğitim tool'u doğrulama+metrik export ile
> mükemmelleştirildi; `tools/doctor.py` + `tools/bench.py`. **Süren:** açık veriyle YOLO26s
> domain fine-tune (license_plate mAP50 ≈ 0.97, epoch ~12/35; final mAP'ler henüz kesin değil).
> Güncel ayrıntı için: `README.md`, `CHANGELOG.md` (2.3.0), `docs/`, ve FTR rehberi `ftr.md`.

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
[Kamera] → [Ön-İşleme] → [YOLO26l + ByteTrack] ──┬─→ [Sürücü ROI] → Katman A: pose-hibrit / YOLO26l
                                ↑                 │                  Katman B: per-ID 16/8 zaman-oylaması
                          [Sınıf oyu]            └─→ [Plaka ROI] → [LP dedektör + dewarp/enhance + Oylama + OCR]
                                                                           ↓
                                                          [QoD Tetik (yaklaşma/kalite/anomali)]
                              [ID-Merkezli Accumulator] ← [Hız (metric oto-kalibrasyon) + Swerving]
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

### Aşama 2 — Tespit ve Takip (YOLO26 + ByteTrack)

Ana tespit motoru. Varsayılan **stok `yolo26l`** (sunucu, doğruluk-önce; `conf 0.10` — stok model gerçek trafikte düşük güven üretir, FP'ler dedup + track-başına sınıf-oyu + `min_track_frames` ile elenir). **Konfigüre edilebilir:** `--profile laptop` (`yolo26s`/MPS), `--profile v4-finetune` (11-sınıf fine-tune, plaka-kritik). Hedef araç sınıflarını (araba, kamyon, otobüs, minibüs, motosiklet) tespit eder. **Püf noktası:** Tam çözünürlüklü kareyi ağır bileşenlere göndermek yerine yalnızca **iki ROI kırpması** üretir:

| ROI | Nereye gider |
|-----|--------------|
| Sürücü | İki-katmanlı sürücü motoruna (pose-hibrit / YOLO26l) |
| Plaka Bölgesi | LP dedektör + OCR Konsensüs Döngüsüne |

Böylece pahalı modüller hiçbir zaman tam kare üzerinde çalışmaz; hesap yükü ve gecikme minimumda kalır.

**ByteTrack** her araca benzersiz bir ID atar. Tüm sistem kararları (hız, sürücü durumu, plaka) bu ID üzerinde zaman içinde biriktirilir. Sistem **ID-merkezli** çalışır, kare-merkezli değil — bu da tutarlı karar üretiminin temelidir.

### Aşama 3 — Kararlılık: 16/8 Kuralı (State Machine)

Kamera kaynaklı anlık "hayalet" tespitlerin (flickering) sistemi yanıltmasını engelleyen state machine katmanıdır.

**Kural:** Bir ID'ye atanmış durumun güncellenebilmesi için, sistemin **16 ardışık karenin en az 8'inde** yeni durumu tutarlı biçimde tespit etmesi şarttır. Bu eşik sağlanmazsa önceki yüksek-güvenilirlikli veri korunur, override yapılmaz. Özellikle kötü ışık ve geçici kapanma senaryolarında yanlış alarmı önler.

### Aşama 4 — Sürücü Durumu (iki katmanlı, ID-merkezli)

Sürücü ROI'sini girdi alır ve şu durumları tespit eder: **telefon kullanımı, sigara içme, emniyet kemeri takmama, yorgunluk.** Birden fazla sınıf aynı anda aktif olabilir (bu bir *detection* problemi, *classification* değil). İki katmanlı çalışır:

- **Katman A (ham bayrak üretici):** `models.driver_state.backend` ile seçilir. Varsayılan `auto` → pose ağırlığı diskte varsa **YOLO26-pose** keypoint geometrisi (bilek↔ağız/kulak göreli yakınlığı; fine-tune ağırlık gerektirmez), yoksa `yolo` (fine-tune/domain detection ağırlığı). Pose backend'inde fine-tune dedektör ROI'de ayrıca koşar (**hibrit nesne kanıtı**: telefon nesnesi görülürse geometrik 'sigara' bastırılır). Sürücü ROI'si POZİSYONA göre seçilir (kabindeki en alttaki/en sağdaki kişi → sürücü; `aura/identity/driver_lock.py`).
- **Katman B (`DriverStateEngine` + `TrackVoter`):** Her `track_id` için ham bayrak **16/8 zaman-oylamasından** geçer (son 16 karenin ≥8'inde True ise event'e döner) → tek-karelik FP'ler elenir; araç sahneden çıkınca tampon `max_age` karede düşer. **Kemer ihlali (`no_seatbelt`) Katman B'de kemer ŞERİDİNİN yokluğundan türetilir** (`no_seatbelt.enabled`, **varsayılan kapalı** — kemer görünürlüğü düşük footage'da FP koruması).

**Önemli mimari karar — landmark kütüphanesi yok:** Pose backend'i Ultralytics YOLO26-pose keypoint'lerini kullanır; MediaPipe gibi ayrı landmark kütüphaneleri kullanılmaz. Gerekçe: trafik kamerası açıları ve değişken görüş mesafelerinde harici landmark sistemleri kırılgandır.

### Aşama 5 — Plaka Okuma ve Konsensüs Döngüsü

Hesap yükü en yüksek parça olduğu için katı kaynak yönetimiyle çalışır:

- **Sweet Spot (Sanal Okuma Bölgesi):** Araç uzaktayken OCR pasiftir. Araç, kameranın en yüksek optik netlik sağladığı önceden tanımlı sanal koordinata girince OCR etkinleşir (config'te `x1,y1,x2,y2` ile tanımlı).
- **LP dedektör + sıkı kırpma:** Özel plaka dedektörü (YOLOv11n LP fine-tune, ~5MB) plakayı araç-altı geniş crop içinde bulup sıkı kırpar → karakter doğruluğu artar. Ağırlık yoksa loglu olarak geniş-crop'a düşülür.
- **W1 — OCR öncesi hazırlık (`plate.dewarp` + `plate.enhance`):** LP kırpığına OCR'dan hemen önce bir kez **fronto-paralel perspektif düzeltme** (açılı plaka → düz; köşe yoksa kimlik) + **CLAHE/gamma/unsharp** uygulanır (karanlık/açılı otopark footage'ı için). Reader'ın düşük-güven CLAHE+2x ikinci-şans varyantından ayrıdır.
- **OCR motoru (`plate.ocr_engine`):** varsayılan **`fast-plate-ocr`** — plakaya-özel hafif ONNX modeli (`global-plates-mobile-vit-v2`, ~5MB; ilk koşuda otomatik iner). 3 gerçek videoda ölçüldü (18 Haz 2026, GT=34TC8532): **3/3 exact-match, CER 0.0** — EasyOCR baseline'ının video_3'te kalan il-kodu misread'ini (3→2, T→I; konsensüse girmediği için `pending`) giderir, v1/v2 exact'ini korur. Alternatifler: **EasyOCR** ve **PaddleOCR** (`paddleocr` çıktısı EasyOCR `readtext` ile uyumlu normalize edilir). Seçilen motor kurulu değilse her durumda **loglu EasyOCR fallback**. Motor seçimi K-004 uyumlu: config-driven, oran-bazlı, videoya-özel sabit/kara-liste yok.
- **Voting Buffer (Oy Havuzu):** Araç sweet-spot içindeyken ardışık OCR okumaları havuzlanır (varsayılan 7 okuma, %60 konsensüs oranı). Oylar OCR güveni × kaynak kalitesi (LP kırpık yüksekliği) ile ağırlıklanır.
- **Karar:**
  - **Konsensüs varsa** → plaka ID'ye kalıcı yazılır, OCR kapatılır (erken çıkış), `PLATE_CONFIRMED` event'i üretilir. Dürüstlük zırhları: pozisyon-veto (her karakter pozisyonu `char_margin` önde olmalı) + zemin koşulu (`confirm_peak_weight` — kazanan plaka en az bir kez net/yakın okunmuş olmalı). Aksi halde `pending`.
  - **Konsensüs yoksa / piksel yetersizse** → `QOD_TRIGGER` (kalite tetiği) + yeniden okuma döngüsü.
- **Post-validasyon:** Türk plaka regex'i `^\d{2}[A-Z]{1,3}\d{2,4}$` ile doğrulanır.

### Aşama 6 — QoD: Dinamik Kaynak Yönetimi (CAMARA QoD)

Bu, projeyi gerçekten **5G-native** kılan parçadır. 5G ağını statik bir bant genişliği olarak değil, talep üzerine şekillenen **dinamik bir kaynak havuzu** olarak kullanır. Yalnızca gerektiğinde devreye girer:

- **Optimizasyon tetiği (LOW_LATENCY):** Hız/yörünge anomalisi veya tehlike sezildiğinde — amaç gecikmeyi düşürmek, FPS'i artırmak.
- **Kalite tetiği (HIGH_THROUGHPUT):** Voting buffer ret kararı verdiğinde veya piksel kalitesi yetersiz kaldığında — amaç yüksek çözünürlük talep etmek.
- **Histerezis:** Minimum aktif süre (3 sn) + cooldown (5 sn) ile "tetikle-bırak-tetikle" salınımı önlenir.

### Aşama 7 — Hız Tahmini (varsayılan: metric oto-kalibrasyon) + Swerving

Hız varsayılanı artık **`metric`** modudur (`speed.mode: metric`) — manuel kalibrasyon dosyası **gerektirmez**:

| Mod | Şart | Yöntem |
|-----|------|--------|
| `metric` (**varsayılan**) | — (oto-kalibre) | Araç-genişliği (varsa plaka 520 mm referansı) → `ppm(y)` ölçek-alanı (aykırı-dayanıklı regresyon) → yer-düzlemi metrik yer değiştirme → pencere-medyan + ivme aykırı reddi + **Kalman + EMA** → gerçek km/h. Isınma bitene dek `is_calibrated=False` (km/h iddia edilmez). Kadraj kenarında ölü bölge (`frame_margin_px`). |
| `tripwire` | Sabit kamera + bilinen mesafe | İki sanal çizgi arası ByteTrack frame-delta × gerçek mesafe |
| `ipm` | Kamera intrinsics + montaj verisi | Homography/IPM ile piksel → gerçek dünya dönüşümü |
| `disabled` | — | Hız üretilmez; yalnız `relative_velocity_flag` (göreli hız bayrağı) |

**Swerving (dikkatsiz sürüş / yalpalama):** Yanal yörüngede zigzag ekstremum sayacı; pencere **saniye** cinsinden (fps-bağımsız), genlik **araç-genişliği** biriminde (ölçek-bağımsız, K-004). Monoton hareket (yaklaşma kayması, tek şerit değişimi) yapısal olarak 0 üretir. Bayrak 16/8 süzgecinden geçer → `swerving_vehicle` kuralı `RISK_ALERT` üretir + QoD optimize tetiği. Ayrıca `SignTracker` + `speed.over_limit` ile aktif hız-limiti tabelası geçilince ihlal (`SPEED_LIMIT_VIOLATION`) üretilir.

**Mimari felsefe:** Oto-kalibrasyon ısınmasını tamamlamadan sistem mutlak km/h iddia etmez (`is_calibrated=False`). Sistemin kendi sınırlarını tanıyan ve aşmayan bu tasarım — jüriye karşı dürüstlük ve güvenilirlik gösterir.

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
- **`models`** — detector (varsayılan stok **`yolo26l`**, conf **0.10**, imgsz 768) ve driver_state (backend `auto` → pose/yolo; pose/voting 16/8 ayarları, `no_seatbelt` varsayılan kapalı); mock dedektör eşikleri. Dedektör/cihaz/eşikler **config profilleriyle** (`server/laptop/v4-finetune`) derin-merge edilir.
- **`stability`** — `window: 16`, `min_consistent: 8` (16/8 kuralı).
- **`plate`** — sweet spot koordinatları, LP dedektör (`lp_detector`), W1 **`dewarp`** + **`enhance`**, OCR motoru (`ocr_engine: fastplate|easyocr|paddleocr`, varsayılan **`fastplate`**), voting buffer boyutu (7), konsensüs oranı (0.6), Türk plaka regex'i, minimum piksel yüksekliği, dürüstlük zırhları (`char_margin`, `confirm_peak_weight`).
- **`speed`** — mod (**`metric` varsayılan**, oto-kalibrasyon: Kalman+EMA), plaka/araç-genişliği referansları, `swerving` bloğu; alternatifler `tripwire`/`ipm`/`disabled`.
- **`sign`** — hız-limiti tabelası tespiti (`SignTracker`) → `speed.over_limit` ihlali.
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

**FTR metrik harness'ı (`--metrics-report`) ve mAP (`--map`):** `python -m aura.eval --metrics-report` video-düzeyi **P/R/F1 + plaka exact-match/CER + araç doğruluğu + FPS** üretir (dedektöre göre A/B; `eval_results/metrics_report.md/.csv/.json`). `python -m aura.eval --map` ise standart bir doğrulama setinde **mAP** ölçer (`eval_results/map_*.json`). Bu çıktılar Final Tasarım Raporu'nun §4 metrik tablolarını dürüst sayılarla doldurur.

**Ölçülen başarım (held-out, gerçek sayılar):**

| Metrik | Değer | Kaynak |
|--------|-------|--------|
| Plaka exact-match (fast-plate-ocr) | **3/3 (CER 0.0)** | 3 gerçek video, `config/default.yaml` ölçüm notu |
| Davranış makro-F1 | **1.0** | 3 video held-out, `eval_results/metrics_report.md` |
| Araç sınıfı doğruluğu | **%100** | aynı set |
| Stok `yolo26l` mAP50-95 / mAP50 | **0.537 / 0.709** | COCO-val2017 held-out (5000 görsel), `eval_results/map_yolo26l.json` |

> **Süren çalışma (dürüst ayrım):** Yukarıdaki mAP **stok** `yolo26l`'in genel başarımıdır. Açık veriyle (CC BY 4.0; license_plate 9123, seatbelt 3104, phone 659, smoking 557) **YOLO26s domain fine-tune'u sürüyor** (license_plate mAP50 ≈ 0.97 @ epoch ~12/35; smoking + seatbelt sırada). Domain-spesifik final mAP'ler henüz kesinleşmedi. Davranış makro-F1 küçük (3-videoluk) held-out sette çalıştığının kanıtıdır; istatistiksel mAP için geniş etiketli set gerekir.

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

- **600+ unit test** (mock modda, model ağırlığı gerektirmez) — `pytest -m "not integration"` (integration testleri CI'da deselect; `services/` test seti genişletiliyor).
- Model gerektiren testler `@pytest.mark.integration` ile işaretli, CI'da skip edilir.
- **`ruff` + `black`** ile lint/format temiz (CI'da sürümler pinli: `ruff==0.15.17`, `black==26.5.1`).
- **GitHub Actions CI** (`.github/workflows/ci.yml`): ruff + black + pytest; torch/ultralytics olmadan hafif kurulumla koşar.
- Test kapsamı: state machine (7/16→ret, 8/16→kabul, flicker), Katman B sürücü motoru (kemer-türetme dahil), per-track sınıf oyu, voting/konsensüs + plaka dürüstlük zırhları, swerving, metric hız (sentetik doğruluk/ısınma/EMA), sürücü/yolcu kilidi, hız-limiti tabelası, risk kuralları, QoD histerezis, API sözleşmeleri, eval metrikleri (P/R/F1, CER), opsiyonel modüllerin lazy-import davranışı.

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

M1–M16'dan sonra gerçek-video geri bildirimi ve FTR hazırlığıyla sürüm-bazlı iyileştirmeler:

- **v2.1** (gece bakım/yenileme) — pose-tabanlı sürücü davranışı (`driver_state/pose.py`, MediaPipe geometrisinin saf-YOLO portu); **swerving** tespiti; TR plaka normalizasyonu + format-öncelikli kalıcı oy havuzu; Stage-1 kanıt füzyonu; **QoD yaklaşma tetiği** (`vehicle_approach`); fine-tune v4 dedektör birincil; araç-kutusu dedup + ağır-aşama kapısı; `tools/test_video.py` (annotated mp4 + JSON kanıt) + `--save-events` JSONL.
- **v2.2** (geri bildirim turu) — sürücü-içi sıkı kırpma (yalnız sürücü kişi kutusu); l-pose yükseltmesi; per-track araç-sınıfı oylaması (car↔truck titremesi giderildi); pozisyon-hizalı plaka karakter füzyonu (yanlış plaka asla onaylanmaz, aksi halde `pending`); boyut-farkında plaka kanıtı + QoD erken bırakma; Windows araç paritesi (`dev.ps1`); **metric hız** (oto-kalibrasyon: Kalman+EMA) + ölü bölge.
- **v2.3** (YOLO26 sunucu sürümü) — varsayılan dedektör stok **`yolo26l`**; **config profil katmanı** (`server/laptop/v4-finetune`, derin-merge); **iki katmanlı sürücü motoru** (`DriverStateEngine` + `TrackVoter`); FTR §4 **metrik harness'ı** (`--metrics-report`); eğitim tool'u doğrulama+metrik export ile mükemmelleştirildi; plaka CONFIRM dürüstlük zırhları (pozisyon-veto + zemin koşulu); sürücü/yolcu pozisyonel kilidi; plaka→hız oto-kalibrasyonu; kemer iki-katman tasarımı; `tools/doctor.py`; hız-limiti tabelası (`SignTracker`).
- **W1** (FTR ön-hazırlık, `feat/ultraplan-w1`) — plaka OCR varsayılanı **`fast-plate-ocr`** (3 gerçek videoda 3/3 exact, CER 0.0; EasyOCR'ın video_3 il-kodu misread'ini kapatır) + opsiyonel **PaddleOCR**; OCR-öncesi **dewarp + enhance**; `aura.eval --map` (mAP harness, stok yolo26l COCO-val2017 0.537/0.709 ölçüldü); hız mutlak-GT MAE/MAPE doğrulaması; **açık veri toplama** (license_plate/seatbelt/phone/smoking, CC BY 4.0) + YOLO26s domain fine-tune (sürüyor); `tools/bench.py` (FPS + p50/p95 kare-süresi profilleme).

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
- **Mühendislik disiplini:** 600+ test, lint/format temiz (sürüm-pinli CI), cross-platform, tek config kaynağı, kapsamlı dokümantasyon.

**Dikkat edilecek noktalar:**

- Varsayılan dedektör **stok YOLO26l**; sürücü-durum domain modeli ve plaka OCR komite footage'ı ile fine-tune edilince doğruluk artacak (`--profile v4-finetune` halihazırda 11-sınıf fine-tune sunar).
- Telekom katmanı tamamen mock; gerçek CAMARA/5G entegrasyonu final aşamasında yapılacak.
- Hız modülü varsayılan **`metric` oto-kalibrasyon** — ısınma tamamlanana dek mutlak km/h iddia edilmez (`is_calibrated=False`); sabit kamera + bilinen mesafe varsa `tripwire`/`ipm` daha kesindir.

---

*Bu doküman repo README'si, v1.1/v2.0 mimari belgeleri, `config/default.yaml`, CHANGELOG (M1–M16 + v2.1/v2.2/v2.3 + W1) ve API referansı incelenerek hazırlanmıştır.*
