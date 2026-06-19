# Proje AURA — Sistem Mimarisi v2.0

**Kapsam:** YZ/İnference katmanı (v1.1, korunmuş) + sistem katmanı (NV, QoD gateway,
event/annotation stream, dashboard/mobil tüketimi, mock↔gerçek sınırı).

> v1.1 (§1–7) YZ mimarisi korunmuştur; üstüne sistem katmanı (§8), yorgunluk/MediaPipe
> gerekçesi (§9) ve kamera enumerasyonu (§10) eklenmiştir.

## Genel Akış
```
[Kamera Girişi]
      ↓
[1. Ön-İşleme] → [2. YOLO26 ROI + ByteTrack + sınıf oyu] ←→ [3. 16/8 Kararlılık]
                       ↓                ↓
              [Sürücü ROI (sıkı)]    [Plaka ROI]
                       ↓                ↓
        [4. Pose/YOLO Driver State] [5. Sweet Spot + LP kırpma + OCR + Füzyon Voting]
                       ↓                ↓
                                 [QoD Tetikleyici]
                                        ↓
                       [6. ID-Merkezli Akümülatör]
                                        ↓
                       [7. Hız — Kalibrasyon Bağımlı]
                                        ↓
                            [Event / Annotation Stream]  →  Dashboard + Mobil
```

> **Yayın diyagramları:** `docs/diagrams/` — yukarıdaki ASCII'nin yayın-kalite Mermaid
> karşılıkları (FTR §3.2 için): [`pipeline_kusbakisi.mmd`](diagrams/pipeline_kusbakisi.mmd)
> (uçtan uca akış + QoD tetikleri), [`sistem_topolojisi.mmd`](diagrams/sistem_topolojisi.mmd)
> (servis topolojisi + gerçek↔mock sınırı), [`plaka_karar_akisi.mmd`](diagrams/plaka_karar_akisi.mmd)
> (plaka onay karar ağacı + dürüstlük zırhları). Render: [`docs/diagrams/README.md`](diagrams/README.md).

---

# YZ Katmanı (v1.1 — korunmuş)

## 1. Dinamik Ön-İşleme Katmanı
Görüntü modele girmeden çevresel gürültüyü temizleyen ilk filtre. Her filtre config'ten
aç/kapa (`preprocessing.*`):
- **Far patlaması maskeleme:** gece farlarının plaka etrafındaki glare'i lokal maskelenir.
- **Motion blur düzeltme:** yüksek hızlı araç bulanıklığı.
- **Yansıma süpürme:** ön cam / ıslak yüzey yansımaları.
- **Occlusion handling:** direk/ağaç/araç çakışmasında geçici görünürlük kaybı yönetimi.

## 2. Aşama 1 — ROI Tespiti ve Takip (YOLO26 + ByteTrack)
Ana tespit motoru. **Varsayılan stok `yolo26l`** (sunucu, doğruluk-önce); `--profile laptop`
ile `yolo26s` (hafif), `--profile v4-finetune` ile 11-sınıf fine-tune. Hedef araç
sınıflarını tespit eder; tam kareyi ağır bileşenlere göndermek yerine **yalnızca iki ROI
kırpması** üretir:

| ROI | Hedef |
|-----|-------|
| Sürücü Kabini | Aşama 2 — YOLO26l Driver State |
| Plaka Bölgesi | Aşama 2 — OCR Konsensüs Döngüsü |

**ByteTrack** her araca benzersiz ID atar; tüm kararlar (hız, sürücü durumu, plaka) bu ID
üzerinde zaman içinde biriktirilir — sistem **ID-merkezli**, kare-merkezli değil.

## 3. Kararlılık — 16/8 Kuralı
Kamera kaynaklı "hayalet" (flickering) tespitlerin durumu bozmasını engelleyen state
machine. Bir ID'nin durumu ancak **16 ardışık karenin ≥8'inde** tutarlı tespit edilirse
güncellenir; aksi halde yüksek güvenli önceki veri korunur (override yok). Kötü ışık ve
occlusion senaryolarında yanlış alarmı engeller.

## 4. Aşama 2 — Sürücü Durumu (iki backend: pose | yolo)
Aşama 1'in ürettiği **Sürücü ROI**'sini girdi alır; telefon, sigara, emniyet kemeri
takmama ve **yorgunluk** durumlarını üretir. **MediaPipe/landmark kütüphanesi
kullanılmaz** (bkz. §9). `models.driver_state.backend` ile seçilir:

- **`yolo`** — fine-tune YOLO26l detection (phone/smoking/no_seatbelt/fatigue sınıfları).
  En doğru yol; bu sınıflar için eğitilmiş ağırlık gerektirir (STOK COCO ağırlığı bu
  sınıfları üretemez — sessiz sıfır).
- **`pose`** (fine-tune ağırlık gerektirmez) — **YOLO26l-pose** (COCO 17 keypoint) ROI'de
  koşulur; **bilek↔ağız / bilek↔kulak GÖRELİ yakınlık** kıyası telefon/sigara çıkarımı
  yapar (eşikler yüz-genişliği biriminde, ölçek-bağımsız). Kulak keypoint'i görünmüyorsa
  karar verilmez (dürüst çekimserlik). Üstüne **hibrit nesne kanıtı**: fine-tune dedektör
  (`v4`, `phone` sınıfı) aynı ROI'de koşulur; NESNE kanıtı geometriden üstündür (telefon
  nesnesi görülünce el-ağızda geometrisi 'sigara' sayılmaz — hoparlörde konuşma durumu).
  ROI ön-işleme: kısa kenar 320px'e büyütme + CLAHE + gamma (cam arkası karanlık kabin).
  - **Opsiyonel özel sigara ikinci-modeli (`smoking_model`):** eğitilmiş `custom_smoking`
    (YOLO26s; held-out **mAP50 0.856 / mAP50-95 0.457**, 557 görsel CigDet/Mendeley) hibrit
    nesne kanıtının **YANINDA** (replace DEĞİL) sürücü ROI'sinde yalnız `smoking` nesnesi arar;
    bulgusu mevcut `smoking` kanıtına OR'lanır (16/8 oylamasından geçer). **Telefon yolu
    (roi_objects + bastırma latch'i) hiç değişmez** — A/B'de drop-in entegrasyon phone
    kanıtını siliyordu (video_2 telefon kaçtı), ayrı kanal regresyonu önler. Ağırlık diskte
    yoksa no-op (loglanır, davranış değişmez).
- **`auto`** (varsayılan): pose ağırlığı (l-pose; yoksa s-pose) diskte varsa `pose`, yoksa `yolo`.

**Sürücü-içi sıkı kırpma (`driver_crop`):** Modele giden alan MINIMUM tutulur. Gelen ROI
(DriverLock kişi kutusu yoksa kabin fallback'i — araç üst %55'i, ön cam + yolcu
yansımaları dahil) önce **sürücünün kişi kutusuna (+%10)** daraltılır; pose ve nesne
kanıtı yalnız bu dar kırpıkta koşar. Kutu track başına önbelleğe alınır (normalize
koordinat; sürücü araç içinde sabit oturur), `redetect_every` karede bir tazelenir →
kare başına TEK pose geçişi korunur. Bu sayede l-pose gibi büyük model affordable olur
(minimum alana maksimum model). Sürücü tarafı `driver_lock.corner` ile aynı sözleşme.

**İki katman (v2.3 — `DriverStateEngine`):** Aşama 2 artık iki katmana ayrılmıştır:
- **Katman A (model):** yukarıdaki pose-hibrit / yolo backend — tek-kare HAM bayrak.
- **Katman B (`aura/driver_state/engine.py` + `voting.py`):** her `track_id` için ayrı
  **zaman-oylaması** (`TrackVoter`); bir bayrak son `window` karenin ≥`min_votes` tanesinde
  True ise aktiftir (varsayılan 16/8 = eski davranış). Araç sahneden çıkınca tampon
  `max_age` karede düşer (bellek). Bu, eski per-(track,alan) `StabilityTracker` çağrısının
  **ID-merkezli** karşılığıdır; ayarlar `models.driver_state.voting.*`. Stage-1'in tam
  karede gördüğü `phone`/`smoking` nesneleri oylamadan ÖNCE ham bayrağa OR'lanır
  (`fuse_detections`/`aux_classes`). Mustafa'nın `feature/stage2-driver-state` dalının
  iki-katman tasarımı bu sürümde regresyonsuz entegre edildi.

Gerçek 4K test videolarında doğrulandı (sigara: video_1 — sürücü kırpma + l-pose; telefon:
video_2 — 110+ kare; çapraz-FP yok; v4 makro-F1 1.0, dedektör A/B `eval_results/metrics_report.md`).

**Araç-sınıfı kararlılığı (`tracking.class_vote`):** Fine-tune dedektör aynı aracı
kareler arasında farklı sınıflarla görüyor — gerçek ölçüm (13 Haz): video_2'de ana araç
İLK 53 kare ham tespitte yüksek-güvenli (0.84) `truck`, yakınlaşınca kalıcı `car`
(uzakta/arkadan car silüeti truck'a benziyor). Track başına **alan-ağırlıklı sınıf oyu**
(`conf × bbox_alan/kare_alan`) bunu çözer: yakın/büyük araç sınıfı daha güvenilir, az
sayıda yakın `car` karesi çok sayıda uzak `truck` karesini devralır (plaka boyut-farkında
kanıtıyla aynı felsefe) + hafif unutma (`decay`). Sınıf pipeline'da tek noktada
(`det.bbox.cls`) güncellenir → hız genişlik-önseli, accumulator, annotation, event'ler aynı
kararlı sınıfı görür. Ayrıca `min_track_frames` artık bir **çıktı kapısı**: genç (2-karelik
hayalet) track'ler annotation/event üretmez (gerçek video_3'te phantom `truck` track'leri
çıktıya sızmıyordu).

## 5. Plaka Okuma ve Konsensüs Döngüsü
Hesap yükü en yüksek parça; katı kaynak yönetimiyle çalışır.
(Karar ağacının yayın diyagramı: [`docs/diagrams/plaka_karar_akisi.mmd`](diagrams/plaka_karar_akisi.mmd).)
- **Sweet Spot:** OCR'ı aracın kadrajdaki konumuna göre kapılar. **19 Haz fix:** bölge
  eskiden test-videolarının "araç alttan yaklaşır" geometrisine dardı (0.18–0.85 /
  0.40–0.90) → canlı/telefon kamerada araç bölgeye girmeyince OCR hiç tetiklenmiyordu.
  Varsayılan artık **neredeyse tam-kadraj** (0.03–0.97 / 0.06–0.98); kaliteyi frame-bölgesi
  değil **piksel-boyut kapısı** (`lp_vote_min_px`/`min_pixel_height`) + oy havuzu + dürüstlük
  zırhları sınırlar (K-004: oran-bazlı, videoya-özel sabit yok).
- **Sıkı plaka kırpma (LP dedektörü):** varsayılan **özel eğitimli `custom_license_plate`**
  (YOLO26s, tek sınıf `license_plate`; held-out **mAP50 0.983 / mAP50-95 0.707**, 9123 görsel
  keremberke/HF, CC BY 4.0), araç-altı geniş crop içinde plakanın kendisini bulup sıkı kırpar —
  OCR karakter doğruluğu belirgin artar. 3-video stok-vs-custom A/B'de plaka 3/3
  `PLATE_CONFIRMED` korundu (regresyon yok) → varsayılana terfi (`plate.lp_detector.path`).
  Ağırlık yoksa bootstrap iner; o da yoksa loglu olarak stok `lp_yolo11n.pt`/geniş-crop'a düşülür.
- **Boyut-farkında kanıt:** okumanın kanıt değeri = OCR güveni × kaynak kalitesi (LP
  kırpık yüksekliği). Çok küçük plaka (`lp_vote_min_px`) oylamaya hiç girmez; küçük plaka
  (`lp_qod_below_px`) görüldüğü AN `plate_too_small` QoD kalite tetiği (consensus_fail
  beklemeden — havuz çöp okumayla zehirlenmeden). Uzak/bulanık karelerin sistematik
  misread'leri (ör. `34→04`) yakın/net okumayı artık ezemez.
- **OCR güçlendirme:** aynı-satır segment birleştirme ("34"+"TC"+"8532" tek okuma),
  parlama testi (far ışığı OCR'a girmez), küçük plakada CLAHE+2x varyantı.
- **Kalıcı oy havuzu (`aura/plate/normalize.py`):** okumalar track ömrü boyunca birikir
  (redde sıfırlanmaz). TR-blok-farkında normalizasyon (O→0, 1→I...) aday üretir; KARAR
  yalnızca **ikamesiz format-geçerli ham okumalarla**, her okuma **güven × kaynak
  kalitesiyle ağırlıklanarak** verilir (min ağırlık + ikinciye fark + oran — erken/yanlış
  kilit koruması). Kesik okumalar (`8532`) alt-dizi kanıtı olarak adaylara destek verir.
- **Pozisyon-hizalı karakter füzyonu (CONFIRMED'e katılır — pozisyon-margin):** OCR aynı
  plakayı varyantlara böler (`34TC8532`/`04TC8532`/`34IC8532` — `3↔0`, `T↔I` misread'i);
  ayrı-aday kararı bunlar arasında bölünüp hangi varyant baskınsa onu (bazen yanlış `04`)
  seçebiliyordu. Füzyon aynı YAPIDAKİ okumaları pozisyon pozisyon birleştirir; **onay için
  HER pozisyonda kazanan karakter ikinciyi `char_margin` MUTLAK ağırlıkla geçmeli.** Bir
  pozisyon belirsizse (`0↔3` eşit, ya da uzaktan `I↔T`) → dürüst `pending` (yanlış plaka
  ASLA onaylanmaz). Doğru plaka varyantlara bölünmüş olsa bile her pozisyonun çoğunluğu
  doğruysa onaylanır.
- **Karar:** ayrı-aday konsensüsü (min ağırlık + fark + oran) VEYA pozisyon-margin füzyonu
  → plakayı ID'ye kalıcı yaz, OCR kapat (erken çıkış), `PLATE_CONFIRMED` + kalite QoD
  oturumu **hemen bırakılır**. Konsensüs yokken en güçlü aday `PlateState.partial` alanında
  **kanıt izi** (kesin değil) olarak taşınır (şartname 4.5). Post-validasyon: TR plaka regex
  `^\d{2}[A-Z]{1,3}\d{2,4}$`. Gerçek videoda: **video_1 ve video_2'de `34TC8532` CONFIRMED**
  (video_1'de füzyon `3→0`/`T→I` bölünmesini birleştirdi); video_3'te plaka uzak/bulanık
  (OCR `3→2` & `T→I`) → dürüst `pending` (yanlış onay yok; aynı araç video_2'den kesin).

## 6. QoD — Dinamik Kaynak Yönetimi (CAMARA QoD)
5G'yi statik bant genişliği değil, talep üzerine şekillenen dinamik kaynak havuzu olarak
kullanır → mimariyi gerçekten 5G-native kılar.
- **Optimizasyon tetiği (LOW_LATENCY):** hız/yörünge anomalisi/tehlike.
- **Kalite tetiği (HIGH_THROUGHPUT):** voting buffer ret / yetersiz piksel.
- **Histerezis:** minimum aktif süre + cooldown ile tetikle-bırak salınımı önlenir.

## 7. Hız Tahmini (Kalibrasyon Bağımlı)
| Mod | Şart | Yöntem |
|-----|------|--------|
| `tripwire` | sabit kamera + bilinen mesafe | iki çizgi arası ByteTrack frame-delta × gerçek mesafe |
| `ipm` | intrinsics + montaj | homography (opsiyonel modül, §11) |
| `disabled` | kalibrasyon yok | hız üretilmez; `relative_velocity_flag` |
Sistem kendi sınırlarını tanır; kalibrasyon yoksa hız iddiasında bulunmaz.

## 7.4 Swerving — Dikkatsiz Sürüş / Yalpalama Tespiti
Hız tahmincisinin yanında, **kalibrasyon gerektirmeyen** yanal yörünge analizi
(`aura/speed/estimator.py::_swerving_flag`). ZigZag ekstremum sayacı: aracın bbox
merkez-x serisi, mevcut uç noktadan `amp_ratio × o-anki-genişlik` kadar GERİ dönünce bir
yön-değişimi sayılır; `min_flips` dönüşe ulaşan track **swerving** bayrağı alır.

Tasarım garantileri (sentetik + 3 gerçek videoda doğrulandı):
- **Monoton hareket** (kameraya yaklaşma perspektif kayması, tek şerit değişimi
  S-eğrisi) yapısal olarak 0 dönüş üretir → trend modeli gerekmez.
- Eşikler **araç genişliği biriminde** (ölçek-bağımsız) ve pencere **saniye** cinsinden
  (fps-bağımsız) — K-004: hiçbir sabit tek çekime özgü değildir.
- Bayrak 16/8 süzgecinden geçer → `speed.swerving` risk tokenı → `swerving_vehicle`
  kuralı `RISK_ALERT` üretir → QoD optimize tetiklenir.

## 7.5 Sahne Katmanı — Trafik Tabelası + Hız-Limiti Çapraz Kontrolü

Tüm karar akışı **ID-merkezlidir** (her araç bir `TrackRecord`). Trafik tabelası ise
bir araca değil **sahneye** aittir; bu yüzden ID-merkezli accumulator'ın *yanına*
ince bir **sahne katmanı** eklenir (`aura/scene/sign_tracker.py`).

Akış:
```
[YOLO26s] ──(araç/kişi DIŞI tabela sınıfları)──► detector.last_signs
                                                      ↓
                              [SignTracker] ── value_map (speed_limit_50→50)
                                                      ↓
                              SceneContext.active_speed_limit_kmh  (persistence_frames boyunca korunur)
                                                      ↓
   [Accumulator.set_scene] ──► risk koşulu `speed.over_limit` ──► SPEED_LIMIT_VIOLATION
```

- **Tespit:** Dedektör tabelaları araç/kişiden ayrı toplar (`Sign` tipi). Hangi sınıfların
  tabela olduğu `sign.classes` + `sign.value_map` ile config'ten gelir.
- **Aktif limit:** `SignTracker` en güvenilir hız-limiti tabelasını seçer, km/h'ye çözer ve
  araç tabelayı **geçtikten sonra da** `persistence_frames` kare boyunca geçerli tutar (kural sürer).
  Limit değişince tek seferlik `SPEED_LIMIT_DETECTED` (sahne event'i, `track_id=-1`).
- **Çapraz kontrol:** Accumulator her kare `set_scene()` ile aktif limiti alır; `speed.over_limit`
  risk koşulu araç hızını (metric km/h) limitle karşılaştırır → `speed_limit_violation` kuralı →
  zengin payload'lı **`SPEED_LIMIT_VIOLATION`** event (hız / limit / aşım / plaka).
- **Dürüstlük:** Aktif limit yoksa (None) kural pasiftir — kalibrasyonsuz hız gibi, **yanlış ihlal üretmez**.
  Feature config-driven; dedektör `speed_limit_*` sınıflarını üretene dek sessizce pasif kalır, çökmez.

Dashboard tabela kutularını ve sol-üstte "LİMİT" banner'ını çizer (annotation `signs` + `scene` alanları).

---

# Sistem Katmanı (v2.0 — yeni)

## 8. Sistem Mimarisi

### 8.1 Bileşen topolojisi
```
                      ┌────────────────── inference_api (:8080) ───────────────────┐
[Kamera/Video/RTSP] → │  Pipeline (gerçek YZ)  →  EventEmitter                      │
                      │     │                         ├─ MJPEG  GET /stream/video   │
                      │     │                         ├─ WS /stream/annotations     │
                      │     │                         └─ WS /stream/events          │
                      │     └─ QoDController ──(opsiyonel sync)──► qod_mock (:8081)  │
                      └───────────────┬──────────────────────────┬─────────────────┘
                                      │ statik serve             │ WS/HTTP
                                 [Dashboard]                 [Mobil (Expo)] ──► nv_mock (:8082)
```

> Yayın diyagramı (gerçek↔mock sınırı renkli): [`docs/diagrams/sistem_topolojisi.mmd`](diagrams/sistem_topolojisi.mmd).

### 8.2 Event + Annotation stream sözleşmesi (iki-kanal)
- **`AnnotationFrame`** (kare başına): `{frame_id, ts, tracks:[{track_id, bbox, cls, plate,
  driver, speed_kmh, risk_flags, qod_active}]}` → dashboard canvas client-side çizer.
- **`AuraEvent`** (durum değişimi): `{event_id, ts, track_id, type, payload, source}`;
  tipler: DETECTION_UPDATE, PLATE_CONFIRMED/REJECTED, DRIVER_STATE, SPEED, QOD_TRIGGER/RELEASE,
  RISK_ALERT.
- Ham video (MJPEG) ile annotation **ayrı kanaldan** akar → bbox toggle sunucuya gidiş-geliş
  olmadan client'ta yapılır.

### 8.3 Number Verification akışı
Mobil açılışta `POST /verify` (nv_mock :8082) → sessiz doğrulama (SMS/OTP yok, SIM/şebeke
bağı). Doğrulanırsa ana ekran; `WS /stream/events` ile tespitler canlı listelenir.

### 8.4 QoD gateway
QoDController karar + histerezisi in-process yönetir; CAMARA sözleşmesini `qod_mock` (:8081)
taklit eder. Session yaşam döngüsü (aç/sorgula/sil) gateway'de izlenebilir.

### 8.5 Mock ↔ Gerçek sınırı
```
GERÇEK (YZ çekirdeği)           │  MOCK (sözleşme taklidi)
preprocessing, detection,       │  qod_mock (CAMARA QoD)
tracking, stability, driver,    │  nv_mock (Number Verification)
plate/OCR, speed, accumulator,  │  5G şebekesi, TOGG video beslemesi
eval, train                     │
```
**Final ortamında yalnızca endpoint/credential değişir; sözleşme ve YZ çekirdeği aynı kalır.**

## 9. Yorgunluk / MediaPipe Çözümü
**Karar:** Sürücü durumları için MediaPipe/landmark KÜTÜPHANESİ kullanılmaz; ya
fine-tune **YOLO26l detection sınıfı** ya da **YOLO26-pose keypoint geometrisi** (saf
ultralytics, ek bağımlılık yok — §4 `pose` backend) kullanılır. Yorgunluk yalnızca
detection yoluyla çözülür (kapalı göz/esneme/baş düşmesi pose ile güvenilir çıkarılamaz).

**Gerekçe:** Trafik kamerası montaj açıları ve değişken görüş mesafelerinde landmark/pose
sistemleri tutarsız ve kırılgan sonuç üretir (yüz çözünürlüğü düşük, açı uç, occlusion sık).
Yorgunluk; **kapalı göz, esneme, baş düşmesi** sahnelerinin `fatigue` sınıfı olarak
etiketlenip detection ile öğrenilmesiyle çözülür — kabin ROI'si üzerinde, tam kareye gerek
olmadan. Bu, hem dayanıklılık hem de cascade pipeline'ın edge-first hesap bütçesiyle uyumludur.

## 10. Kamera Enumerasyonu + iPhone Continuity
`GET /cameras` OpenCV ile 0–N indekslerini dener; açılan her cihaz için çözünürlük ve
platforma özgü isim döner:
- **macOS:** AVFoundation cihaz adları (`system_profiler SPCameraDataType`).
- **Windows:** DirectShow `filter_info`.
- **Linux:** `/sys/class/video4linux/`. Fallback: `Camera {i}`.
- **iPhone Continuity Camera** (macOS Ventura+): standart webcam olarak bir index'te görünür;
  listede "iPhone Camera (Continuity)" olarak gösterilir.
- **RTSP/IP kamera:** dashboard'da manuel URL girişi (EpocCam, Camo, DroidCam uyumlu).

`AURA_CAMERA_PROBE=0` ile başsız/CI ortamında donanım taraması atlanır.

## 11. Opsiyonel Ek Modüller (§8 toggle)
Sıfır-atık payload, süper çözünürlük, homography/IPM — **default kapalı**, lazy import.
Detay yalnızca **[`docs/mimari_ek_moduller.md`](mimari_ek_moduller.md)**'de.

---

## Mimari Tasarım Kararları — Özet
| Karar | Gerekçe |
|-------|---------|
| Cascade pipeline (YOLO26s→YOLO26l) | Ağır modeli yalnızca gerektiği boyut/noktada çalıştır |
| ID-merkezli birikim | Kare-bazlıya göre tutarlı, sürdürülebilir karar |
| 16/8 state machine | Flickering / geçici gürültüden izole kararlı durum |
| CAMARA QoD | 5G'yi dinamik kaynak havuzu olarak kullan (5G-native) |
| Edge-first | İşlem uçta; downstream'e yalnızca event/annotation stream |
| Decoupled output | YZ modülü upstream/downstream'i bilmez (mikroservis) |
| Kalibrasyon-bağımlı hız | Sistemin kendi sınırlarını tanıması |
| No-MediaPipe yorgunluk | Trafik kamerası koşullarında dayanıklılık (§9) |
| Pose-geometri sürücü davranışı | Fine-tune ağırlık beklemeden gerçek tespit; saf YOLO26, ek bağımlılık yok (§4) |
| Swerving = ZigZag yanal analiz | Kalibrasyonsuz, fps/ölçek-bağımsız dikkatsiz-sürüş kanıtı (§7.4) |
| Kanıt-izi alanları (partial, --save-events) | Şartname 4.5: kanıtlanamayan hedef puanlanmaz |

## Şartname İzlenebilirlik (özet)
Tam eşleme: [`docs/sartname_izlenebilirlik.md`](sartname_izlenebilirlik.md).
| Şartname | Bileşen |
|---|---|
| Araç/plaka/hız/araç-içi tespit (%40) | `aura/` YZ çekirdeği + `train/` + `aura/eval` |
| QoD yalnızca kritik anda + kanıt (%40) | `aura/qod` + `qod_mock` + A/B harness + dashboard paneli |
| Number Verification | `nv_mock` + `mobile/` |
| Tespitlerin mobil gösterimi | `mobile/` + `WS /stream/events` |
| Modern mimari / rapor (%20) | repo yapısı + `docs/` + CI |
