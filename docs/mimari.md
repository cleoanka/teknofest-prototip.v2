# Proje AURA — Sistem Mimarisi v2.0

**Kapsam:** YZ/İnference katmanı (v1.1, korunmuş) + sistem katmanı (NV, QoD gateway,
event/annotation stream, dashboard/mobil tüketimi, mock↔gerçek sınırı).

> v1.1 (§1–7) YZ mimarisi korunmuştur; üstüne sistem katmanı (§8), yorgunluk/MediaPipe
> gerekçesi (§9) ve kamera enumerasyonu (§10) eklenmiştir.

## Genel Akış
```
[Kamera Girişi]
      ↓
[1. Ön-İşleme] → [2. YOLO26s ROI + ByteTrack] ←→ [3. 16/8 Kararlılık]
                       ↓                ↓
              [Sürücü Kabini ROI]   [Plaka ROI]
                       ↓                ↓
        [4. YOLO26l Driver State] [5. Sweet Spot + OCR + Voting]
                       ↓                ↓
                                 [QoD Tetikleyici]
                                        ↓
                       [6. ID-Merkezli Akümülatör]
                                        ↓
                       [7. Hız — Kalibrasyon Bağımlı]
                                        ↓
                            [Event / Annotation Stream]  →  Dashboard + Mobil
```

---

# YZ Katmanı (v1.1 — korunmuş)

## 1. Dinamik Ön-İşleme Katmanı
Görüntü modele girmeden çevresel gürültüyü temizleyen ilk filtre. Her filtre config'ten
aç/kapa (`preprocessing.*`):
- **Far patlaması maskeleme:** gece farlarının plaka etrafındaki glare'i lokal maskelenir.
- **Motion blur düzeltme:** yüksek hızlı araç bulanıklığı.
- **Yansıma süpürme:** ön cam / ıslak yüzey yansımaları.
- **Occlusion handling:** direk/ağaç/araç çakışmasında geçici görünürlük kaybı yönetimi.

## 2. Aşama 1 — ROI Tespiti ve Takip (YOLO26s + ByteTrack)
Uç cihazda çalışan ana tespit motoru. Hedef araç sınıflarını tespit eder; tam kareyi
ağır bileşenlere göndermek yerine **yalnızca iki ROI kırpması** üretir:

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

## 4. Aşama 2 — Sürücü Durumu (YOLO26l)
Aşama 1'in ürettiği **Sürücü Kabini ROI**'sini girdi alır; şu durumları tespit eder:
telefon, sigara, emniyet kemeri takmama, **yorgunluk**. Çoklu sınıf aynı anda aktif
olabilir (detection, classification değil). **MediaPipe/landmark kullanılmaz** (bkz. §9).

## 5. Plaka Okuma ve Konsensüs Döngüsü
Hesap yükü en yüksek parça; katı kaynak yönetimiyle çalışır.
- **Sweet Spot:** araç uzaktayken OCR pasif; önceden tanımlı sanal koordinata (en yüksek
  optik netlik) girince etkinleşir.
- **Voting Buffer:** sweet-spot içinde ardışık okumalar havuzlanır.
- **Karar:** konsensüs → plakayı ID'ye kalıcı yaz, OCR kapat (erken çıkış),
  `PLATE_CONFIRMED`. Ret → `QOD_TRIGGER` (kalite) + yeniden okuma. Post-validasyon: Türk
  plaka regex `^\d{2}[A-Z]{1,3}\d{2,4}$`.

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
**Karar:** Yorgunluk dahil tüm sürücü durumları **YOLO26l detection sınıfı** olarak öğrenilir;
MediaPipe/landmark tabanlı hiçbir yaklaşım kullanılmaz.

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

## Şartname İzlenebilirlik (özet)
Tam eşleme: [`docs/sartname_izlenebilirlik.md`](sartname_izlenebilirlik.md).
| Şartname | Bileşen |
|---|---|
| Araç/plaka/hız/araç-içi tespit (%40) | `aura/` YZ çekirdeği + `train/` + `aura/eval` |
| QoD yalnızca kritik anda + kanıt (%40) | `aura/qod` + `qod_mock` + A/B harness + dashboard paneli |
| Number Verification | `nv_mock` + `mobile/` |
| Tespitlerin mobil gösterimi | `mobile/` + `WS /stream/events` |
| Modern mimari / rapor (%20) | repo yapısı + `docs/` + CI |
