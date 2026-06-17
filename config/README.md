# `config/` — Merkezi Yapılandırma

## Ne yapar
`default.yaml` AURA'nın **tek doğruluk kaynağıdır**. Hiçbir eşik/flag koda gömülmez;
tüm çalışma zamanı davranışı buradan yönetilir. `aura.config.load_config()` bu dosyayı
yükler, seçili env değişkenlerini override olarak uygular ve noktalı erişim sağlar:

```python
from aura.config import load_config
cfg = load_config()                       # sadece default.yaml
cfg = load_config(profile="server")       # default.yaml + profiles/server.yaml (derin-merge)
cfg.get("plate.voting_buffer_size")       # 7
cfg.get("stability.window")               # 16
```

## Profiller (`config/profiles/*.yaml`)
`default.yaml` üzerine **derin-merge** edilen overlay'ler; yalnız farkları içerir. Seçim
sırası: `--profile` argümanı > `AURA_PROFILE` env > yok. CLI: `--profile` bayrağı
(`aura`, `aura.eval`, `tools/test_video.py`, `tools/doctor.py`).

| Profil | Dedektör | Cihaz | imgsz | Hedef |
|---|---|---|---|---|
| `server` | yolo26l | auto (CUDA) | 960 | sunucu, maksimum doğruluk (önerilen) |
| `laptop` | yolo26s | auto (MPS) | 640 | geliştirme, hafif |
| `v4-finetune` | yolguvenligi_types_v4 | auto | 768 | 11-sınıf fine-tune (plaka-kritik) |

Kendi profilinizi ekleyin (`config/profiles/uretim.yaml`) → `--profile uretim`. Liste:
`python -c "from aura.config import available_profiles as a; print(a())"`. Detay: `docs/dagitim.md`.

## Dosyalar
| Dosya | Açıklama |
|---|---|
| `default.yaml` | Aktif config (tüm parametreler) |
| `default.yaml.template` | Bootstrap fallback'i (`default.yaml` yoksa kopyalanır) |
| `calibration/ornek_kamera.yaml` | Tripwire/IPM hız kalibrasyon örneği |

## Parametre referansı

### `runtime`
| Anahtar | Değerler | Açıklama |
|---|---|---|
| `device` | `auto`/`cpu`/`cuda`/`mps` | Torch backend (auto = donanıma göre) |
| `source` | path / index / RTSP URL | Varsayılan girdi kaynağı |
| `log_level` | `DEBUG`/`INFO`/`WARNING` | Log seviyesi |
| `ai_mode` | `real`/`mock`/`auto` | `auto`: ultralytics+ağırlık varsa real, yoksa mock |

### `models.detector` / `models.driver_state`
`path` (ağırlık), `conf` (güven eşiği), `iou`, `imgsz`, sınıf listeleri. Custom fine-tune
ağırlığı için `path`'i değiştirin (bkz. `docs/egitim.md`).

### `stability`
`window: 16`, `min_consistent: 8` → 16/8 kuralı. Yeni durum ancak son 16 karenin ≥8'inde
tutarlıysa yazılır (flicker koruması).

### `plate`
`sweet_spot` (normalize ROI — yanal yaklaşan araçlar için geniş x aralığı),
`voting_buffer_size` (rejected-event kadansı), `consensus_ratio`, `ocr_lang`,
`regex` (Türk plaka), `min_pixel_height` (altında kalite tetiklenir),
`ocr_max_side` / `ocr_enhance_below_px` (OCR girdi boyut yönetimi),
`ocr_engine` (`easyocr` varsayılan | `paddleocr`; paddleocr kurulu değilse loglu EasyOCR
fallback — `pip install 'aura[paddle]'`),
**`dewarp.enabled`** (WP-A1: fronto-paralel perspektif düzeltme — açılı plaka düzleştirilir,
köşe bulunamazsa kimlik) ve **`enhance.enabled`/`enhance.clahe_clip`/`enhance.gamma`**
(WP-A1: CLAHE+gamma+unsharp; LP kırpığına OCR'dan HEMEN ÖNCE bir kez uygulanır — reader'ın
düşük-güven CLAHE+2x ikinci-şans varyantından AYRIDIR; karanlık/açılı otoparkta il-kodu
misread'ini 3→0/2 azaltır — bkz. `aura/plate/dewarp.py`, `aura/plate/enhance.py`),
`lp_detector.*` (sıkı plaka kırpma — özel YOLOv11n plaka modeli; yoksa loglu fallback),
`lp_vote_min_px` / `lp_qod_below_px` (boyut-farkında kanıt: çok küçük LP oylamaya
girmez; küçük LP görüldüğü an `plate_too_small` QoD tetiği — consensus_fail beklemeden),
`voting.*` (kalıcı oy havuzu: `min_weight` = min kanıt, `margin_weight` = kazananla
ikinci arasındaki min **mutlak** fark (ASIL ayrım kriteri); `consensus_ratio` düşük
tutulur, 0.35 — dağınık misread'ler toplamı şişirip oranı düşürdüğünden margin esas
alınır; `fix1/fix2_weight`, `substring_weight`; `size_full_px`/`size_floor`/`no_lp_weight`
= okuma ağırlığı OCR güveni × kaynak kalitesi; `char_consensus` = pozisyon-hizalı karakter
füzyonu (CONFIRMED'e katılır: her pozisyonda kazanan ikinciyi `char_margin` MUTLAK ağırlıkla
geçmeli, değilse `pending`); **`confirm_peak_weight`** (v2.3) = CONFIRM zemin koşulu: kazanan
plaka en az bir kez bu etkin-ağırlıkla (OCR güveni × kırpık kalitesi) okunmuş olmalı —
hep-uzak sistematik misread onaylanmaz; 0 = kapalı. Ek **pozisyon-veto** (v2.3): ayrı-aday
bütün-string marjını geçse bile her karakter pozisyonu belirsizse onay verilmez. — bkz.
`aura/plate/normalize.py`).

### `models.driver_state` (backend seçimi)
`backend: auto|pose|yolo` — pose: YOLO26-pose keypoint geometrisi (`pose_path` =
varsayılan `yolo26l-pose.pt`, yoksa s-pose'a loglu fallback; `pose_conf`,
`pose_kp_conf`, `phone_ear_ratio`, `smoke_mouth_ratio`, `roi_min_side`, `roi_enhance`) +
hibrit ROI nesne kanıtı (`roi_objects.*`); yolo: fine-tune YOLO26l detection.
`driver_crop.*`: modele giden alanı sürücünün kişi kutusuna (+`pad_ratio`) daraltır
(`redetect_every` = önbellek tazeleme, `min_gain` = ROI zaten darsa kırpmayı atla).
`fuse_detections` + `aux_classes`: Stage-1'in tam karede gördüğü phone/smoking
nesneleri araca düşüyorsa sürücü bayrağına OR'lanır.
`voting.*` (v2.3 — **Katman B `DriverStateEngine`**): per-`track_id` zaman-oylaması;
`window`/`min_votes` (16/8 = mevcut davranış) bir bayrağı kararlı saymak için pencere +
min True oy; `max_age` araç görünmeyince tamponun düşürüleceği kare. Eski per-(track,alan)
StabilityTracker çağrısının ID-merkezli karşılığıdır (aynı davranış + bellek temizliği).

### `speed`
`mode`: `metric` (oto-kalibrasyon) / `tripwire` / `ipm` / `disabled` (yalnızca
`relative_velocity_flag`). `calibration_file` ile kalibrasyon yüklenir.
`swerving.*`: dikkatsiz sürüş / yalpalama tespiti — `window_s` (saniye, fps-bağımsız),
`min_flips`, `amp_ratio` (o anki araç genişliği birimi, ölçek-bağımsız).

### `tracking`
`tracker`, `reid_model`, `min_track_frames` (ağır aşama + ÇIKTI kapısı: track bu kadar
kare yaşamadan OCR/pose çalışmaz VE annotation/event üretmez — tek/iki-kare hayalet
track koruması). `class_vote.*` (track başına **alan-ağırlıklı** araç-sınıfı oylaması:
`enabled`, `decay` = yardımcı unutma, `area_floor` = uzak araç oy tabanı — oy
`conf × bbox_alan/kare_alan` ile ağırlıklanır: yakın/büyük araç sınıfı daha güvenilir,
uzak araç onlarca kare `truck` görünse de yakındaki `car` kanıtı devralır).
Ek: `models.detector.dedup_iou` — aynı araca üretilen kopya kutuları bastırır.

### `qod`
`backend` (`mock`/`camara`), `endpoint`, `profiles` (optimize/quality), `histeresis`
(min_active_seconds + cooldown_seconds — tetikle-bırak salınımını önler),
`approach.*` (yaklaşma tetiği: `window`, `growth`, `min_area_ratio` — şartnamenin
"TOGG yaklaşınca QoD" senaryosu, `reason=vehicle_approach`).

### `risk`
ID-merkezli accumulator risk kuralları. `rules[].all_of` koşulları sağlanınca `RISK_ALERT`.

### `optional_modules`
§8 modülleri (`zero_waste_payload`, `super_resolution`, `homography_ipm`). **Default kapalı**;
kapalıyken import bile yapılmaz (lazy). Detay: `docs/mimari_ek_moduller.md`.

### `dashboard`
`serve` (statik serve), `default_bbox`, `theme` (`dark`/`light`).

## Env override
`.env` (veya kabuk) bazı değerleri override eder: `AURA_PROFILE` (config profili),
`AI_MODE`, `AURA_DEVICE`, `AURA_INFERENCE_PORT`, `AURA_QOD_MOCK_PORT`, `AURA_NV_MOCK_PORT`.
