# `config/` — Merkezi Yapılandırma

## Ne yapar
`default.yaml` AURA'nın **tek doğruluk kaynağıdır**. Hiçbir eşik/flag koda gömülmez;
tüm çalışma zamanı davranışı buradan yönetilir. `aura.config.load_config()` bu dosyayı
yükler, seçili env değişkenlerini override olarak uygular ve noktalı erişim sağlar:

```python
from aura.config import load_config
cfg = load_config()
cfg.get("plate.voting_buffer_size")   # 7
cfg.get("stability.window")           # 16
```

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
`lp_detector.*` (sıkı plaka kırpma — özel YOLOv11n plaka modeli; yoksa loglu fallback),
`lp_vote_min_px` / `lp_qod_below_px` (boyut-farkında kanıt: çok küçük LP oylamaya
girmez; küçük LP görüldüğü an `plate_too_small` QoD tetiği — consensus_fail beklemeden),
`voting.*` (kalıcı, güven-ağırlıklı oy havuzu: `min_weight`, `margin_weight`,
`fix1/fix2_weight`, `substring_weight`; `size_full_px`/`size_floor`/`no_lp_weight` =
okuma ağırlığı OCR güveni × kaynak kalitesi; `char_consensus`/`char_margin` =
pozisyon-hizalı karakter füzyonu (en güçlü tek okuma çapa; ikinciyi `char_margin`
ile geçmeli), T↔I / 3↔0 misread'ini çoğunlukla düzeltir —
bkz. `aura/plate/normalize.py`).

### `models.driver_state` (backend seçimi)
`backend: auto|pose|yolo` — pose: YOLO26-pose keypoint geometrisi (`pose_path` =
varsayılan `yolo26l-pose.pt`, yoksa s-pose'a loglu fallback; `pose_conf`,
`pose_kp_conf`, `phone_ear_ratio`, `smoke_mouth_ratio`, `roi_min_side`, `roi_enhance`) +
hibrit ROI nesne kanıtı (`roi_objects.*`); yolo: fine-tune YOLO26l detection.
`driver_crop.*`: modele giden alanı sürücünün kişi kutusuna (+`pad_ratio`) daraltır
(`redetect_every` = önbellek tazeleme, `min_gain` = ROI zaten darsa kırpmayı atla).
`fuse_detections` + `aux_classes`: Stage-1'in tam karede gördüğü phone/smoking
nesneleri araca düşüyorsa sürücü bayrağına OR'lanır.

### `speed`
`mode`: `metric` (oto-kalibrasyon) / `tripwire` / `ipm` / `disabled` (yalnızca
`relative_velocity_flag`). `calibration_file` ile kalibrasyon yüklenir.
`swerving.*`: dikkatsiz sürüş / yalpalama tespiti — `window_s` (saniye, fps-bağımsız),
`min_flips`, `amp_ratio` (o anki araç genişliği birimi, ölçek-bağımsız).

### `tracking`
`tracker`, `reid_model`, `min_track_frames` (ağır aşama + ÇIKTI kapısı: track bu kadar
kare yaşamadan OCR/pose çalışmaz VE annotation/event üretmez — tek/iki-kare hayalet
track koruması). `class_vote.*` (track başına araç-sınıfı oylaması: `enabled`, `decay` —
tek-kare `car↔truck` titremesini güven-ağırlıklı çoğunlukla düzeltir).
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
`.env` (veya kabuk) bazı değerleri override eder: `AI_MODE`, `AURA_DEVICE`,
`AURA_INFERENCE_PORT`, `AURA_QOD_MOCK_PORT`, `AURA_NV_MOCK_PORT`.
