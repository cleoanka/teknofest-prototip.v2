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
`sweet_spot` (normalize ROI), `voting_buffer_size`, `consensus_ratio`, `ocr_lang`,
`regex` (Türk plaka), `min_pixel_height` (altında kalite tetiklenir).

### `speed`
`mode`: `tripwire` (sabit kamera+mesafe) / `ipm` (homography) / `disabled` (yalnızca
`relative_velocity_flag`). `calibration_file` ile kalibrasyon yüklenir.

### `qod`
`backend` (`mock`/`camara`), `endpoint`, `profiles` (optimize/quality), `histeresis`
(min_active_seconds + cooldown_seconds — tetikle-bırak salınımını önler).

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
