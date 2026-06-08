# `aura/` — YZ Çekirdeği

AURA'nın **gerçek** yapay zekâ / bilgisayarlı görü çekirdeği. Upstream (kamera) ve
downstream'i (dashboard/mobil) bilmez; yalnızca event + annotation stream yayar
(decoupled mikroservis prensibi).

## Pipeline akışı
```
preprocessing → detection(+ByteTrack) → ROI crop ──┬─ driver_state (16/8 ile kararlı)
                                                    └─ plate (sweet spot+voting+OCR)
                                          → speed → accumulator → events + annotations
```

## Modüller
| Paket | Sorumluluk | Milestone |
|---|---|---|
| `preprocessing/` | Far/blur/yansıma/occlusion ön-işleme | M-sonrası |
| `detection/` | YOLO26s + ByteTrack + ROI crop | M3 |
| `stability/` | 16/8 state machine (flicker koruması) | M4 |
| `driver_state/` | YOLO26l: phone/smoking/no_seatbelt/fatigue (no-landmark) | M4 |
| `plate/` | Sweet spot + voting buffer + OCR + Türk plaka regex | M5 |
| `speed/` | tripwire / ipm / disabled (relative_velocity_flag) | M6 |
| `accumulator/` | ID-merkezli TrackRecord + risk kuralları | M3+ |
| `qod/` | CAMARA QoD istemcisi + histerezis | M5+ |
| `events/` | AuraEvent / AnnotationFrame emitter | M7 |
| `pipeline/` | Orkestratör | M2+ |
| `optional/` | §8 modüller (lazy, default kapalı) | M12 |
| `eval/` | Metrikler + QoD A/B harness | M9 |

## Yardımcı modüller
| Modül | Açıklama |
|---|---|
| `config.py` | `load_config()` — `config/default.yaml` yükleyici (noktalı erişim) |
| `schema.py` | Pydantic v2 sözleşmeleri (TrackRecord, AuraEvent, …) |
| `synthetic.py` | `python -m aura.synthetic` — sentetik örnek video + GT |
| `smoke.py` | `python -m aura.smoke` — adaptif kurulum/pipeline smoke testi |

## Çalıştırma
```bash
python -m aura --source data/samples/ornek.mp4 --device auto
python -m aura --help
```

`ai_mode: auto` (config) → ultralytics + ağırlık varsa gerçek YOLO; yoksa deterministik
numpy mock dedektör (model olmadan tüm hat ve testler uçtan uca çalışır).
