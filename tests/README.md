# `tests/` — Test Paketi

Model gerektirmeyen **unit testler** (mock modda çalışır, CI-uyumlu) + ağırlık gerektiren
**integration testler** (`@pytest.mark.integration`, CI'da skip).

## Çalıştırma
```bash
pytest                      # tümü (integration'lar ağırlık yoksa otomatik skip)
pytest -m "not integration" # yalnızca unit (CI'nın yaptığı)
pytest -m integration       # gerçek model (ultralytics + ağırlık gerekir)
```

## Kapsam
| Dosya | Test eder |
|---|---|
| `test_contracts.py` | §6.0 pydantic sözleşmeleri + accumulator + risk kuralları |
| `test_detection.py` | Mock dedektör + IoU takipçi + ROI geometri |
| `test_stability.py` | 16/8 state machine (7/16→ret, 8/16→kabul, flicker) |
| `test_driver_state.py` | Sürücü-durum mock (renk→durum eşlemesi) |
| `test_plate.py` | Voting buffer, sweet-spot gating, regex, ret→QoD |
| `test_qod.py` | QoD histerezisi (tetik/bırak, cooldown) |
| `test_speed.py` | disabled relative flag + tripwire |
| `test_events.py` | AuraEvent/AnnotationFrame şema + emitter + WS push |
| `test_api_contracts.py` | inference_api + qod_mock + nv_mock (TestClient) |
| `test_eval.py` | QoD A/B harness + metrikler (CER, plaka doğruluğu) |
| `test_train.py` | Dataset split + data.yaml |
| `test_optional.py` | §8 lazy import (kapalıyken import yok) + işlevsellik |
| `test_integration.py` | Gerçek YOLO26 (skip'lenebilir) |

`conftest.py`: `cfg` fixture (`config/default.yaml`) + repo kökünü `sys.path`'e ekler.
