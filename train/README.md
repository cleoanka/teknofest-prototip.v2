# `train/` — Model Eğitimi

YOLO26 fine-tune pipeline'ları. `--help` torch gerektirmez (ultralytics lazy import).

## Komutlar
```bash
python -m train --help
python -m train dataset --input data/raw/ --output data/processed/ --classes car,truck,bus,minibus
python -m train detector --data data/processed/data.yaml --epochs 100 --imgsz 640
python -m train driver-state --data data/processed_driver/data.yaml --imgsz 320
python -m train.roboflow_pull --workspace W --project P --version 1   # ROBOFLOW_API_KEY
```

## Dosyalar
| Dosya | Sorumluluk |
|---|---|
| `__main__.py` | `python -m train` (detector / driver-state / dataset subcommand) |
| `train_detector.py` | YOLO26s fine-tune → `weights/custom_detector.pt` |
| `train_driver_state.py` | YOLO26l fine-tune (320px) → `weights/custom_driver.pt` |
| `prepare_dataset.py` | train/val/test split + `data.yaml` (torch gerektirmez) |
| `roboflow_pull.py` | Roboflow veri çekme (opsiyonel) |
| `utils.py` | cihaz çözümleme + best ağırlık swap |
| `configs/` | `detector.yaml`, `driver_state.yaml` data.yaml örnekleri |

## Custom ağırlık swap
Eğitim sonrası `weights/custom_*.pt` üretilir. `config/default.yaml` →
`models.detector.path` / `models.driver_state.path` değerini güncelleyin; inference
yeniden başladığında yeni ağırlık yüklenir.

Detaylar: `docs/egitim.md` · veri toplama: `docs/veri_seti.md`.
