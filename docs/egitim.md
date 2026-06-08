# Eğitim Akışı

## Ne yapar
AURA'nın iki YZ modelini (Stage-1 araç tespiti = YOLO26s, Stage-2 sürücü durumu =
YOLO26l) kendi veri setinizle fine-tune eder ve çıktı ağırlığı inference'a swap'lar.

## Gereksinimler
- `./setup.sh` ile kurulmuş ortam (ultralytics + torch).
- YOLO formatında etiketli veri (bkz. `docs/veri_seti.md`).
- Tercihen GPU (CUDA) veya Apple Silicon (MPS); CPU yavaş ama çalışır.

## Akış

### 1. Veri hazırlama (split + data.yaml)
```bash
python -m train dataset \
  --input data/raw/ --output data/processed/ \
  --classes car,truck,bus,minibus --train 0.8 --val 0.1
```
`data/processed/{train,val,test}/{images,labels}` + `data.yaml` üretir.

### 2. Stage-1 araç dedektörü (YOLO26s)
```bash
python -m train detector --data data/processed/data.yaml \
  --epochs 100 --imgsz 640 --batch 16 --device auto
```
Çıktı: `weights/custom_detector.pt`.

### 3. Stage-2 sürücü durumu (YOLO26l, 320px)
```bash
python -m train driver-state --data data/processed_driver/data.yaml \
  --epochs 100 --imgsz 320 --device auto
```
Çıktı: `weights/custom_driver.pt`.
> Yorgunluk bir **detection sınıfı** olarak öğrenilir (kapalı göz/esneme/baş düşmesi).
> MediaPipe/landmark **kullanılmaz** (mimari kararı — bkz. `docs/mimari.md` §9).

### 4. Inference'a swap
`config/default.yaml`:
```yaml
models:
  detector: { path: weights/custom_detector.pt }
  driver_state: { path: weights/custom_driver.pt }
```
`./run.sh` yeniden başlatın; yeni ağırlıklar yüklenir (`ai_mode: real`).

## Hyperparameter rehberi
| Parametre | Öneri | Not |
|---|---|---|
| `epochs` | 100–300 | Erken durdurma ultralytics'te otomatik |
| `imgsz` | detector 640, driver 320 | Cabin ROI küçük → 320 yeterli |
| `batch` | GPU belleğine göre 8–32 | OOM'da düşürün |
| augmentation | mozaik+flip+hsv+karartma | gece koşulları için karartma kritik |

## Örnekler
```bash
# Hızlı deneme (az epoch, CPU)
python -m train detector --data data/processed/data.yaml --epochs 5 --device cpu
```

## Sorun Giderme
- **`ultralytics yok`**: `./setup.sh` çalıştırın veya `pip install -e ".[dev]"`.
- **CUDA OOM**: `--batch` düşürün veya `--imgsz` küçültün.
- **mAP düşük**: veri dengesizliği/az örnek — `docs/veri_seti.md` augmentasyon stratejisine bakın.
- **best.pt bulunamadı**: eğitim erken kesilmiş; `runs/<project>/<name>/weights/` kontrol edin.
