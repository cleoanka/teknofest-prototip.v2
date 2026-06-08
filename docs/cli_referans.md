# CLI Referansı

> Bu dosya **gerçek çalıştırılmış `--help` çıktılarından** üretilmiştir.
> Yeniden üretmek için: ilgili komutu `--help` ile çalıştırıp çıktıyı yapıştırın.

## İçindekiler
- [bootstrap.py](#bootstrappy) — kurulum
- [python -m aura](#python--m-aura) — inference pipeline
- [python -m aura.eval](#python--m-auraeval) — değerlendirme + QoD A/B
- [python -m train](#python--m-train) — eğitim (alt komutlar)
- [python -m aura.synthetic](#python--m-aurasynthetic) — örnek veri
- [python -m aura.smoke](#python--m-aurasmoke) — smoke test
- [python -m train.roboflow_pull](#python--m-trainroboflow_pull) — Roboflow

## bootstrap.py

```
usage: python bootstrap.py [-h] [--skip-weights] [--skip-node] [--skip-deps]
                           [--force] [--dev]

AURA kurulum bootstrap'i — tek komutla sıfırdan kurulum.

options:
  -h, --help      show this help message and exit
  --skip-weights  Model ağırlığı indirmeyi atla
  --skip-node     Node.js/mobil kurulumunu atla
  --skip-deps     pip kurulumlarını atla (yalnızca yapı/config/ağırlık)
  --force         Mevcut .venv'i sil ve sıfırdan kur
  --dev           Dev bağımlılıklarını da kur (pytest, ruff, black)

örnekler:
  python bootstrap.py                 # tam kurulum
  python bootstrap.py --dev           # dev bağımlılıklarıyla
  python bootstrap.py --skip-weights  # ağırlık indirmeden
  python bootstrap.py --force         # .venv'i sıfırdan kur
```

## python -m aura

```
usage: python -m aura [-h] [--config PATH] [--source SOURCE]
                      [--device {auto,cpu,cuda,mps}] [--no-bbox]
                      [--max-frames MAX_FRAMES]
                      [--log-level {DEBUG,INFO,WARNING}]

AURA inference pipeline — araç, plaka, sürücü durumu ve hız tespiti.

options:
  -h, --help            show this help message and exit
  --config PATH         Config dosyası (varsayılan: config/default.yaml)
  --source SOURCE       Video dosyası, kamera index (0,1,2...) veya RTSP/HTTP
                        URL
  --device {auto,cpu,cuda,mps}
                        İşlem birimi (varsayılan: config'ten / auto)
  --no-bbox             Ham video akışı (annotation overlay olmadan)
  --max-frames MAX_FRAMES
                        En fazla bu kadar kare işle (test/demo için)
  --log-level {DEBUG,INFO,WARNING}
                        Log seviyesi (varsayılan: INFO)

örnekler:
  python -m aura --source 0
  python -m aura --source video.mp4 --device mps
  python -m aura --source rtsp://10.0.0.5:8554/cam --log-level DEBUG
```

## python -m aura.eval

```
usage: python -m aura.eval [-h] [--source SOURCE]
                           [--ground-truth GROUND_TRUTH] [--qod-comparison]
                           [--output OUTPUT] [--config CONFIG]

AURA model değerlendirme — doğruluk metrikleri ve QoD A/B karşılaştırması

options:
  -h, --help            show this help message and exit
  --source SOURCE       Test video dosyası
  --ground-truth GROUND_TRUTH
                        Ground-truth JSON dosyası
  --qod-comparison      QoD açık/kapalı senaryolarını karşılaştır (şartname
                        kanıtı)
  --output OUTPUT       Rapor çıktı dizini
  --config CONFIG       Config dosyası

örnekler:
  python -m aura.eval --source data/samples/ornek.mp4 --ground-truth data/samples/ornek_gt.json
  python -m aura.eval --source test.mp4 --ground-truth gt.json --qod-comparison
```

## python -m train

```
usage: python -m train [-h] {detector,driver-state,dataset} ...

AURA model eğitimi

positional arguments:
  {detector,driver-state,dataset}
    detector            Stage-1 araç tespit modelini eğit (YOLO26s fine-tune)
    driver-state        Stage-2 sürücü durumu modelini eğit (YOLO26l fine-
                        tune)
    dataset             Ham veriyi YOLO formatına dönüştür ve split uygula

options:
  -h, --help            show this help message and exit

örnekler:
  python -m train detector --data data/detector.yaml --epochs 100
  python -m train driver-state --data data/driver.yaml --imgsz 320
  python -m train dataset --input data/raw/ --output data/processed/
```

### python -m train detector

```
usage: python -m train detector [-h] --data DATA [--epochs EPOCHS]
                                [--imgsz IMGSZ] [--batch BATCH]
                                [--weights WEIGHTS]
                                [--device {auto,cpu,cuda,mps}]
                                [--project PROJECT] [--name NAME]

options:
  -h, --help            show this help message and exit
  --data DATA           data.yaml yolu
  --epochs EPOCHS
  --imgsz IMGSZ
  --batch BATCH
  --weights WEIGHTS     Başlangıç ağırlığı
  --device {auto,cpu,cuda,mps}
  --project PROJECT
  --name NAME
```

### python -m train driver-state

```
usage: python -m train driver-state [-h] --data DATA [--epochs EPOCHS]
                                    [--imgsz IMGSZ] [--batch BATCH]
                                    [--weights WEIGHTS]
                                    [--device {auto,cpu,cuda,mps}]
                                    [--project PROJECT] [--name NAME]

options:
  -h, --help            show this help message and exit
  --data DATA           data.yaml yolu
  --epochs EPOCHS
  --imgsz IMGSZ         Cabin ROI küçük → 320 önerilir
  --batch BATCH
  --weights WEIGHTS
  --device {auto,cpu,cuda,mps}
  --project PROJECT
  --name NAME
```

### python -m train dataset

```
usage: python -m train dataset [-h] --input INPUT --output OUTPUT
                               [--train TRAIN] [--val VAL] [--classes CLASSES]
                               [--seed SEED]

options:
  -h, --help         show this help message and exit
  --input INPUT      Ham veri dizini (images/ + labels/)
  --output OUTPUT    Çıktı dizini
  --train TRAIN      Train oranı
  --val VAL          Val oranı (kalan → test)
  --classes CLASSES  Virgülle sınıf listesi (örn. car,truck)
  --seed SEED
```

## python -m aura.synthetic

```
usage: python -m aura.synthetic [-h] [--out OUT] [--frames FRAMES] [--fps FPS]
                                [--width WIDTH] [--height HEIGHT]

Sentetik trafik test videosu + ground-truth üret (deterministik).

options:
  -h, --help       show this help message and exit
  --out OUT        Çıktı dizini
  --frames FRAMES  Kare sayısı (varsayılan 90)
  --fps FPS        FPS (varsayılan 30)
  --width WIDTH    Genişlik
  --height HEIGHT  Yükseklik

örnek:
  python -m aura.synthetic --out data/samples --frames 90
```

## python -m aura.smoke

```
usage: python -m aura.smoke [-h] [--frames FRAMES]

AURA adaptif smoke test (kurulum + pipeline doğrulama).

options:
  -h, --help       show this help message and exit
  --frames FRAMES  İşlenecek kare sayısı
```

## python -m train.roboflow_pull

```
usage: python -m train.roboflow_pull [-h] --workspace WORKSPACE
                                     --project PROJECT [--version VERSION]
                                     [--format FORMAT] [--output OUTPUT]

Roboflow veri seti indirme (ROBOFLOW_API_KEY gerekir).

options:
  -h, --help            show this help message and exit
  --workspace WORKSPACE
  --project PROJECT
  --version VERSION
  --format FORMAT       Etiket formatı (YOLO uyumlu)
  --output OUTPUT
```

