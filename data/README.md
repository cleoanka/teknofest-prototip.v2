# `data/` — Veri

## `samples/`
Bootstrap tarafından üretilen **deterministik sentetik trafik senaryosu**:
- `ornek.mp4` — 90 kare, 3 araç, farklı şeritler/zamanlar, plaka varyasyonları (`.gitignore`'lu, yeniden üretilebilir).
- `ornek_gt.json` — kare-bazlı ground-truth (bbox, vehicle_class, plaka, sürücü durumu, hız).

Yeniden üret:
```bash
python -m aura.synthetic --out data/samples --frames 90
```

GT yapısı (`ornek_gt.json`):
```json
{ "video": "ornek.mp4", "fps": 30, "width": 640, "height": 360,
  "frames": [ { "frame": 0, "objects": [
    { "id": 1, "bbox": [x1,y1,x2,y2], "vehicle_class": "car",
      "plate": "34ABC123", "driver": {"phone": true, ...}, "speed_kmh": 40.0 } ] } ] }
```

Gerçek TOGG veri seti geldiğinde `data/samples/` üzerine yazılır; `data/raw/`
(ham, etiketsiz veri) `.gitignore`'ludur.

## `raw/`
Ham/etiketlenmemiş eğitim verisi (git'e dahil edilmez). Eğitim akışı: `docs/veri_seti.md`.
