# Model Ağırlıkları

Bu dizin `bootstrap.py` tarafından doldurulur ve `.gitignore`'ludur.

- **Tespit edilen torch backend:** `mps`
- **Son kurulum platformu:** Darwin arm64

## Ağırlıklar

| Dosya | Durum | SHA256 (ilk 16) | Kaynak |
|---|---|---|---|
| `yolo26s.pt` | missing | `646f8bc3fe0a6568` | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt |
| `yolo26l.pt` | missing | `9fe3c544f2b19beb` | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt |

## Trust-on-first-use

Şartname için resmi SHA256 yayımlandığında `bootstrap.py` içindeki `WEIGHTS` sözlüğüne yazın; bozuk indirmeler otomatik yeniden indirilir. İlk indirmede hesaplanan hash `weights.lock.json`'a yazılır ve sonraki çalıştırmalarda doğrulanır.

## Custom ağırlık swap

Fine-tune sonrası `weights/custom_detector.pt` üretip `config/default.yaml` →
`models.detector.path` değerini güncelleyin. Inference yeniden başladığında yeni
ağırlık yüklenir. Detay: `docs/egitim.md`.
