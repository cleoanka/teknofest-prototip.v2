# Model Ağırlıkları

> Bu README, `bootstrap.py` ilk çalıştığında **kurulum durumu + SHA256** ile yeniden yazılır.
> Aşağıdaki, bootstrap öncesi başlangıç içeriğidir.

Bu dizin `bootstrap.py` tarafından doldurulur ve `.gitignore`'ludur (`*.pt` commit edilmez).

## Ağırlıklar
| Dosya | Rol |
|---|---|
| `yolo26s.pt` | Stage-1 araç tespiti (YOLO26s) |
| `yolo26l.pt` | Stage-2 sürücü durumu (YOLO26l, base — fine-tune bekleniyor) |

## Otomatik indirme + doğrulama
`bootstrap.py` her ağırlık için: dosya yoksa indirir, varsa SHA256 doğrular, bozuksa
yeniden indirir (idempotent). İlk indirmede hesaplanan hash `weights.lock.json`'a yazılır
(**trust-on-first-use**) ve sonraki çalıştırmalarda bütünlük buna karşı doğrulanır.

İndirme başarısız olursa (404/ağ) kurulum durmaz; pipeline ağırlık olmadan **mock** modda
çalışır. Ağırlığı manuel yerleştirmek için ilgili `.pt` dosyasını bu dizine kopyalayın.

## Custom ağırlık swap
Fine-tune sonrası `weights/custom_detector.pt` üretip `config/default.yaml` →
`models.detector.path` değerini güncelleyin. Inference yeniden başladığında yeni ağırlık
yüklenir. Detay: `docs/egitim.md`.
