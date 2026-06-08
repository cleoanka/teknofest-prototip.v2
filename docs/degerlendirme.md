# Değerlendirme — Metrikler + QoD A/B Protokolü

## Ne yapar
Şartname puanının %80'i doğrudan burada ölçülür: doğruluk metrikleri ve QoD'nin
ölçülebilir başarım katkısı (A/B).

## Metrikler
- **Plaka:** exact-match accuracy, CER (Character Error Rate).
- **Tespit:** kare-bazlı tespit oranı, küçük/uzak nesne tespit oranı.
- **Hız:** MAE/RMSE (kalibrasyon varsa).
- **Sürücü durumu:** precision/recall/F1 (sınıf bazında — genişletilebilir).
- **FPS:** ortalama (referans amaçlı).

## QoD A/B harness (kritik)
Aynı video iki senaryoda koşulur:
1. **QoD OFF** — düşük çözünürlük (düşük bant simülasyonu): küçük plaka ROI'leri
   `min_pixel_height` altına düşer, küçük/uzak araçlar kaçar.
2. **QoD ON** — tam çözünürlük (HIGH_THROUGHPUT benzeri).

Her senaryo için tam metrik seti; çıktı **delta tablosu** (mutlak + yüzde fark).

```bash
.venv/bin/python -m aura.eval --source data/samples/ornek.mp4 \
  --ground-truth data/samples/ornek_gt.json --qod-comparison
```
Çıktı: `eval_results/report.md` + `report.json`; ayrıca `GET /eval/results` ve dashboard
Chart.js paneli.

### Örnek sonuç (sentetik video)
| Metrik | QoD OFF | QoD ON | Δ |
|---|---|---|---|
| Plaka doğruluğu (%) | 33.3 | 66.7 | +33.4 |
| Küçük nesne tespiti (%) | 46.8 | 98.2 | +51.4 |
| Tespit oranı (%) | 74.5 | 100.0 | +25.5 |

> QoD yalnızca kritik anda devreye girerek küçük/uzak plaka ROI'lerinin yeterli pikselle
> okunmasını sağlar; ölçülen pozitif delta bunun kanıtıdır (şartname %40).

## Rapor yorumlama
- **Plaka doğruluğu** ↑: QoD'nin kalite tetiği OCR'ı kurtardı.
- **Küçük nesne** ↑: yüksek çözünürlük uzak araçları yakaladı.
- Gerçek model/veriyle mutlak değerler değişir; **delta'nın pozitifliği** QoD katkısının kanıtıdır.

## Sorun Giderme
| Belirti | Çözüm |
|---|---|
| Delta ≈ 0 | Ground-truth doğru mu? Senaryo yeterince zorlayıcı mı (küçük nesneler)? |
| `no_results` | Önce `POST /eval/run` / `--qod-comparison` |
| GT eşleşmiyor | `ornek_gt.json` videoyla aynı çözünürlük/sahne mi? |
