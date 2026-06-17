# Değerlendirme — Metrikler + QoD A/B Protokolü

## Ne yapar
Şartname puanının %80'i doğrudan burada ölçülür: doğruluk metrikleri ve QoD'nin
ölçülebilir başarım katkısı (A/B).

## Metrikler
- **Plaka:** exact-match accuracy, CER (Character Error Rate).
- **Tespit:** kare-bazlı tespit oranı, küçük/uzak nesne tespit oranı.
- **Hız:** MAE/RMSE (kalibrasyon varsa).
- **Sürücü durumu / swerving:** precision/recall/F1 (video-düzeyi; `prf1`/`accuracy`).
- **FPS:** ortalama (referans amaçlı).

## FTR §4 metrik raporu (v2.3) — `--metrics-report`
`tools/test_video.py` özetlerinden **video-düzeyi P/R/F1 + plaka exact-match/CER + araç
doğruluğu + FPS** üretir; **dedektöre göre gruplar** (yolo26l vs v4-finetune A/B). Doğrudan
FTR §4 tablolarına yapıştırılabilir.
```bash
python tools/test_video.py --source ~/video_1.mp4 --json eval_results/ab/video_1_yolo26l.json
python tools/test_video.py --source ~/video_1.mp4 --profile v4-finetune --json eval_results/ab/video_1_v4.json
python -m aura.eval --metrics-report --summaries eval_results/ab   # → eval_results/metrics_report.md+csv+json
```
> Dürüstlük: 3-videoluk küçük held-out set (davranış tespitinin *çalıştığının* kanıtı).
> İstatistiksel mAP için etiketli set + `python -m train` (`model.val` mAP/P/R/F1 export).
> Detay + doldurulabilir taslak: `ftr.md` §4.

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
