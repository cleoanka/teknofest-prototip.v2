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

### Ölçülen sonuçlar (17 Haz 2026; dewarp/enhance OFF; `eval_results/`)
- **Stok dedektör (`yolo26l`) held-out mAP** (COCO val2017, 5000 görsel,
  `map_yolo26l.json`): mAP50-95 **0.537**, mAP50 **0.709**, P **0.740**, R **0.641**.
- **v4 fine-tune** (yolov8m, 11 sınıf): held-out **mAP50 0.788** (model kartı). DİKKAT:
  `license_plate/cigarette/seatbelt/headphone` için eğitim verisi yoktu → o sınıflar güvenilmez.
- **Davranış (3 gerçek video, video-düzeyi):** v4-finetune makro-F1 **1.0**; yolo26l makro-F1
  **0.933** (yolo26l video_2'de 1 `swerving` yanlış-pozitif). phone/smoking/swerving P=R=F1=1.0 (v4).
- **Plaka (dürüst çerçeve):** **ÖNERİLEN `--profile v4-finetune`** → 2/3 exact-match (66.7%),
  CER **0.083**, **0 yanlış-onay** (video_3 dürüst `pending`/partial). Stok `yolo26l` → 1/3
  exact ama **2 yanlış-onay** (`04TC8532`, `24IC8532`) → plaka-kritik değildir; ölçümle
  kanıtlanmış dedektör-kalitesi sınırı (voting eşiğiyle overfit olmadan düzeltilemez).
- **Araç sınıfı doğruluğu:** %100 (her iki dedektör). **FPS (MPS, M4 Pro):** ~5.0 (v4) /
  ~5.9 (yolo26l) — MPS alt-sınırı; CUDA sunucuda belirgin daha yüksektir.
- **Eğitim hattı doğrulaması:** açık `coco128` setinde uçtan uca gerçek `best.pt` + val mAP
  üretildi (mAP50 0.790, mAP50-95 0.619) → boru hattı çalışır; komite verisiyle tek komut.

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

### Ölçülen sonuç (sentetik kontrollü örnek — kare-düzeyi GT, 17 Haz 2026)
Gerçek videolarda kare-düzeyi ground-truth bulunmadığından QoD A/B bu **kontrollü sentetik
sette** ölçülür (dürüst not). Kaynak: `eval_results/report.md`.

| Metrik | QoD OFF | QoD ON | Δ |
|---|---|---|---|
| Plaka doğruluğu (%) | 66.7 | 100.0 | **+33.3** |
| Küçük nesne tespiti (%) | 41.4 | 92.8 | **+51.4** |
| Tespit oranı (%) | 71.8 | 97.3 | **+25.5** |

> QoD yalnızca kritik anda devreye girerek küçük/uzak plaka ROI'lerinin yeterli pikselle
> okunmasını sağlar; ölçülen pozitif delta bunun kanıtıdır (şartname %40). Delta'nın
> pozitifliği QoD katkısının kanıtıdır; mutlak değerler gerçek model/veriyle değişir.

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
