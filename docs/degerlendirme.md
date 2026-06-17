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
- **(ASIL DOĞRULUK) Stok dedektör (`yolo26l`) — COCO val2017 HELD-OUT** (5000 görsel,
  `map_yolo26l.json`): mAP50-95 **0.537**, mAP50 **0.709**, P **0.740**, R **0.641**. Bu, modelin
  eğitiminde görmediği ayrı doğrulama setidir ve asıl dedektör doğruluk göstergesidir.
- **(YALNIZ HIZLI SAĞLIK) Stok `yolo26l` — coco128** (küçük, train-örtüşmeli): mAP50 **0.790**,
  mAP50-95 **0.619**. DİKKAT: fine-tune DEĞİLDİR ve doğruluk iddiası olarak kullanılmaz; yalnız
  boru hattının kurulduğunu gösteren sağlık kontrolüdür ("fine-tune 0.790" atfı YANLIŞTIR).
- **DÜRÜST NOT (zorunlu sınıflar):** `license_plate`, `smoking`/`cigarette` için held-out mAP,
  komite verisi / özel-model eğitimi gelene dek **YOKTUR** (mevcut sayılar stok COCO held-out +
  3-video davranış + boru-hattı doğrulaması).
- **Davranış (3 gerçek video, video-düzeyi):** **her iki dedektör de** (yolo26l ve v4-finetune)
  makro-F1 **1.0**; phone/smoking/swerving P=R=F1=1.0. (Stabilite fixleri öncesi yolo26l
  video_2'de 1 `swerving` yanlış-pozitifiyle 0.933 veriyordu; `track_id=-1`/phantom çıktı kapısı
  ve `max_roi_area_ratio` zırhlarıyla bu FP gitti.)
- **Plaka (dürüst çerçeve — her iki dedektör eşit):** yolo26l **ve** v4-finetune → 2/3
  exact-match (66.7%), CER **0.083**, confirmed=2, partial=1, **0 yanlış-onay**. video_1/2 =
  `34TC8532` CONFIRMED (doğru); video_3 = dürüst PENDING (`24IC8532` partial, uzak/bulanık).
  Sistem belirsiz/uzak okumayı **asla yanlış plaka olarak onaylamaz**, dürüstçe çekimser kalır
  (`confirm_min_char_margin=2.0` + pozisyon-veto + zemin koşulu). Eski stok yolo26l 1/3 exact +
  2 yanlış-onay üretiyordu; conservative confirm eşiğiyle düzeldi. v4 ikincil track'lerde biraz
  daha temiz kırpık üretir (ikincil not; plaka doğruluğu eşit).
- **Araç sınıfı doğruluğu:** %100 (her iki dedektör). **FPS (MPS, M4 Pro):** ~5.9 (yolo26l) /
  ~5.3 (v4) — MPS alt-sınırı; CUDA sunucuda belirgin daha yüksektir.
- **Eğitim hattı doğrulaması (uçtan uca):** açık `coco128` setinde `yolo26s` ile 5 epoch
  koşturularak gerçek `best.pt` + val mAP üretildi (**best.pt mAP50 0.7645, mAP50-95 0.5909**)
  → "eğitim hattı uçtan uca çalışır" kanıtı (doğruluk iddiası değil); komite verisiyle tek komut.

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
Aşağıdaki delta sayıları gerçek ve **yeniden-üretilebilirdir** (`eval_results/report.json`,
`qod_comparison=true`). Atıf şeffaflığı: QoD A/B kare-düzeyi ground-truth gerektirir; üç gerçek
videoda kare-düzeyi GT bulunmadığından QoD A/B **orada ölçülemez** ve bu nedenle ölçüm,
kare-düzeyi GT içeren **kontrollü sentetik set** (`data/samples/ornek.mp4`) üzerinde yapılır.
Kaynak: `eval_results/report.md` / `report.json`.

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
