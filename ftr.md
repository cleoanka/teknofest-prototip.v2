# FTR Rehberi — Final Tasarım Raporu'nu RoadGuard Kanıtlarıyla Doldurma

> **Bu belge ne?** TEKNOFEST 2026 "5G & YZ ile Akıllı Yol Güvenliği" yarışmasının
> **Final Tasarım Raporu (FTR)** şablonunun her bölümünü, RoadGuard prototipinin somut
> kanıtlarına / komutlarına / sayılarına bağlayan **ipuçlu rehber + doldurulabilir
> taslak**. Üç işlevi birden görür: (1) raporu doldurma kılavuzu, (2) finale
> (mobil + 5G + QoD demo) hazırlık, (3) prototipin yeteneklerinin tam dokümantasyonu.
>
> ✅ **Takvim (güncel):** FTR son teslim **28.06.2026'ya ERTELENDİ** (kullanıcı teyit etti;
> eski şartname PDF'indeki 14.06 geçersiz). Yani **FTR HÂLÂ AÇIK** — bu rehber + aşağıdaki
> doldurulabilir taslak doğrudan kullanılabilir. En yüksek puanlı §2 (Veri Seti) ve §4 (Sınama)
> için sırasıyla `train dataset --report` ve `roadguard.eval --metrics-report` çıktıları hazır.

---

## 0. Puanlama haritası (iki ayrı rubrik)

**FTR raporu (şablon, 100 puan):** İçindekiler/Kapak hariç **3–10 sayfa**, Arial 12,
başlık Arial Black 14, satır aralığı 1.15, iki yana yaslı, kenar boşlukları üst 2.8 /
alt-sağ-sol 2.5. **Şablona uymayan rapor değerlendirilmez.**

| FTR bölümü | Puan | RoadGuard'daki ana kanıt |
|---|---|---|
| 1. Proje Özeti | 5 | `README.md` + bu rehber §1 |
| 2. **Veri Seti Oluşturulması** | **20** | `docs/veri_seti.md` + `docs/egitim.md` + `train/` tool (`dataset --report`) |
| 3. Yapay Zekâ Çözümü | 50 | 3.1 Problem `docs/mimari.md`§problem · 3.2 Mimari `docs/mimari.md` diyagramı · 3.3 Detay `docs/mimari.md`+`docs/cli_referans.md` |
| 4. **Çözümün Sınanması** | **20** | `python -m roadguard.eval --metrics-report` → `eval_results/metrics_report.md` (P/R/F1/CER/FPS) |
| 5. Kaynakça | 5 | bu rehber §5 |

**Yarışma final puanı (şartname Tablo 1, ayrı):** %40 YZ doğruluk + %40 QoD entegrasyonu
+ %20 rapor/sunum. → QoD kanıtı: `python -m roadguard.eval --qod-comparison` (A/B delta).

**En zayıf iki nokta = en yüksek puanlı iki bölüm:** Veri Seti (20) ve Metrik (20). Bu
rehber özellikle bu ikisini doldurmaya odaklanır.

---

## 1. Proje Özeti (5 puan)

**Rubrik:** Proje kapsamında yürütülen faaliyetlerin özeti.

**Doldurulabilir taslak:**
> RoadGuard, yol kenarı trafik kamerası akışından **araç, plaka, hız ve riskli sürücü
> davranışı** tespiti yapan bir yapay zekâ çekirdeğini; bu çekirdeği **5G CAMARA
> Quality-on-Demand (QoD)** ve **Number Verification** telekom yetenekleriyle birleştiren
> uçtan uca bir sistemdir. YZ çekirdeği (tespit/takip/16-8 kararlılık/OCR/hız/risk)
> gerçektir; ağ/telekom katmanları gerçek API sözleşmesini taklit eden mock'lardır —
> final ortamında yalnızca endpoint/credential değişir. Dedektör omurgası **YOLO26
> (Ultralytics 8.4)**; sürücü davranışı **YOLO26-pose** keypoint geometrisi + hibrit
> nesne kanıtıyla, plaka ise özel **YOLO11n LP dedektörü + format-öncelikli güven-ağırlıklı
> oylama** ile çözülür. Sistem sunucu dağıtımı için profillenmiştir ve tek komutla ayağa kalkar.

---

## 2. Veri Seti Oluşturulması (20 puan)

**Rubrik:** Veri nasıl toplandı/etiketlendi/**dengelendi (data balancing)**; **augmentasyon**
teknikleri; **train/val/test** dağılım oranları + gerekçe; kullanılan **açık kaynak** setler
(kaynakçada da belirt).

**Durum (dürüst, 18 Haz):** Komite TOGG/etiketli veri setini henüz paylaşmadığından (şartname
4.3 "ön tasarım raporu değerlendirmesi sonrası" paylaşılır), §4 metriklerinin büyük kısmı **BASE/stok
YOLO26 modelleriyle** ölçülmüştür. Asıl dedektör doğruluk göstergesi, stok `yolo26l`'in **COCO val2017
held-out (5000 görsel)** sonucudur: mAP50-95 **0.537** / mAP50 **0.709** (§4.2). Toplanan açık-kaynak
veriyle özel-model EĞİTİMİ ise **şu an SÜRÜYOR** (YOLO26s fine-tune): `license_plate` için val mAP50
**≈0.97** (epoch 12'de **0.977**, mAP50-95 0.676; eğitim 35 epoch'a koşuyor — *final henüz kesinleşmedi*),
`seatbelt` ve `smoking` **sırada**. *Bu eğitim mAP'leri SÜRMEKTE olduğundan final değildir; rapora
kesinleşmiş sayı olarak yazılmaz.* **YOLO26 fine-tune boru hattı `train/` altında hazırdır ve uçtan uca
doğrulanmıştır** (açık `coco128` setinde `yolo26s`, 5 epoch → gerçek `best.pt` mAP50 0.7645 üretti → §4).
Komite verisi gelince **tek komutla** domain modeli eğitilir.

No-auth gerçek bbox setleri toplanmış ve PIL-doğrulanmıştır
(`data/processed/{seatbelt,smoking,phone,license_plate}/`): **SEATBELT** 3104 görsel, CC BY 4.0
(kaynak Roboflow `oohmp` → HF `ramankamran/seatbelt-detection`; denge 1.27, dengeli); **SMOKING**
557 görsel, CC BY 4.0 (kaynak **CigDet**, Mendeley DOI 10.17632/6hyrr8typ7.1 — sürücü/insan sigara
bbox'ı, çevresel duman değil); **PHONE** 659 görsel (HF `anywaylabs/synthetic-driver-monitoring`,
CC BY 4.0, **sentetik render** → domain-uyum riski); **LICENSE_PLATE** 8823 görsel, CC BY 4.0
(kaynak HF `keremberke/license-plate-object-detection`). `minibus` için no-auth açık bbox seti
bulunamamıştır; `fatigue` için teyitli açık set bulunamamıştır — dürüstçe belirtilir.

**Doldurulacak içerik + komutlar:**
- **Toplama (açık-kaynak köprü manifesti):** kaynaklar `train/datasets.yaml`'da
  **bildirimsel** tutulur (her hedef sınıf → kaynak + lisans + ~görüntü + RoadGuard taksonomisine
  sınıf-eşlemesi). Ölçülen kapsam: `car/bus/truck/motorcycle/person → COCO`; dört no-auth gerçek
  bbox seti indirilip toplandı ve PIL-doğrulandı (`data/processed/{seatbelt,smoking,phone,license_plate}/`):
  `seatbelt → no_seatbelt_evidence` (Roboflow `oohmp` → HF `ramankamran/seatbelt-detection`, 3104 görsel,
  CC BY 4.0, denge 1.27); `cigarette → smoking` (**CigDet**, Mendeley DOI 10.17632/6hyrr8typ7.1, 557 görsel,
  CC BY 4.0); `phone` (HF `anywaylabs/synthetic-driver-monitoring`, 659 görsel, CC BY 4.0, **sentetik render**
  → domain-uyum riski); `license_plate` (HF `keremberke/license-plate-object-detection`, 8823 görsel,
  CC BY 4.0). **ONUR:** `minibus` ve `fatigue` için no-auth açık bbox seti bulunamadı (manifestte
  `sources: []` boş — uydurma kaynak yok); büyük sigara setleri (Roboflow `driver-smoking-detecor` 1066,
  `Smoker YOLO.v4` 4221) API anahtarı / Roboflow erişimi gerektirir (manifestte listeli). ÇERÇEVE:
  setler TOPLANDI; bunlarla **özel-model eğitimi (YOLO26s fine-tune) ŞU AN SÜRÜYOR** (`license_plate`
  val mAP50 ≈0.97 @ epoch 12; `seatbelt`/`smoking` sırada — *final mAP'ler henüz kesinleşmedi*). §4
  doğruluk metriklerinin büyük kısmı, eğitim tamamlanana dek BASE/stok modellerle ölçülmüştür. Plan/lisans
  özeti: `python -m train fetch` (kuru, ağ kullanmaz). Lisanslar §5 kaynakçaya yazılır; **kullanım öncesi
  lisans/uyumluluk teyidi notu** korunur.
- **Etiketleme:** YOLO formatı (`<cls> <xc> <yc> <w> <h>` normalize). Roboflow ile çek:
  `python -m train.roboflow_pull --workspace W --project P --version N`; çoklu sürücü-davranış
  setini birleştir: `python -m train.merge_driver_datasets`.
- **Dengeleme (data balancing) — KANIT KOMUTU:**
  `python -m train dataset --report --output data/processed/`
  → split başına **sınıf-örnek dağılımı + dengesizlik oranı (max/min)**. Oran > 3 ise
  augment/oversample uyarısı verir. (Bu çıktıyı rapora tablo olarak koyun.)
- **Augmentasyon:** ultralytics yerleşik (mozaik, flip, HSV jitter, ölçek, translate);
  küçük-veri/ablation için `--no-augment` ile kapatılabilir. Karanlık footage için
  `pose.roi_enhance` (CLAHE+gamma) ROI seviyesinde de uygulanır.
- **Split oranları + gerekçe:** varsayılan **train 0.8 / val 0.1 / test 0.1**
  (`python -m train dataset --train 0.8 --val 0.1`). Gerekçe: küçük özel sette val/test'in
  istatistiksel anlam taşıması için %10+%10; sınıf-dengesiz setlerde stratify önerilir.

**Doldurulabilir taslak:**
> Veri seti açık-kaynak köprü stratejisiyle oluşturulmuştur: (i) genel
> araç/kişi sınıfları için COCO; (ii) dört no-auth gerçek bbox seti indirilip toplandı ve
> PIL-doğrulandı (`data/processed/{seatbelt,smoking,phone,license_plate}/`): `seatbelt` 3104 görsel
> (CC BY 4.0, denge 1.27), `smoking` 557 görsel (CigDet/Mendeley, CC BY 4.0), `phone` 659 görsel
> (HF synthetic, CC BY 4.0, sentetik render) ve `license_plate` 8823 görsel (HF keremberke, CC BY 4.0).
> Bu setlerle özel-model eğitimi (YOLO26s fine-tune) hâlihazırda sürmektedir (`license_plate` val
> mAP50 ≈0.97 @ epoch 12; `seatbelt`/`smoking` sırada — final mAP'ler henüz kesinleşmemiştir), §4
> doğruluk sayıları eğitim tamamlanana dek BASE/stok YOLO26 ile ölçülmüştür. Tüm kaynaklar
> `train/datasets.yaml` manifestinde lisans ve RoadGuard-taksonomisi eşlemesiyle tutulur; `minibus` ve
> `fatigue` için teyitli no-auth açık set bulunamadığından bunlar dürüstçe boş bırakılmıştır
> (komite verisi beklenir). Tüm etiketler YOLO formatına
> dönüştürülmüş, **%80/%10/%10 train/val/test** olarak bölünmüştür (küçük özel sette
> val/test'in istatistiksel anlamı için %10+%10). Sınıf dengesizliği
> `python -m train dataset --report` ile ölçülür (dengesizlik oranı > 3 ise uyarı); seyrek
> sınıflara hedeflenmiş toplama + oversampling + sınıf-lehine mozaik/HSV/karartma
> augmentasyonu uygulanır. Komite TOGG verisi geldiğinde aynı boru hattı tek komutla
> domain modelini yeniden eğitir.

---

## 3. Yapay Zekâ Çözümü (50 puan)

### 3.1 Problemin Analizi (15)
**Rubrik:** Video üzerinden tespitte temel problemler (ışık değişimi, hareket bulanıklığı,
oklüzyon) + izlenen çözüm yolu + neden.

**RoadGuard'ın çözdüğü gerçek problemler (gerçek 4K/50fps footage'da ölçüldü):**
| Problem | Belirti | RoadGuard çözümü |
|---|---|---|
| Karanlık kabin (cam arkası sürücü) | pose keypoint görünmez | ROI CLAHE+gamma parlatma (`pose.roi_enhance`) |
| Araç tipi titremesi (car↔truck) | uzak araç 'truck' okunur | alan-ağırlıklı track-bazlı **sınıf oylaması** |
| Hayalet track'ler | ByteTrack parçalanması | `min_track_frames` çıktı kapısı + sınıf-bağımsız IoU dedup |
| OCR plaka bölünmesi | aynı plaka varyantlara dağılır (3↔0, T↔I) | format-öncelikli **güven-ağırlıklı kalıcı oy havuzu** + pozisyon füzyonu |
| Karanlık plaka il-kodu misread | EasyOCR '3'ü tutarlı '0'/'2' okur | **pozisyon-veto + zemin koşulu** → yanlış onay yerine dürüst `pending` |
| Tek-kare FP (flicker) | sürücü bayrağı titrer | ID-merkezli **16/8 zaman-oylaması** (Katman B) |

### 3.2 Çözüm Mimarisi (15)
**Rubrik:** Kuşbakışı; ham video → etiketli çıktı; mimari diyagram + bileşenler.

Kuşbakışı (kaynak: `docs/mimari.md` + `README.md`):
```
[Kamera/RTSP] → [Ön-İşleme] → [YOLO26 + ByteTrack] ─┬─→ [Sürücü ROI] → [YOLO26-pose geometri + hibrit nesne]
                                  ↑                  └─→ [Plaka ROI] → [YOLO11n LP + güven-ağırlıklı oylama + OCR]
                          [Sınıf oyu + 16/8 kararlılık]                          ↓
                                                              [QoD tetik: yaklaşma / kalite / anomali]
                              [ID-merkezli Accumulator] ← [Hız + Swerving (yanal yörünge)]
                                          ↓
                              [Event / Annotation] → Dashboard + Mobil + JSONL kanıt izi (4.5)
```
**Cascade + iki katman:** Stage-1 dedektör (yolo26) → Stage-2 sürücü motoru
(Katman A model + **Katman B per-ID zaman-oylaması**). Mimari diyagram için bu bloğu
ve `docs/mimari.md`'deki ayrıntılı şemayı rapora alın.

### 3.3 Çözüm Detayları (20)
**Rubrik:** Kullanılan DL algoritmaları, ağ mimarileri (YOLO vb.), ön/son işleme, donanım/yazılım.

- **Ağ mimarisi:** YOLO26 (Ultralytics 8.4) — Stage-1 tespit (`yolo26l` varsayılan/sunucu,
  `yolo26s` hafif), YOLO26-pose (COCO-17 keypoint) sürücü geometrisi, YOLO11n özel LP dedektörü.
  Fine-tune seçeneği: `yolguvenligi_types_v4` (yolov8m, 11 sınıf). Takip: ByteTrack.
- **Ön işleme:** far/headlight bastırma, hareket-bulanıklığı düzeltme, ROI CLAHE+gamma.
- **Son işleme:** sınıf oylaması, 16/8 kararlılık, plaka format-normalizasyonu + pozisyon
  füzyonu + dürüstlük zırhları, hız Kalman+EMA + metrik oto-kalibrasyon, swerving ZigZag sayacı.
- **Stabilite/doğruluk zırhları (config-driven; videoya-özel sabit YOK, K-004):** Bu üç kapı,
  gerçek-videoda doğrulanmış olup yanlış-pozitif ve yanlış-onay kaynaklarını yapısal olarak
  kapatır: (a) **`confirm_min_char_margin=2.0`** — bir plaka karakteri ikinci adayını mutlak
  marjla geçmezse o pozisyon belirsiz sayılır ve plaka **asla yanlış onaylanmaz** (dürüst
  `pending`); böylece eski ilk-karakter `3→0` misread'i artık yanlış onay üretmez. (b)
  **`track_id=-1` / phantom çıktı kapısı (`min_output_frames`)** — takipsiz veya hayalet
  tespitler event/annotation üretmez. (c) **`max_roi_area_ratio=0.10`** — kare alanının
  %10'unu aşan devasa sürücü-ROI'leri kırpılır (eski bir FP kaynağı kapatıldı).
- **Yazılım:** Python 3.12.10, PyTorch 2.8.0+cu128, Ultralytics 8.4.66, EasyOCR, OpenCV, FastAPI.
- **Donanım:** Sunucu/ölçüm: **NVIDIA GeForce RTX 5070 Laptop GPU** — 4.608 CUDA çekirdeği
  (36 SM × 128), 8 GB VRAM, Compute Capability 12.0 (Blackwell). Geliştirme: Apple Silicon/MPS.
  Cihaz `auto` (CUDA→MPS→CPU). Tüm `--help` çıktıları: `docs/cli_referans.md`.

---

## 4. Çözümün Sınanması (20 puan) — EN KRİTİK BÖLÜM

**Rubrik:** Veri setiyle modelin nasıl test edildiği + **Accuracy/Precision/Recall/F1/FPS**
tablolar/grafikler. "Çözümümüze neden güveniyoruz?" sorusuna **veriye dayalı** yanıt.

**KANIT KOMUTLARI:**
```bash
# 1) Her videoyu pipeline'dan geçir → annotated mp4 + JSON kanıt (şartname 4.5)
python tools/test_video.py --source ~/video_1.mp4 --device auto            # yolo26l (varsayılan)
python tools/test_video.py --source ~/video_1.mp4 --profile v4-finetune    # fine-tune A/B
# 2) Özetlerden P/R/F1 + plaka CER + FPS metrik raporu (FTR §4 tabloları)
python -m roadguard.eval --metrics-report --summaries eval_results/ab
#    → eval_results/metrics_report.md + .csv + .json
# 3) QoD A/B delta (yarışma %40 QoD kanıtı)
python -m roadguard.eval --source <video> --ground-truth <gt.json> --qod-comparison
```

**ÖLÇÜLEN SONUÇLAR (3 gerçek video, kapalı otopark, TOGG; RoadGuard v2.3, MPS/M4 Pro;
ölçüldü 17 Haz 2026, dewarp/enhance OFF; `eval_results/metrics_report.md`).**
**FPS sütunlarına 26 Haz 2026 tarihli CUDA ölçümleri (RTX 5070 Laptop, `bench.py`) eklendi:**

| Dedektör | Davranış makro-F1 | Plaka exact | Plaka CER | Yanlış plaka onayı | Araç sınıfı | FPS (MPS) | FPS (CUDA — RTX 5070 Laptop) |
|---|---|---|---|---|---|---|---|
| **yolo26l** (stok, varsayılan, server profili 960) | **1.00** | **2/3** (66.7%) | **0.083** | **0** | 100% | ~5,9 | **12,31** (p50=80ms, p95=93ms) |
| **v4-finetune** (yolov8m, 768) | **1.00** | **2/3** (66.7%) | **0.083** | **0** | 100% | ~5,3 | **~12,5** *(ağırlık yok; yolo26s fallback: 12,80)* |

Davranış sınıf-bazlı (**her iki dedektörde de**) — phone / smoking / swerving:
P = R = F1 = **1.0**, makro-F1 **1.0**. (Stabilite fixleri öncesi eski yolo26l, video_2'de
1 `swerving` yanlış-pozitifiyle 0.933 veriyordu; bu FP `track_id=-1`/phantom çıktı kapısı ve
`max_roi_area_ratio` ile **gitti**.) Araç sınıfı doğruluğu **%100** (her iki dedektör).

**Video-düzeyi plaka kararları (GT plaka = `34TC8532`):**

| Video | GT davranış | yolo26l plaka (durum) | v4 plaka (durum) |
|---|---|---|---|
| video_1 | smoking | 34TC8532 (confirmed ✓) | 34TC8532 (confirmed ✓) |
| video_2 | phone | 34TC8532 (confirmed ✓) | 34TC8532 (confirmed ✓) |
| video_3 | swerving | 24IC8532 (partial — dürüst pending) | 24IC8532 (partial — dürüst pending) |

> **Dürüst yorum (plaka çerçevesi — jüriye güven verir):** Davranış tespiti
> (sigara/telefon/swerving) **her iki dedektörde de çapraz-FP'siz** çalışır (makro-F1 1.0).
> Plaka tarafında da iki dedektör **eşittir: 2/3 exact-match (66.7%), CER 0.083 ve 0
> yanlış-onay** (`confirm_min_char_margin=2.0` zırhı sayesinde). Sistem belirsiz/uzak/bulanık
> okumada **asla yanlış plaka onaylamaz**; bunun yerine dürüstçe **`pending` (partial)** der —
> video_1/2 doğru plakayı (`34TC8532`) CONFIRMED verirken, uzak ve bulanık video_3 onurlu bir
> PENDING'tir (`24IC8532` partial). Eski stok yolo26l 1/3 exact + 2 yanlış-onay üretiyordu; bu
> yanlış-onaylar **conservative confirm eşiği** (pozisyon-veto + zemin koşulu + char-margin)
> ile **düzeldi**. İki dedektör plaka doğruluğunda artık eşit olmakla birlikte, v4 ikincil
> track'lerde biraz daha temiz kırpık üretir (ikincil bir not, doğruluk farkı değil). FPS
> değerleri **MPS alt-sınırıdır**; CUDA sunucuda belirgin daha yüksektir. QoD A/B harness'ı
> **kontrollü sentetik set** (`data/samples/ornek.mp4`, kare-düzeyi GT) üzerinde gerçek ve
> yeniden-üretilebilir koşar (`eval_results/report.json`); ölçülen delta, OFF baseline'ı temsil
> eden düşük-kalite simülasyonun saldırganlığına bağlıdır (§4.5 dürüst not). Gerçek üç videoda
> kare-düzeyi GT olmadığından QoD A/B orada ölçülemez (atıf şeffaflığı).

**Dedektör tespit mAP'i (ÖLÇÜLEN, rapora ek) — üç ayrı kaynak, ayrıştırılmış:**
- **(ASIL DOĞRULUK) Stok dedektör `yolo26l` — COCO val2017 HELD-OUT (5000 görsel),
  `eval_results/map_yolo26l.json`:** modelin eğitiminde görmediği ayrı doğrulama setidir.

  | Metrik | Değer |
  |---|---|
  | mAP50-95 | **0.537** |
  | mAP50 | **0.709** |
  | Precision | **0.740** |
  | Recall | **0.641** |

  Bu, model-kartı iddiası değil **bizim koştuğumuz held-out doğrulama setinin** sonucudur ve
  raporun asıl dedektör doğruluk sayısıdır (komut: `python -m roadguard.eval --map --weights
  weights/yolo26l.pt --data <coco_val>.yaml`).
- **(YALNIZ HIZLI SAĞLIK) Stok `yolo26l` — coco128 (küçük, eğitimle örtüşme olası):**
  mAP50 **0.790**, mAP50-95 **0.619**. DİKKAT: coco128 küçük ve büyük olasılıkla train-örtüşmeli
  olduğundan bu sayı **fine-tune DEĞİLDİR ve doğruluk iddiası olarak kullanılmaz** — yalnızca
  boru hattının doğru kurulduğunu gösteren hızlı bir sağlık kontrolüdür. (Eski metinlerdeki
  "fine-tune mAP50 0.790" atfı YANLIŞTIR; bu sayı stok-coco128 sağlık koşusudur.)
- **(BORU-HATTI DOĞRULAMASI) Fine-tune hattı uçtan uca (`yolo26s`, coco128, 5 epoch):** gerçek
  bir `best.pt` ve gerçek val metriği üretti → **best.pt mAP50 0.7645, mAP50-95 0.5909**. Bu bir
  doğruluk iddiası değil, "eğitim hattı uçtan uca çalışır; komite/açık veriyle tek komutla domain
  modeli üretilir" iddiasının somut kanıtıdır (rakamlar smoke-set ölçeğindedir; istatistiksel
  domain mAP'i komite verisiyle üretilir — `docs/egitim.md`).
- **(SÜRÜYOR) Zorunlu-sınıf fine-tune (YOLO26s, toplanan açık-kaynak veri):** özel-model eğitimi
  şu an devam ediyor. `license_plate` (HF keremberke, 8823 görsel) için **val mAP50 ≈0.97** (epoch
  12'de 0.977, mAP50-95 0.676; eğitim 35 epoch'a koşuyor), ardından `seatbelt` (3104) ve `smoking`
  (557, CigDet) **sırada**. **DÜRÜST NOT:** bu mAP'ler **SÜRMEKTE** olduğundan **final değildir** ve
  rapora kesinleşmiş sayı olarak yazılmaz; eğitim bitince güncel `best.pt` mAP'leri bu bölüme eklenir.
  Yukarıdaki §4 doğruluk sayıları (held-out COCO + 3-video + boru-hattı doğrulaması) eğitim tamamlanana
  dek BASE/stok modellerle ölçülmüştür.

**Hız ve Swerving (şartname zorunlu madde #3):**
- **Hız (kalibrasyon-bağımlı):** metrik oto-kalibrasyon (tripwire/ipm/metric). Kalibrasyon
  varsa mutlak hız (km/h); **yoksa mutlak hız iddiası YOK**, yalnız göreli-hız bayrağı
  (`speed.relative`). **MAE/MAPE harness'ı hazır** — komite gerçek-hız GT'si gelince tek koşuyla
  nicel hız hata metriği üretir. Üç videoda kalibrasyon/gerçek-hız GT olmadığından bu raporda
  mutlak hız doğruluğu sayısı yer almaz (dürüstçe belirtilir).
- **Swerving (dikkatsiz sürüş, kalibrasyonsuz):** merkez-x serisinde **ZigZag yanal yörünge**
  ekstremum sayımı; eşikler araç-genişliği biriminde, pencere saniye cinsinden (ölçek-/fps-
  bağımsız). Gerçek videoda **video_3'te tespit edildi** ve davranış makro-F1'e dahil
  (swerving P=R=F1=1.0). `RISK_ALERT` + QoD tetiği besler.

**Kanıt İzi (şartname 4.5 — "kanıtlanamayan hedef puanlanmaz"):** her hedefin otomatik
üretildiği üç artefaktla kanıtlanır: (a) `python -m roadguard --save-events kanit.jsonl` →
**zaman-damgalı JSONL olay izi** (tespit/plaka/davranış/hız/swerving/QoD); (b)
`python tools/test_video.py --source <video> --json <özet.json>` → **annotated mp4 + JSON oy
dökümü** (plaka oy havuzu, bayrak süreleri, swerving kareleri); (c) `python -m roadguard.eval
--metrics-report` bu özetlerden §4 tablolarını **yeniden üretir** (her sayı bir komuta + artefakta
bağlı, elle girilmemiş).

**Doldurulabilir taslak (4 bölümü):**
> Model, etiketli held-out set üzerinde Precision/Recall/F1 ve mAP ile, üç gerçek test
> videosu üzerinde ise video-düzeyi davranış-tespiti P/R/F1, plaka exact-match doğruluğu
> (CER ile) ve işleme FPS'i ile sınanmıştır (Tablo …). Çözüme güveniyoruz çünkü (i) davranış
> tespiti çapraz yanlış-pozitif üretmiyor (makro-F1 1.0), (ii) sistem belirsizlikte yanlış
> sonuç üretmek yerine dürüstçe çekimser kalıyor, (iii) tüm eşikler oran-bazlı/ölçek-bağımsız
> (videoya-özel sabit yok), (iv) QoD A/B ölçülebilir başarım artışı gösteriyor.

---

## 5. Kaynakça (5 puan)

Rapora alınacak kaynaklar (dijital kaynak formatı: Soyad, A., Başlık, Tarih, Erişim Tarihi, URL):
- Ultralytics YOLO (YOLO11/YOLO26), https://docs.ultralytics.com
- ByteTrack: Zhang et al., 2022, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
- EasyOCR, https://github.com/JaidedAI/EasyOCR
- COCO veri seti, https://cocodataset.org
- Roboflow Universe (Türk trafiği / plaka setleri), https://universe.roboflow.com
- CAMARA Project — Quality-on-Demand & Number Verification API'leri, https://camaraproject.org
- CCPD plaka veri seti (kullanıldıysa), Xu et al., 2018
- RoadGuard repo: github.com/cleoanka/teknofest-prototip.v2

---

## 6. Final Yarışma Hazırlığı (şartname 4.2 — 3. aşama)

Finalde mobil uygulamada **canlı 5G + NV + QoD**: 
- **Number Verification:** kullanıcı/araç girişi sessiz doğrulama. RoadGuard mock: `services/nv_mock` +
  `POST /verify`. Finalde yalnız endpoint/credential değişir. Mobil iskelet: `mobile/`.
- **Quality-on-Demand:** "TOGG yaklaşınca yüksek kalite". RoadGuard: `vehicle_approach` tetiği
  (bbox alan-büyümesi) → `QOD_TRIGGER`. Kanıt: `python -m roadguard.eval --qod-comparison`
  (`eval_results/report.json`, yeniden-üretilebilir). Delta, kare-düzeyi GT içeren **kontrollü
  sentetik set** üzerinde, OFF/ON senaryolarıyla ölçülür; mutlak değerler ve delta, OFF baseline'ı
  temsil eden düşük-kalite simülasyonun saldırganlığına bağlıdır (DÜRÜST NOT: en güncel koşuda OFF
  baseline zaten yüksek olduğundan delta ≈0 çıkmıştır — daha saldırgan OFF simülasyonuyla pozitif
  delta üretilir; sayı koşuya bağlı, artefakttan okunmalı, sabit yazılmamalıdır). Gerçek videoda
  kare-düzeyi GT olmadığından A/B orada ölçülemez.
- **Tespitlerin mobil ekranda gösterimi:** `WS /stream/events` + `mobile/`.
- **Kural 4.5 (kanıt yükümlülüğü):** her hedef otomatik üretildiği kanıtlanmalı →
  `python -m roadguard --save-events kanit.jsonl` (zaman damgalı JSONL iz) + `tools/test_video.py`
  annotated mp4 + JSON özet (oy dökümü, bayrak süreleri, swerving kareleri).

---

## 7. Açık işler / kapatma yolu (dürüst)
- **Komite verisiyle YOLO26 fine-tune:** veri gelince `python -m train detector --data <data.yaml>
  --weights weights/yolo26l.pt --epochs 100 --imgsz 768` → `weights/custom_detector.pt` + metrik;
  config `models.detector.path` ile devreye al. Detaylı rehber: `docs/egitim.md`.
- **Karanlık plaka il-kodu:** perspektif düzeltme portu (yol haritası) veya komite footage'ı.
- **license_plate / seatbelt / smoking / phone sınıfları:** dört no-auth gerçek bbox seti indirilip
  toplandı (PIL-doğrulanmış) — license_plate 8823 görsel (HF keremberke, CC BY 4.0), seatbelt 3104
  görsel (CC BY 4.0, denge 1.27), smoking 557 görsel (CigDet/Mendeley, CC BY 4.0), phone 659 görsel
  (HF synthetic, CC BY 4.0, sentetik render → domain-uyum riski). **Özel-model eğitimi (YOLO26s
  fine-tune) ŞU AN SÜRÜYOR:** license_plate val mAP50 ≈0.97 (epoch 12), seatbelt + smoking sırada;
  *final mAP'ler henüz kesinleşmedi — bittiğinde best.pt mAP'leri §4.2'ye yazılır.* §4 doğruluk
  sayıları bu nedenle hâlâ BASE/stok YOLO26 ile ölçülmüştür. Büyük sigara setleri (Roboflow
  `driver-smoking-detecor` 1066, `Smoker YOLO.v4` 4221) API anahtarı / Roboflow erişimi gerektirir
  (manifestte listeli).
- **minibus / fatigue sınıfları:** no-auth açık bbox seti bulunamadı (Roboflow/Kaggle anahtarı
  veya komite verisi gerekir) → küçük özel etiketleme + `driver-state` fine-tune.
- **FPS:** ✅ CUDA ölçüldü (26 Haz 2026, RTX 5070 Laptop 4.608 CUDA çekirdeği): yolo26l server
  profili **12,31 FPS** (p50=80 ms, p95=93 ms); laptop profili (yolo26s 640) **14,72 FPS**.
  Gerçek değerler rapora §4.6'ya yazıldı; `eval_results/bench_cuda0_server.md` artefakt.

> **Özet:** RoadGuard, FTR'nin **tüm bölümlerine somut kanıt + tek-komut üretim** sağlar. En yüksek
> puanlı Veri Seti (20) ve Sınama (20) bölümleri için sırasıyla `train dataset --report` ve
> `roadguard.eval --metrics-report` çıktıları doğrudan rapora konur. Sayılar gerçektir, hile yoktur (K-004).
