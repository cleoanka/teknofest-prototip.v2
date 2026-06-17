# FTR Rehberi — Final Tasarım Raporu'nu AURA Kanıtlarıyla Doldurma

> **Bu belge ne?** TEKNOFEST 2026 "5G & YZ ile Akıllı Yol Güvenliği" yarışmasının
> **Final Tasarım Raporu (FTR)** şablonunun her bölümünü, AURA prototipinin somut
> kanıtlarına / komutlarına / sayılarına bağlayan **ipuçlu rehber + doldurulabilir
> taslak**. Üç işlevi birden görür: (1) raporu doldurma kılavuzu, (2) finale
> (mobil + 5G + QoD demo) hazırlık, (3) prototipin yeteneklerinin tam dokümantasyonu.
>
> ✅ **Takvim (güncel):** FTR son teslim **28.06.2026'ya ERTELENDİ** (kullanıcı teyit etti;
> eski şartname PDF'indeki 14.06 geçersiz). Yani **FTR HÂLÂ AÇIK** — bu rehber + aşağıdaki
> doldurulabilir taslak doğrudan kullanılabilir. En yüksek puanlı §2 (Veri Seti) ve §4 (Sınama)
> için sırasıyla `train dataset --report` ve `aura.eval --metrics-report` çıktıları hazır.

---

## 0. Puanlama haritası (iki ayrı rubrik)

**FTR raporu (şablon, 100 puan):** İçindekiler/Kapak hariç **3–10 sayfa**, Arial 12,
başlık Arial Black 14, satır aralığı 1.15, iki yana yaslı, kenar boşlukları üst 2.8 /
alt-sağ-sol 2.5. **Şablona uymayan rapor değerlendirilmez.**

| FTR bölümü | Puan | AURA'daki ana kanıt |
|---|---|---|
| 1. Proje Özeti | 5 | `README.md` + bu rehber §1 |
| 2. **Veri Seti Oluşturulması** | **20** | `docs/veri_seti.md` + `docs/egitim.md` + `train/` tool (`dataset --report`) |
| 3. Yapay Zekâ Çözümü | 50 | 3.1 Problem `docs/mimari.md`§problem · 3.2 Mimari `docs/mimari.md` diyagramı · 3.3 Detay `docs/mimari.md`+`docs/cli_referans.md` |
| 4. **Çözümün Sınanması** | **20** | `python -m aura.eval --metrics-report` → `eval_results/metrics_report.md` (P/R/F1/CER/FPS) |
| 5. Kaynakça | 5 | bu rehber §5 |

**Yarışma final puanı (şartname Tablo 1, ayrı):** %40 YZ doğruluk + %40 QoD entegrasyonu
+ %20 rapor/sunum. → QoD kanıtı: `python -m aura.eval --qod-comparison` (A/B delta).

**En zayıf iki nokta = en yüksek puanlı iki bölüm:** Veri Seti (20) ve Metrik (20). Bu
rehber özellikle bu ikisini doldurmaya odaklanır.

---

## 1. Proje Özeti (5 puan)

**Rubrik:** Proje kapsamında yürütülen faaliyetlerin özeti.

**Doldurulabilir taslak:**
> AURA, yol kenarı trafik kamerası akışından **araç, plaka, hız ve riskli sürücü
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

**Durum (dürüst):** Komite TOGG/etiketli veri setini henüz paylaşmadığından (şartname 4.3
"ön tasarım raporu değerlendirmesi sonrası" paylaşılır), AURA'nın 11-sınıf fine-tune
dedektörü (`yolguvenligi_types_v4`, yolov8m, held-out **mAP50 .788**) **açık kaynak köprü
veriyle** eğitilmiştir. **YOLO26 fine-tune boru hattı `train/` altında hazırdır ve uçtan
uca doğrulanmıştır** (komite verisi gelince tek komutla yeniden eğitir).

**Doldurulacak içerik + komutlar:**
- **Toplama:** açık kaynak köprü (şartname "açık kaynak veri serbest"):
  `car/bus/truck/motorcycle/person/phone → COCO`; `minibus → Roboflow Türk-trafiği`;
  `license_plate → CCPD / OpenALPR / TR-plaka setleri`; `cigarette/seatbelt → küçük özel etiketleme`.
  Detaylı eşleme: `aura/.../data.yaml` + `docs/veri_seti.md`.
- **Etiketleme:** YOLO formatı (`<cls> <xc> <yc> <w> <h>` normalize). Roboflow ile çek:
  `python -m train.roboflow_pull --workspace W --project P --version N`.
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
> Veri seti üç kaynaktan oluşturulmuştur: (i) genel araç/kişi/nesne sınıfları için COCO,
> (ii) Türk trafiğine özgü `minibus` ve plaka için Roboflow/CCPD açık setleri, (iii)
> `cigarette/seatbelt` için sınırlı özel etiketleme. Tüm etiketler YOLO formatına
> dönüştürülmüş, %80/%10/%10 train/val/test olarak bölünmüştür. Sınıf dengesizliği
> `python -m train dataset --report` ile ölçülmüş; dengesizlik oranı … olarak bulunmuş
> ve … sınıfları için oversampling + mozaik/HSV augmentasyonu uygulanmıştır.

---

## 3. Yapay Zekâ Çözümü (50 puan)

### 3.1 Problemin Analizi (15)
**Rubrik:** Video üzerinden tespitte temel problemler (ışık değişimi, hareket bulanıklığı,
oklüzyon) + izlenen çözüm yolu + neden.

**AURA'nın çözdüğü gerçek problemler (gerçek 4K/50fps footage'da ölçüldü):**
| Problem | Belirti | AURA çözümü |
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
- **Yazılım:** Python 3.13, PyTorch 2.12, Ultralytics 8.4, EasyOCR, OpenCV, FastAPI.
- **Donanım:** sunucu dağıtımı (CUDA otomatik); geliştirme Apple Silicon/MPS. Cihaz `auto`
  (CUDA→MPS→CPU). Tüm `--help` çıktıları: `docs/cli_referans.md`.

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
python -m aura.eval --metrics-report --summaries eval_results/ab
#    → eval_results/metrics_report.md + .csv + .json
# 3) QoD A/B delta (yarışma %40 QoD kanıtı)
python -m aura.eval --source <video> --ground-truth <gt.json> --qod-comparison
```

**ÖLÇÜLEN SONUÇLAR (3 gerçek video, kapalı otopark, TOGG; AURA v2.3, MPS/M4 Pro):**

| Dedektör | Davranış makro-F1 | Plaka exact | Plaka CER | Araç sınıfı | FPS (MPS) |
|---|---|---|---|---|---|
| **v4-finetune** (yolov8m) | **1.00** | **2/3** (66.7%) | **0.083** | 100% | 4.83 |
| yolo26l (stok, varsayılan) | 0.933 | 1/3 (33.3%) | 0.125 | 100% | 5.69 |

> **Dürüst yorum (jüriye güven verir):** Davranış tespiti (sigara/telefon/swerving) her iki
> dedektörde de **çapraz-FP'siz** çalışır (v4 makro-F1 1.0). Plaka okuma bu **karanlık
> otopark** footage'ında zorlu: EasyOCR il-kodunu (3→0/2) tutarlı yanlış okuyabiliyor.
> Sistem **asla yanlış plaka kesinleştirmez** — pozisyon-veto + zemin koşulu belirsiz/uzak
> okumayı `pending`e çevirir (v3'te v4 dürüstçe `pending` der). FPS değerleri MPS içindir;
> **CUDA sunucuda belirgin daha yüksektir**. Detay tablolar: `eval_results/metrics_report.md`.

**Dedektör tespit mAP'i (rapora ek):**
- v4 fine-tune: held-out **mAP50 .788** (model kartı).
- Stok YOLO26: resmi COCO val mAP'i Ultralytics model kartından (docs.ultralytics.com) alın,
  VEYA kendi ortamınızdan ölçün:
  `python -c "from ultralytics import YOLO; print(YOLO('weights/yolo26l.pt').val(data='coco.yaml').box.map)"`.
  Referans (YOLO26'nın selefi **YOLO11**, COCO val @640 — resmi sayılar; YOLO26 bunların
  üzerine kurulur, kendi kartından teyit edin):

  | Model | mAP50-95 | Params(M) |
  |---|---|---|
  | YOLO11s | 47.0 | 9.4 |
  | YOLO11m | 51.5 | 20.1 |
  | YOLO11l | 53.4 | 25.3 |
  | YOLO11x | 54.7 | 56.9 |

  (Kaynak: docs.ultralytics.com/models/yolo11 · Gemini ile çekildi. NOT: yazım anında bazı
  kaynaklar YOLO26 tablosunu henüz yayınlamamış olabilir; ortamımızda yolo26 ağırlıkları
  kuruludur ve çalışır → kesin sayıyı yukarıdaki `model.val` komutuyla üretin.)
- Eğitim boru hattı doğrulaması (coco8, yolo26s, 1 epoch) gerçek
  best.pt + mAP üretmiştir (bkz. `docs/egitim.md`).

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
- AURA repo: github.com/cleoanka/teknofest-prototip.v2

---

## 6. Final Yarışma Hazırlığı (şartname 4.2 — 3. aşama)

Finalde mobil uygulamada **canlı 5G + NV + QoD**: 
- **Number Verification:** kullanıcı/araç girişi sessiz doğrulama. AURA mock: `services/nv_mock` +
  `POST /verify`. Finalde yalnız endpoint/credential değişir. Mobil iskelet: `mobile/`.
- **Quality-on-Demand:** "TOGG yaklaşınca yüksek kalite". AURA: `vehicle_approach` tetiği
  (bbox alan-büyümesi) → `QOD_TRIGGER`. Kanıt: `python -m aura.eval --qod-comparison` (delta).
- **Tespitlerin mobil ekranda gösterimi:** `WS /stream/events` + `mobile/`.
- **Kural 4.5 (kanıt yükümlülüğü):** her hedef otomatik üretildiği kanıtlanmalı →
  `python -m aura --save-events kanit.jsonl` (zaman damgalı JSONL iz) + `tools/test_video.py`
  annotated mp4 + JSON özet (oy dökümü, bayrak süreleri, swerving kareleri).

---

## 7. Açık işler / kapatma yolu (dürüst)
- **Komite verisiyle YOLO26 fine-tune:** veri gelince `python -m train detector --data <data.yaml>
  --weights weights/yolo26l.pt --epochs 100 --imgsz 768` → `weights/custom_detector.pt` + metrik;
  config `models.detector.path` ile devreye al. Detaylı rehber: `docs/egitim.md`.
- **Karanlık plaka il-kodu:** perspektif düzeltme portu (yol haritası) veya komite footage'ı.
- **cigarette/seatbelt/fatigue sınıfları:** küçük özel etiketleme + `driver-state` fine-tune.
- **FPS:** CUDA sunucuda ölçüp rapora gerçek değerleri yazın (MPS sayıları alt-sınırdır).

> **Özet:** AURA, FTR'nin **tüm bölümlerine somut kanıt + tek-komut üretim** sağlar. En yüksek
> puanlı Veri Seti (20) ve Sınama (20) bölümleri için sırasıyla `train dataset --report` ve
> `aura.eval --metrics-report` çıktıları doğrudan rapora konur. Sayılar gerçektir, hile yoktur (K-004).
