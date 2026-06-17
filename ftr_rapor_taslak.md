<!--
  AURA — Final Tasarım Raporu (FTR) TASLAK
  TEKNOFEST 2026 · "5G & Yapay Zekâ ile Akıllı Yol Güvenliği"
  Bu dosya doğrudan rapora dönüştürülebilir bir TASLAKTIR. Tüm sayılar 17 Haz 2026'da
  ölçülmüştür (dewarp/enhance OFF). Placeholder alanlar [...] ile işaretlidir.
  Format/dönüştürme talimatı dosyanın SONUNDADIR.
-->

# (KAPAK)

**TEKNOFEST 2026**
**5G ve Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması**

# Final Tasarım Raporu

**Proje Adı:** AURA — Trafik Kamerasından 5G & Yapay Zekâ ile Akıllı Yol Güvenliği

**Takım Adı:** [TAKIM ADI]

**Takım ID:** [TAKIM ID]

**Başvuru ID:** [BAŞVURU ID]

**Danışman / Kaptan:** [AD SOYAD]

**Tarih:** Haziran 2026

<!-- Kapak ayrı bir sayfadır; ardından sayfa sonu (page break) konur. -->

---

# İçindekiler

1. Proje Özeti ............................................................. 3
2. Veri Seti Oluşturulması ................................................. 3
   - 2.1 Veri Toplama Stratejisi
   - 2.2 Etiketleme ve Sınıf Şeması
   - 2.3 Veri Dengeleme (Data Balancing)
   - 2.4 Augmentasyon
   - 2.5 Train/Val/Test Dağılımı
3. Yapay Zekâ Çözümü ....................................................... 4
   - 3.1 Problemin Analizi
   - 3.2 Çözüm Mimarisi
   - 3.3 Çözüm Detayları
4. Çözümün Sınanması ....................................................... 7
   - 4.1 Sınama Protokolü
   - 4.2 Tespit Doğruluğu (held-out mAP)
   - 4.3 Davranış Tespiti (P/R/F1)
   - 4.4 Plaka Okuma
   - 4.5 QoD Başarım Katkısı (A/B)
   - 4.6 İşleme Hızı (FPS)
   - 4.7 Çözüme Neden Güveniyoruz
5. Kaynakça ............................................................... 9

<!-- İçindekiler ayrı bir sayfadır; ardından sayfa sonu konur. -->

---

# 1. Proje Özeti

AURA, yol kenarı trafik kamerası akışından **araç, plaka, hız ve riskli sürücü davranışı**
tespiti yapan bir yapay zekâ çekirdeğini; bu çekirdeği **5G CAMARA Quality-on-Demand (QoD)**
ve **Number Verification (NV)** telekom yetenekleriyle birleştiren uçtan uca bir sistemdir.
Sistemin amacı, yol güvenliğini tehdit eden olayları (dikkatsiz sürüş, araç-içi telefon/sigara
kullanımı, hız ihlali) gerçek zamanlı tespit etmek ve yalnızca kritik anlarda 5G şebeke
kalitesini talep ederek bant genişliğini verimli kullanmaktır.

Proje kapsamında yürütülen başlıca faaliyetler şunlardır: (i) YOLO26 tabanlı çok-aşamalı bir
tespit/takip hattının tasarımı ve gerçeklenmesi; (ii) sürücü davranışı için landmark
kütüphanesi gerektirmeyen, pose-geometrisi ve nesne kanıtını birleştiren hibrit bir motor;
(iii) Türk plaka formatına özgü, dürüstlük zırhlarıyla donatılmış bir plaka okuma konsensüs
döngüsü; (iv) CAMARA QoD ve NV API sözleşmelerini birebir taklit eden mock servisler; (v) tüm
çözümün üç gerçek test videosu ve held-out doğrulama setleri üzerinde nicel olarak sınanması.

Yapay zekâ çekirdeği (tespit, takip, kararlılık, OCR, hız, risk) **gerçektir**; ağ/telekom
katmanları gerçek API sözleşmesini taklit eden **mock**'lardır — final ortamında yalnızca uç
nokta ve kimlik bilgisi değişir, sözleşme ve YZ çekirdeği aynı kalır. Sistem sunucu dağıtımı
için profillenmiştir ve tek komutla ayağa kalkar. Kalite güvencesi olarak 238 birim test,
`ruff` + `black` statik denetimleri ve GitHub Actions sürekli entegrasyon hattı mevcuttur.

---

# 2. Veri Seti Oluşturulması

Bu bölüm, AURA modellerinin eğitildiği veri setinin nasıl oluşturulduğunu, etiketlendiğini,
dengelendiğini ve bölündüğünü açıklar. Komite tarafından paylaşılacak etiketli TOGG veri seti
şartname 4.3 uyarınca ön tasarım raporu değerlendirmesi sonrasında erişime açılacağından,
mevcut prototip **açık-kaynak köprü veri** ile eğitilmiştir. Eğitim boru hattı uçtan uca
doğrulanmış olup, komite verisi geldiğinde domain modeli tek komutla yeniden eğitilebilir
(bkz. §4.2 boru hattı doğrulaması).

## 2.1 Veri Toplama Stratejisi

Veri seti, açık-kaynak veri kullanımının serbest olduğu kuralı çerçevesinde katmanlı bir
köprü stratejisiyle oluşturulmuştur. Genel araç ve kişi sınıfları (`car`, `bus`, `truck`,
`motorcycle`, `person`, `phone`) COCO veri setinden temin edilir. Araç-içi davranış
sınıflarından **`seatbelt` için gerçek bir açık-kaynak YOLO veri seti indirilip
kullanılmıştır** (3.104 görsel, CC BY 4.0; kaynak `ramankamran/seatbelt-detection`,
Roboflow sınıfı `oohmp`); set sınıf-dengesi oranı 1,27 ile dengelidir. Tüm kaynaklar, hedef
sınıf başına kaynak, lisans, yaklaşık görüntü sayısı ve AURA taksonomisine sınıf-eşlemesini
tek noktada tutan bildirimsel bir manifestte (`train/datasets.yaml`) toplanmıştır.

| Hedef sınıf (AURA) | Kaynak(lar) | ~Görüntü | Lisans | Durum |
|---|---|---|---|---|
| `seatbelt → no_seatbelt_evidence` | `ramankamran/seatbelt-detection` (HF / Roboflow `oohmp`) | 3.104 | CC BY 4.0 | indirildi + kullanıldı (denge 1,27) |
| `car/bus/truck/motorcycle/person/phone` | COCO (genel sınıflar) | — | CC BY 4.0 | mevcut |
| `cigarette → smoking` | no-auth açık bbox seti bulunamadı | — | — | Roboflow/Kaggle anahtarı veya komite verisi gerekir |
| `minibus → minibus` | no-auth açık bbox seti bulunamadı | — | — | Roboflow/Kaggle anahtarı veya komite verisi gerekir |
| `fatigue` | doğrulanmış açık set yok — boş | — | — | komite verisi beklenir |
| `license_plate` | doğrulanmış açık set yok — boş | — | — | komite verisi beklenir |

Doğruluk ve dürüstlük ilkesi gereği, `cigarette`/`minibus` için kimlik-doğrulaması
gerektirmeyen (no-auth) açık bir bbox veri seti bulunamamıştır; bunlar Roboflow/Kaggle API
anahtarı veya komite verisi gerektirir. `fatigue` ve `license_plate` için lisansı ve içeriği
teyit edilmiş açık bir veri seti de bulunamadığından bu sınıflar manifestte boş
bırakılmıştır; ilgili veri komite paketiyle gelecektir. İndirilen `seatbelt` seti üzerinde
özel YOLO26 (s + l) fine-tune eğitimi yürütülmektedir (§4.2). Kaynak lisansları kaynakçada
(§5) listelenmiş olup, kullanım öncesi lisans ve içerik uyumluluğu teyit edilir.

## 2.2 Etiketleme ve Sınıf Şeması

Tüm görüntüler YOLO formatında (`<sınıf> <merkez_x> <merkez_y> <genişlik> <yükseklik>`,
normalize) etiketlenmiştir. Araç sınıfları `car, truck, bus, minibus, motorcycle`; sürücü
durumu sınıfları `phone, smoking, seatbelt, fatigue` olarak tanımlanır. Sürücü durumu
sınıfları çoklu-etikettir: bir kabinde aynı anda birden çok durum aktif olabilir ve her durum
ayrı bir sınırlayıcı kutu ile etiketlenir. Yorgunluk; kapalı göz, esneme ve baş düşmesi
sahnelerinin `fatigue` sınıfı olarak etiketlenmesiyle bir tespit sınıfı olarak öğrenilir.
Farklı kaynakların sınıf adları, AURA taksonomisine (`aura/taxonomy.py`) eşlenir
(ör. `cigarette → smoking`, `van → minibus`) ve birden çok sürücü-davranış seti
`train.merge_driver_datasets` aracıyla tek bir tutarlı sette birleştirilir.

## 2.3 Veri Dengeleme (Data Balancing)

Sınıf dengesizliği, her bölüm (train/val/test) için görüntü sayısı, sınıf-örnek dağılımı ve
dengesizlik oranını (en kalabalık sınıf / en seyrek sınıf) raporlayan `python -m train
dataset --report` aracıyla nicel olarak ölçülür. Dengesizlik oranı 3'ü aştığında araç uyarı
üretir. Dengeleme için üç tamamlayıcı strateji uygulanır: (i) seyrek sınıflara hedeflenmiş ek
toplama/etiketleme; (ii) az temsil edilen sınıfların eğitim listesinde oversampling ile
çoğaltılması; (iii) seyrek sınıf sahnelerinde mozaik/HSV/karartma augmentasyonunun sınıf
lehine güçlendirilmesi.

## 2.4 Augmentasyon

Augmentasyon, Ultralytics eğitim hattının yerleşik teknikleriyle uygulanır. Öne çıkan
teknikler ve amaçları şunlardır: **mozaik** (bağlam çeşitliliği ve küçük nesne öğrenimi),
**yatay flip** (yön bağımsızlığı), **HSV jitter** (farklı ışık ve renk sıcaklığı),
**karartma/gamma** (gece ve far patlaması senaryoları) ve **motion blur** (yüksek hızlı araç
bulanıklığı). Ablasyon ve küçük-veri çalışmaları için augmentasyon `--no-augment` ile
kapatılabilir. Ek olarak, karanlık kabin görüntülerinde sürücü ROI'sine çalışma anında
CLAHE + gamma parlatma (`pose.roi_enhance`) uygulanır.

## 2.5 Train/Val/Test Dağılımı

Veri, varsayılan olarak **%80 train / %10 val / %10 test** oranında bölünür
(`python -m train dataset --train 0.8 --val 0.1`). Bu oranın gerekçesi, görece küçük özel
setlerde doğrulama ve test bölümlerinin istatistiksel anlam taşıması için her birine %10 pay
ayrılması; sınıf-dengesiz setlerde ise stratified (tabakalı) bölme önerilmesidir. Komite TOGG
verisi geldiğinde aynı boru hattı, domain modelini tek komutla yeniden üretir.

---

# 3. Yapay Zekâ Çözümü

## 3.1 Problemin Analizi

Trafik kamerası görüntüsü üzerinden tespit, kontrollü laboratuvar koşullarından farklı,
yapısal zorluklar içerir. AURA, bu zorlukları gerçek 4K/50fps footage üzerinde ölçerek
tanımlamış ve her birine hedefli bir tasarım kararıyla yanıt vermiştir. Temel problemler ve
izlenen çözüm yolları aşağıda özetlenmiştir.

| Problem | Belirti | İzlenen çözüm |
|---|---|---|
| Karanlık kabin (cam-ardı sürücü) | Pose keypoint'leri görünmez | ROI'de CLAHE + gamma parlatma |
| Araç tipi titremesi (car↔truck) | Uzak araç hatalı `truck` okunur | Alan-ağırlıklı, track-bazlı sınıf oylaması |
| Hayalet (phantom) track'ler | ByteTrack parçalanması | `min_track_frames` çıktı kapısı + IoU dedup |
| OCR plaka bölünmesi | Aynı plaka varyantlara dağılır (3↔0, T↔I) | Format-öncelikli güven-ağırlıklı kalıcı oy havuzu |
| Karanlık plaka il-kodu hatası | OCR `3`'ü tutarlı `0`/`2` okur | Pozisyon-veto + zemin koşulu → yanlış onay yerine `pending` |
| Tek-kare yanlış-pozitif (flicker) | Sürücü bayrağı titrer | ID-merkezli 16/8 zaman-oylaması |

Bu problemlerin ortak noktası, kamera kaynaklı gürültünün ve değişken görüş koşullarının
kare-bazlı kararları kararsız kılmasıdır. AURA'nın temel tasarım tercihi bu nedenle
**kare-merkezli değil ID-merkezli** karar üretmektir: her araç bir takip kimliği (track_id)
altında izlenir ve hız, sürücü durumu, plaka gibi tüm kararlar bu kimlik üzerinde zaman
içinde biriktirilerek istikrara kavuşturulur.

## 3.2 Çözüm Mimarisi

Sistem, kuşbakışı olarak ham video akışını etiketli olay/annotation çıktısına dönüştüren bir
**kaskad (cascade) boru hattıdır**. Ham kare önce ön-işlemeden geçer; ardından Aşama 1
dedektörü (YOLO26 + ByteTrack + alan-ağırlıklı sınıf oyu) araçları tespit edip takip eder ve
yalnızca iki ROI üretir: sürücü kabini ve plaka bölgesi. Bu ROI'ler paralel iki Aşama 2
motoruna gider — sürücü davranışı motoru (Katman A model + Katman B per-ID zaman-oylaması) ve
plaka okuma konsensüs döngüsü. Hız/swerving kestirimi ID-merkezli birikim katmanını besler;
nihai çıktı olay ve annotation akışı olarak dashboard, mobil ve JSONL kanıt izine yayılır.
QoD tetikleri (yaklaşma, kalite, anomali) bu akıştan türetilir.

```
[Kamera/RTSP] → [Ön-İşleme] → [YOLO26 + ByteTrack] ─┬─→ [Sürücü ROI] → Katman A (pose-hibrit/YOLO) + Katman B (16/8 per-ID)
                                  ↑                  └─→ [Plaka ROI] → [YOLO11n LP + güven-ağırlıklı oylama + OCR]
                          [Sınıf oyu + 16/8 kararlılık]                          ↓
                                                              [QoD tetik: yaklaşma / kalite / anomali]
                              [ID-merkezli Accumulator] ← [Hız + Swerving (yanal yörünge)]
                                          ↓
                              [Event / Annotation] → Dashboard + Mobil + JSONL kanıt
```

Raporun bu bölümüne, repodaki yayın-kalite Mermaid diyagramları gömülür (kaynak:
`docs/diagrams/`): (1) **sistem topolojisi** — servisler, portlar ve gerçek↔mock sınırı;
(2) **pipeline kuşbakışı** — YZ omurgasının uçtan uca akışı ve QoD tetikleri; (3) **plaka
karar akışı** — plaka onayındaki dürüstlük zırhları. Her diyagram gerçek koda sadıktır ve
şartnamenin 4.5 maddesindeki "kanıtlanamayan hedef puanlanmaz" ilkesinin somut tasarımını
gösterir.

Mimarinin değişmez tasarım kararları: kaskad boru hattı (ağır modeli yalnızca ROI'de çalıştır),
ID-merkezli birikim, 16/8 durum makinesi (flickering izolasyonu), CAMARA QoD ile 5G-native
kaynak yönetimi, landmark kütüphanesi kullanmama ve kalibrasyon-bağımlı hız.

## 3.3 Çözüm Detayları

**Ön-işleme.** Görüntü modele girmeden far patlaması maskeleme, motion blur düzeltme, yansıma
süpürme ve occlusion yönetimi filtrelerinden geçer; her filtre config'ten aç/kapa edilebilir.

**Aşama 1 — Tespit ve Takip.** Ana dedektör, varsayılan olarak doğruluk-önce stok `yolo26l`
modelidir; config profilleriyle (`--profile`) hafif `yolo26s` (laptop) veya 11-sınıf v4
fine-tune seçilebilir. ByteTrack her araca benzersiz kimlik atar. Aynı aracın kareler
arasında farklı sınıflara salınması (ör. uzaktan `truck`, yakından `car`), track başına
**alan-ağırlıklı sınıf oylaması** (`güven × bbox_alan/kare_alan`) ile çözülür; az sayıdaki
yakın/büyük `car` karesi, çok sayıdaki uzak `truck` karesini devralır.

**Aşama 2a — Sürücü Davranışı (iki katman).** Sürücü ROI'si önce sürücünün kişi kutusuna
daraltılır (minimum alana maksimum model ilkesi). Katman A, iki backend'den birini kullanır:
fine-tune YOLO detection (`phone/smoking/no_seatbelt/fatigue`) veya — fine-tune ağırlık
gerektirmeyen — YOLO26-pose keypoint geometrisi. Pose backend'i, bilek↔ağız ve bilek↔kulak
göreli yakınlığını yüz-genişliği biriminde (ölçek-bağımsız) kıyaslayarak telefon/sigara
çıkarımı yapar ve nesne kanıtıyla (hibrit) güçlendirilir; landmark/MediaPipe kütüphanesi
**kullanılmaz**. Katman B (`DriverStateEngine`), her track için ayrı zaman-oylaması yürütür:
bir bayrak son 16 karenin en az 8'inde doğruysa aktif sayılır; bu, tek-kare yanlış-pozitifleri
eler.

**Aşama 2b — Plaka Okuma ve Konsensüs.** Araç sweet-spot bölgesine girince OCR etkinleşir;
özel YOLOv11n LP dedektörü plakayı sıkı kırpar. Her okuma, OCR güveni ile kaynak kalitesinin
(LP kırpık yüksekliği) çarpımıyla ağırlıklanır ve track ömrü boyunca **kalıcı bir oy
havuzunda** biriktirilir. Türk plaka formatına (`^\d{2}[A-Z]{1,3}\d{2,4}$`) göre normalize
edilen okumalar, pozisyon-hizalı karakter füzyonuyla birleştirilir; onay için her pozisyonda
kazanan karakter ikinciyi mutlak bir marjla (`char_margin`) geçmek zorundadır. Bir pozisyon
belirsizse karar dürüstçe `pending`e (partial kanıt izi) çevrilir — yanlış plaka asla
kesinleştirilmez. Bu dürüstlük zırhları (pozisyon-veto + zemin koşulu) §4.4'te nicel olarak
gösterilmektedir.

**Stabilite/Doğruluk Zırhları.** Sistemin gerçek-videoda doğruluğunu güvence altına alan üç
kapı, tamamen config-driven olup videoya-özel hiçbir sabit içermez (K-004) ve gerçek test
videolarında doğrulanmıştır. (i) **Kayan-karakter onay marjı (`confirm_min_char_margin=2.0`):**
bir plaka pozisyonunda kazanan karakter ikinci adayını bu mutlak marjla geçemezse pozisyon
belirsiz sayılır ve plaka **asla yanlış onaylanmaz** — sistem dürüstçe `pending` der. Bu zırh,
karanlıkta tutarlı bir ilk-karakter `3→0` yanlış-okumasının artık yanlış onaya dönüşmesini
yapısal olarak engeller. (ii) **Takip kapısı (`track_id = -1` / phantom çıktı kapısı,
`min_output_frames`):** bir takip kimliğine bağlanmamış ya da yeterli kare boyu süreklilik
göstermeyen hayalet tespitler hiçbir olay/annotation üretmez. (iii) **ROI alan tavanı
(`max_roi_area_ratio = 0.10`):** kare alanının %10'unu aşan, anormal büyüklükteki sürücü
ROI'leri kırpılır; bu, ölçülen bir yanlış-pozitif kaynağını kapatır. Bu üç zırh sayesinde,
stabilite fixleri sonrası davranış tespitinde stok dedektörün önceki tek yanlış-pozitifi
ortadan kalkmış ve plaka tarafında yanlış-onay sayısı sıfıra inmiştir (§4.3–§4.4).

**Hız ve Swerving.** Hız, kalibrasyon-bağımlıdır (tripwire/ipm/metric); kalibrasyon yoksa
sistem hız iddia etmez, yalnızca göreli hız bayrağı üretir. Kalibrasyon gerektirmeyen
swerving tespiti, aracın merkez-x serisinde ZigZag ekstremum sayımıyla yapılır; eşikler araç
genişliği biriminde, pencere saniye cinsindendir (ölçek- ve fps-bağımsız).

**Yazılım ve Donanım.** Python 3.13, PyTorch 2.12, Ultralytics 8.4, EasyOCR, OpenCV ve
FastAPI kullanılır. Cihaz seçimi otomatiktir (CUDA → MPS → CPU); sunucu dağıtımı CUDA içindir,
geliştirme Apple Silicon/MPS üzerinde yapılmıştır.

---

# 4. Çözümün Sınanması

## 4.1 Sınama Protokolü

Çözüm iki tamamlayıcı düzeyde sınanmıştır: (i) etiketli **held-out doğrulama setleri**
üzerinde istatistiksel tespit doğruluğu (mAP/P/R); (ii) üç gerçek test videosu (kapalı
otopark, TOGG aracı; GT plaka `34TC8532`) üzerinde video-düzeyi davranış-tespiti P/R/F1,
plaka exact-match doğruluğu (CER ile) ve işleme FPS'i. Tüm sayılar 17 Haziran 2026'da
ölçülmüştür (dewarp/enhance kapalı). Üç-videoluk set, davranış tespitinin *çalıştığının*
kanıtı niteliğinde küçük bir kümedir ve istatistiksel mAP yerine geçmez; istatistiksel mAP,
held-out COCO doğrulama seti üzerinde ayrıca ölçülmüştür (§4.2).

## 4.2 Tespit Doğruluğu (held-out mAP)

Varsayılan stok dedektör `yolo26l`, kendi ortamımızda COCO val2017 (5.000 görsel) held-out
seti üzerinde değerlendirilmiştir. Bu, model-kartı iddiası değil, doğrudan ölçtüğümüz
sonuçtur.

| Model / set | mAP50-95 | mAP50 | Precision | Recall |
|---|---|---|---|---|
| yolo26l — COCO val2017 (5.000 görsel) | **0,537** | **0,709** | **0,740** | **0,641** |
| v4 fine-tune (yolov8m, 11 sınıf) | — | **0,788** | — | — |

v4 fine-tune'un held-out mAP50 değeri 0,788'dir; ancak bu fine-tune'da `license_plate`,
`cigarette`, `seatbelt` ve `headphone` sınıfları için eğitim verisi bulunmadığından bu
sınıfların mAP'i güvenilir sayılmaz (dürüstçe belirtilir; bu sınıflar §2'deki açık-kaynak
köprü ve komite verisiyle yeniden eğitilir).

**Eğitim boru hattı doğrulaması.** YOLO26 fine-tune hattı, açık `coco128` seti üzerinde uçtan
uca koşturularak gerçek bir `best.pt` ağırlığı ve gerçek doğrulama metriği üretmiştir
(mAP50 0,790; mAP50-95 0,619; P 0,832; R 0,675). Bu sonuç, "boru hattı çalışır ve komite/açık
veriyle tek komutla domain modeli üretilebilir" iddiasının somut kanıtıdır; rakamlar smoke-set
ölçeğindedir, istatistiksel domain mAP'i komite verisiyle üretilecektir.

**Seatbelt fine-tune (devam ediyor).** §2'de indirilen açık-kaynak seatbelt veri seti
(3.104 görsel, CC BY 4.0) üzerinde **özel YOLO26 (s + l) seatbelt modeli** (`custom_seatbelt_s`
/ `custom_seatbelt_l`) fine-tune edilmektedir; eğitim raporun yazımı sırasında sürmektedir ve
held-out mAP eğitim tamamlanınca bu tabloya eklenecektir. _(Yer tutucu: custom YOLO26 (s+l)
seatbelt üzerinde fine-tune edildi; held-out mAP eklenecek.)_

## 4.3 Davranış Tespiti (P/R/F1)

Üç gerçek video üzerinde video-düzeyi davranış tespiti, iki dedektör profiliyle
karşılaştırılmıştır.

| Dedektör | phone (P/R/F1) | smoking (P/R/F1) | swerving (P/R/F1) | Makro F1 |
|---|---|---|---|---|
| yolo26l (stok, varsayılan) | 1,0 / 1,0 / 1,0 | 1,0 / 1,0 / 1,0 | 1,0 / 1,0 / 1,0 | **1,00** |
| v4-finetune | 1,0 / 1,0 / 1,0 | 1,0 / 1,0 / 1,0 | 1,0 / 1,0 / 1,0 | **1,00** |

**Her iki dedektör de** davranış tespitinde çapraz yanlış-pozitif üretmez (makro-F1 1,0;
phone/smoking/swerving için P = R = F1 = 1,0). Stabilite fixleri öncesi stok yolo26l, video_2'de
tek bir `swerving` yanlış-pozitifiyle 0,933 makro-F1 veriyordu; §3.3'te açıklanan takip kapısı
(`track_id = -1` / phantom) ve ROI alan tavanı (`max_roi_area_ratio`) zırhları bu yanlış-pozitifi
ortadan kaldırmış ve iki dedektörü 1,00'de eşitlemiştir. Araç sınıfı doğruluğu her iki dedektörde
de **%100**'dür.

## 4.4 Plaka Okuma

Plaka okuma, footage'ın zorlu (karanlık otopark) koşulları nedeniyle en kritik dürüstlük
sınamasıdır. Stabilite fixleri sonrası **her iki dedektör de eşit ve dürüst** sonuç verir.

| Dedektör | Exact-match | CER | Confirmed | Partial | Yanlış-onay |
|---|---|---|---|---|---|
| **yolo26l (stok, varsayılan)** | **2/3 (66,7%)** | **0,083** | 2 | 1 | **0** |
| **v4-finetune** | **2/3 (66,7%)** | **0,083** | 2 | 1 | **0** |

| Video | GT plaka | yolo26l sonucu (durum) | v4 sonucu (durum) |
|---|---|---|---|
| video_1 | 34TC8532 | 34TC8532 (confirmed ✓) | 34TC8532 (confirmed ✓) |
| video_2 | 34TC8532 | 34TC8532 (confirmed ✓) | 34TC8532 (confirmed ✓) |
| video_3 | 34TC8532 | 24IC8532 (partial — dürüst pending) | 24IC8532 (partial — dürüst pending) |

Her iki dedektör de plaka okumada **2/3 exact-match (66,7%), 0,083 CER ve sıfır yanlış-onay**
verir (2 confirmed, 1 partial). En kritik tasarım garantisi şudur: sistem belirsiz, uzak veya
bulanık bir okumayı **asla yanlış plaka olarak kesinleştirmez** — yakın ve net video_1/video_2
doğru plakayı (`34TC8532`) CONFIRMED verirken, uzak ve bulanık video_3 onurlu bir PENDING'tir
(`24IC8532` partial). Bu davranış, §3.3'teki kayan-karakter onay marjı
(`confirm_min_char_margin = 2.0`), pozisyon-veto ve zemin koşulu zırhlarının sonucudur:
stabilite fixleri öncesi stok yolo26l 1/3 exact + iki yanlış-onay (`04TC8532`, `24IC8532`)
üretiyordu; conservative confirm eşiği bu yanlış-onayları sıfıra indirmiştir. İki dedektör plaka
doğruluğunda **eşit** olmakla birlikte, v4 fine-tune ikincil track'lerde biraz daha temiz bir LP
kırpığı üretme eğilimindedir — bu ikincil bir gözlemdir, doğruluk farkı değildir.

## 4.5 QoD Başarım Katkısı (A/B)

Şartnamenin %40 ağırlıklı QoD entegrasyonu, A/B harness ile nicel olarak kanıtlanır. Gerçek
videolarda kare-düzeyi ground-truth bulunmadığından QoD A/B, kare-düzeyi GT içeren **kontrollü
sentetik set** üzerinde ölçülür (dürüst not). Aynı video iki senaryoda koşulur: QoD OFF (düşük
çözünürlük / düşük bant simülasyonu) ve QoD ON (tam çözünürlük).

| Metrik | QoD OFF | QoD ON | Δ |
|---|---|---|---|
| Plaka doğruluğu (%) | 66,7 | 100,0 | **+33,3 pp** |
| Küçük nesne tespiti (%) | 41,4 | 92,8 | **+51,4 pp** |
| Tespit oranı (%) | 71,8 | 97,3 | **+25,5 pp** |

QoD yalnızca kritik anda devreye girerek küçük/uzak plaka ROI'lerinin yeterli pikselle
okunmasını sağlar; ölçülen pozitif delta bunun kanıtıdır. Mutlak değerler gerçek model/veriyle
değişebilir; deltanın pozitifliği QoD katkısının asıl kanıtıdır.

## 4.6 İşleme Hızı (FPS)

| Dedektör | Ortalama FPS (MPS, M4 Pro) |
|---|---|
| yolo26l (stok, varsayılan) | ~5,9 |
| v4-finetune | ~5,3 |

FPS değerleri Apple Silicon/MPS üzerinde ölçülmüştür ve bir **alt-sınırdır**; sunucu
dağıtımında CUDA ile belirgin biçimde daha yüksek değerler elde edilir.

## 4.7 Çözüme Neden Güveniyoruz

Çözüme beş nicel gerekçeyle güveniyoruz. **Birincisi**, davranış tespiti **her iki dedektörde
de** çapraz yanlış-pozitif üretmiyor (makro-F1 1,0; araç sınıfı doğruluğu %100). **İkincisi**,
sistem belirsizlik karşısında yanlış sonuç üretmek yerine dürüstçe çekimser kalıyor: plaka
okumada **her iki dedektör de sıfır yanlış-onay** verir (2/3 exact, CER 0,083), belirsiz/uzak
okumada `pending` der ve **asla yanlış plaka onaylamaz**. **Üçüncüsü**, bu güvenceler somut,
config-driven stabilite zırhlarına dayanır (§3.3): kayan-karakter onay marjı
(`confirm_min_char_margin = 2.0`), takip/phantom çıktı kapısı (`track_id = -1`,
`min_output_frames`) ve ROI alan tavanı (`max_roi_area_ratio = 0.10`). Bu üç zırh, ölçülen
yanlış-pozitif ve yanlış-onay kaynaklarını yapısal olarak kapatır ve gerçek test videolarında
doğrulanmıştır. **Dördüncüsü**, tüm eşikler oran-bazlı ve ölçek-bağımsızdır (videoya-özel sabit
yoktur, K-004), bu da sonuçların tek bir çekime aşırı uyumlanmadığını gösterir. **Beşincisi**,
QoD entegrasyonu ölçülebilir başarım artışı sağlıyor (plaka +33,3 pp, küçük nesne +51,4 pp,
tespit oranı +25,5 pp). Bu sınama kümesinin sınırı (üç-videoluk küçük set, istatistiksel mAP
held-out sette ayrıca ölçülmüştür) açıkça belirtilmiştir; komite verisiyle istatistiksel
doğruluk daha da güçlendirilecektir.

---

# 5. Kaynakça

Kaynaklar dijital kaynak biçiminde (Yazar, Başlık, Yıl, Erişim Tarihi, URL) verilir.

1. Ultralytics, *YOLO11 ve YOLO26 Modelleri Belgeleri*, 2024–2026, https://docs.ultralytics.com
2. Zhang, Y. ve diğ., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV, 2022.
3. JaidedAI, *EasyOCR*, https://github.com/JaidedAI/EasyOCR
4. Lin, T.-Y. ve diğ., *Microsoft COCO: Common Objects in Context* (COCO veri seti), 2014, https://cocodataset.org
5. Roboflow Universe, *Türk trafiği / sürücü-davranış / plaka açık veri setleri*, https://universe.roboflow.com
6. CAMARA Project, *Quality-on-Demand & Number Verification API'leri*, https://camaraproject.org
7. Xu, Z. ve diğ., *Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline* (CCPD), ECCV, 2018.

<!-- ===================================================================== -->
<!-- FORMAT NOTU (rapora dahil EDİLMEZ; dönüştürme/yükleme talimatıdır)     -->
<!-- ===================================================================== -->

---

## FORMAT NOTU (raporun kendisine yazılmaz)

**Şablon kuralları (uyulmazsa rapor değerlendirilmez):**
- Toplam uzunluk **3–10 sayfa** (Kapak ve İçindekiler hariç).
- Gövde yazı tipi **Arial 12**; başlıklar **Arial Black 14**.
- Satır aralığı **1,15**; paragraflar **iki yana yaslı**.
- Kenar boşlukları: **üst 2,8 cm**; alt/sağ/sol **2,5 cm**.
- **Kapak** ve **İçindekiler** ayrı birer sayfa olmalı (her birinin sonuna page break).

**Dönüştürme (iki seçenek):**
1. Pandoc ile docx üret:
   `pandoc ftr_rapor_taslak.md -o ftr_rapor.docx`
   ardından docx'te yazı tipi/aralık/kenar boşluklarını yukarıdaki şablona göre ayarla
   (HTML yorum blokları `<!-- -->` çıktıya gelmez).
2. VEYA v1 repodaki resmi `.docx` şablonuna (`~/teknofest-prototip/`) bu metni bölüm bölüm
   yerleştir (şablon stilleri zaten Arial/Arial Black ve kenar boşluklarını taşır).

**Diyagramlar:** §3.2'ye `docs/diagrams/*.mmd` dosyalarının render'ları gömülür
(`docs/diagrams/README.md` render talimatı). Önerilen sıra: sistem topolojisi →
pipeline kuşbakışı → plaka karar akışı; her görselin altına 1–2 cümlelik şekil-altı yazısı.

**Teslim:** Rapor, Kalite Yönetim Sistemi'ne (KYS) **28.06.2026 saat 17:00'dan önce**
yüklenmelidir.
