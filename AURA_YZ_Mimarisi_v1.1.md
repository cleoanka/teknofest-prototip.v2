# Proje AURA: 5G Akıllı Yol Güvenliği — YZ Mimarisi

**Sürüm:** 1.1 — Mimari Taslak  
**Kapsam:** Yalnızca YZ/İnference Katmanı

> Output/Alert katmanı (baz istasyonu, mobil istemci, dashboard entegrasyonu) kasıtlı olarak bu dokümanın kapsamı dışındadır. Bu modül event-driven bir microservice olarak bağımsız çalışır; harici servislere API arayüzü üzerinden bağlanır.

> **Not:** Mimari iskelet içindeki YOLO26s, YOLO26l ve OCR bileşenleri şu an yer tutucudur. Proje olgunlaştıkça bu birimler, AURA'nın özel veri setiyle eğitilmiş custom modellerle ikame edilecektir.

---

## Genel Mimari Akışı

```
[Kamera Girişi]
      ↓
[1. Ön-İşleme Katmanı]
      ↓
[2. Aşama 1 — YOLO26s: ROI Tespiti + ByteTrack]  ←→  [Kararlılık / 16/8 Kuralı]
           ↓                          ↓
   [Sürücü Kabini ROI]          [Plaka ROI]
           ↓                          ↓
[4. YOLO26l — Driver State]   [5. Sweet Spot + OCR + Voting Buffer]
           ↓                          ↓
                                [QoD Tetikleyici]
                                       ↓
                         [ID-Merkezli Karar Akümülatörü]
                                       ↓
                         [Hız Tahmini — Kalibrasyon Bağımlı]
                                       ↓
                              [Event / Data Stream]
                              (Downstream API'lara)
```

---

## 1. Dinamik Ön-İşleme Katmanı

Görüntü herhangi bir modele girmeden önce çevresel gürültüyü temizleyen ilk filtredir.

### 1.1 Far Patlaması Maskeleme (Headlight Suppression)
Gece koşullarında araç farlarının plaka etrafında oluşturduğu beyaz glare bölgesi tespit edilir ve lokal olarak maskelenir.

### 1.2 Algoritmik Görüntü Onarıcıları

- **Motion Blur Düzeltme:** Yüksek hızlı araçların ürettiği hareket bulanıklığını giderir.
- **Yansıma Süpürme:** Ön cam ve ıslak yüzey yansımalarını bastırır.
- **Occlusion Handling:** Direk, ağaç veya araç çakışması gibi geçici kapanma senaryolarında görünürlük kaybını yönetir.

---

## 2. Aşama 1 — ROI Tespiti ve Takip (YOLO26s + ByteTrack)

Tüm karenin işlendiği, uç cihazda çalışan ana tespit motorudur.

### 2.1 YOLO26s — Hedef Tespiti ve ROI Kırpma

Hedef araç sınıflarını (araba, kamyon, otobüs) tespit eder. Pipeline'daki ağır bileşenlere tam çözünürlüklü kare göndermek yerine yalnızca iki spesifik ROI kırpması üretir:

| ROI | Hedef Bileşen |
|-----|---------------|
| Sürücü Kabini | Aşama 2 — YOLO26l Driver State |
| Plaka Bölgesi | Aşama 2 — OCR Konsensüs Döngüsü |

Bu tasarım sayesinde YOLO26l ve OCR modülleri hiçbir zaman tam kare üzerinde çalışmaz; hesap yükü ve gecikme minimumda tutulur.

### 2.2 ByteTrack — Kimlik Yönetimi

Tespit edilen her araca benzersiz bir ID atanır. Hız verisi, sürücü durumu ve plaka bilgisi dahil tüm sistem kararları bu ID üzerinde zaman içinde biriktirilir. Sistem ID-merkezli çalışır, kare-merkezli değil.

---

## 3. Kararlılık ve Durum Koruması — 16/8 Kuralı

Kamera kaynaklı anlık "hayalet" tespitlerin (flickering) sistem durumunu bozmasını engelleyen state machine katmanıdır.

**Kural:** Bir ID'ye atanmış mevcut durumun güncellenebilmesi için sistemin, 16 ardışık karenin en az 8'inde mevcut confidence skorunun üzerinde bir skor ile yeni durumu tutarlı biçimde tespit etmesi şarttır. Bu eşik sağlanmadığında yüksek güvenilirliğe sahip önceki veri korunur; override yapılmaz.

Bu mekanizma özellikle kötü ışık koşulları ve geçici kapanma (occlusion) senaryolarında sistemin yanlış alarma sürüklenmesini engeller.

---

## 4. Aşama 2 — Derin Analiz ve Sınıflandırma (YOLO26l)

Aşama 1'in ürettiği hedefli ROI verileri üzerinde çalışan detaylı analiz katmanıdır.

### 4.1 YOLO26l — Sürücü Durumu Tespiti (Driver State)

YOLO26s'in kırpıp ilettiği Sürücü Kabini ROI'sini girdi olarak alır. Aşağıdaki durum sınıflarını tespit eder ve sonucu ilgili ID'ye yazar:

- Telefon kullanımı
- Sigara içme
- Emniyet kemeri takmama

MediaPipe gibi landmark tabanlı yaklaşımlar bu mimaride kullanılmaz. Trafik kamerası montaj açıları ve değişken görüş mesafeleri göz önüne alındığında landmark sistemleri tutarsız ve kırılgan sonuçlar üretir.

---

## 5. Plaka Okuma ve Konsensüs Döngüsü

Sistemin hesap yükü açısından en maliyetli parçası olduğundan katı kaynak yönetimi kurallarıyla çalışır.

### 5.1 Sanal Okuma Bölgesi (Sweet Spot)

Araç uzaktayken OCR modülü pasif konumdadır. Araç, kameranın en yüksek optik netliği sağlayabildiği önceden tanımlı sanal koordinata girdiğinde OCR etkinleşir.

### 5.2 Voting Buffer — Oy Birliği Havuzu

OCR, araç Sweet Spot bölgesindeyken ardışık okumalar yapar ve sonuçları havuzda toplar.

### 5.3 Karar Mekanizması

```
Voting Buffer Doldu
        ↓
 Konsensüs var mı?
    ↓           ↓
  EVET         HAYIR
    ↓             ↓
Plaka ID'ye    QoD Kalite Tetikleyicisi
kalıcı yaz  →  Yüksek çözünürlük talebi
OCR kapat   →  Yeniden okuma döngüsü
Erken çıkış
```

---

## 6. QoD — Dinamik Kaynak Yönetimi (CAMARA QoD API)

Yalnızca gerektiğinde ağ ve işlem kaynaklarını artıran akıllı yönetim sistemidir. 5G ağını statik bir bant genişliği olarak değil, talep üzerine şekillendirilebilen dinamik bir kaynak havuzu olarak kullanan bu tasarım, projenin mimarisini gerçek anlamda 5G-native kılar.

### 6.1 Optimizasyon Tetikleyicisi

Hız ve yörünge analizinde anormallik veya tehlike sezildiğinde anında devreye girer. Hedef: gecikmeyi (latency) düşürmek ve FPS'i artırmak.

### 6.2 Kalite Tetikleyicisi

Şu koşullardan birinde tetiklenir:

- Voting Buffer'dan ret kararı geldiğinde
- İlk plaka veya Driver State tespiti için piksel kalitesi yetersiz kaldığında

---

## 7. Hız Tahmini Modülü (Kalibrasyon Bağımlı)

Hız ölçümü doğrudan kamera kurulum parametrelerine bağımlıdır. Bu modül, sahaya özgü kalibrasyon şartlarına göre üç moddan biriyle çalışır:

| Mod | Şart | Açıklama |
|-----|------|----------|
| `tripwire` | Sabit kamera, bilinen mesafe | İki sanal çizgi arası ByteTrack frame delta'sı × gerçek mesafe |
| `ipm` | Kamera intrinsics + montaj verisi mevcut | Homography/IPM ile piksel → gerçek dünya dönüşümü |
| `disabled` | Kalibrasyon verisi yok | Hız üretilmez; `relative_velocity_flag` üretilir |

Kalibrasyon şartları karşılanamıyorsa sistem hız iddiasında bulunmaz; bunun yerine anormal hız davranışı flagı üretmeye geri döner. Bu, sistemin kendi sınırlarını tanıyan ve aşmayan bir mimari karardır.

---

## 8. Planlanan Ek Modüller (Gelişmiş Optimizasyonlar)

Çevresel şartların ve 5G ağ kapasitesinin elverdiği durumlarda sisteme entegre edilecek veya toggle edilebilecek esnek yapılardır.

### 8.1 Sıfır Atık Veri Aktarımı (Zero-Waste Payload)

5G bant genişliğini gereksiz yere tüketmemek için downstream servislere hiçbir zaman tam çözünürlüklü video gönderilmez. Yalnızca YOLO26s'in ürettiği küçük boyutlu ROI görüntüleri ve ID'ye bağlı yapılandırılmış metin verisi iletilir.

### 8.2 Süper Çözünürlük (ESRGAN vb.)

QoD tetiklense bile optik sınırların aşılamadığı çok uzak mesafelerde, kırpılan bulanık plaka görüntüleri OCR'a girmeden önce yapay zeka tabanlı upscaling algoritmaları ile netleştirilir.

### 8.3 Homography / IPM (Kuş Bakışı Dönüşümü)

Hız ve yörünge hesabı için piksel koordinatlarını gerçek dünya metriklerine dönüştüren matematiksel matrisler. Her kamera açısı ve montaj konfigürasyonu bu dönüşüme uygun olmadığından toggle edilebilir bir opsiyondur; yalnızca şartların tam karşılandığı sahnelerde etkinleştirilir.

---

## Mimari Tasarım Kararları — Özet

| Karar | Gerekçe |
|-------|---------|
| Cascade pipeline (YOLO26s → YOLO26l) | Ağır modeli yalnızca gerektiği boyutta ve noktada çalıştırmak |
| ID-merkezli durum birikimi | Kare bazlı sistemlere kıyasla tutarlı ve sürdürülebilir karar üretimi |
| 16/8 State Machine | Flickering ve geçici gürültüden izole edilmiş kararlı durum yönetimi |
| CAMARA QoD entegrasyonu | 5G'yi statik bant genişliği değil dinamik kaynak havuzu olarak kullanmak |
| Edge-first işlem | İşlem uçta gerçekleşir, downstream'e yalnızca event/data stream gönderilir |
| Decoupled output katmanı | YZ modülü upstream'i ve downstream'i bilmez; microservice prensibi |
| Kalibrasyon bağımlı hız modülü | Sistemin kendi sınırlarını tanıması ve koşullara göre doğru modu seçmesi |
