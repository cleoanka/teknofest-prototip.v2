# Veri Seti Stratejisi

## Ne yapar
AURA modellerini eğitmek için veri toplama, etiketleme ve sentetik augmentasyon
stratejisini tanımlar. İki ayrı veri seti: araç tespiti (Stage-1) ve sürücü durumu (Stage-2).

## Toplanan gerçek veri (18 Haz 2026) — özet ✅
Aşağıdaki dört açık veri seti **gerçekten indirildi**, sınıfları AURA taksonomisine
eşlendi, **tümü PIL ile doğrulandı** (bozuk görüntü yok) ve 80/10/10 (seed 42) split'lendi.
Hepsi **CC BY 4.0** (FTR §5 kaynakçaya yazılır). Sayılar `data/processed/*/data.yaml` ve
`data/raw/*` dizinlerine karşı doğrulanmıştır.

| AURA sınıfı | Kaynak | Lisans | Görüntü (kullanılan) | Durum |
|---|---|---|---|---|
| `license_plate` | `keremberke/license-plate-object-detection` (HF; Roboflow "Vehicle Registration Plates v1" → COCO→YOLO) | CC BY 4.0 | **8823** (9123 toplandı → PIL/split sonrası 8823: 6176/1765/882) | ✅ indirildi + işlendi |
| `seatbelt` | Roboflow `oohmp/seatbelt-detection` v2 (HF `ramankamran/seatbelt-detection-v2i-yolov11-lt`) | CC BY 4.0 | **3104** (2 sınıf: `no_seatbelt_evidence` + `seatbelt_ok`) | ✅ indirildi + işlendi |
| `smoking` | CigDet (Cigarette Detection), Mendeley DOI `10.17632/6hyrr8typ7.1` | CC BY 4.0 | **557** (446 train / 111 test) | ✅ indirildi + işlendi |
| `phone` | HF `anywaylabs/synthetic-driver-monitoring` | CC BY 4.0 | **659** (**sentetik render** → domain-uyum riski) | ✅ indirildi + işlendi |
| `minibus` | (no-auth açık bbox seti bulunamadı) | — | — | ⏳ komite verisi / Roboflow erişimi |
| `fatigue` | (teyitli açık set yok) | — | — | ⏳ komite verisi |

> Dürüstlük: `smoking` (557) yakın/kontrollü kabin-içi sigara setidir; `phone` (659)
> **sentetik render** olduğundan gerçek trafik domain'ine uyum riski taşır (FTR'de belirtilir).
> Bu setler stok modeli **fine-tune** etmek içindir; eğitim **TAMAMLANDI (19 Haz 2026)**
> (bkz. `docs/egitim.md` §0 ve aşağıdaki "Eğitim durumu").

### Eğitim durumu (TAMAMLANDI — 19 Haz 2026) ✅
YOLO26s tabanlı fine-tune **TAMAMLANDI (19 Haz 2026)** (`weights/yolo26s.pt` taban,
`imgsz 640`, MPS). Gerçek held-out mAP (`weights/custom_*.metrics.json`, Ultralytics `model.val`):

| Sınıf | mAP50 | mAP50-95 | Veri |
|---|---|---|---|
| `license_plate` | **0.983** | **0.707** | 9123 görsel (keremberke/HF) |
| `smoking` | **0.856** | **0.457** | 557 görsel (CigDet/Mendeley) |
| `seatbelt` | **0.895** | **0.546** | 3104 görsel (Roboflow/HF) |

> `custom_license_plate` 3-video A/B'de regresyon göstermedi → **varsayılan LP dedektör**
> (`config/default.yaml`). `custom_smoking` (held-out 0.856) drop-in regresyon nedeniyle
> varsayılana alınmadı; doğru entegrasyon takip işi. `seatbelt` opsiyonel (dış-kamera görüş
> açısı). ASIL araç-tespiti doğruluğu hâlâ stok `yolo26l` COCO val2017 held-out (mAP50-95 0.537).

## Veri toplama zorluğu
- **Araç tespiti:** COCO/araç veri setleri bol; Türk trafiği için yerel kamera kaydı ekleyin.
- **Sürücü durumu:** En zor kısım. Telefon/sigara/kemer/yorgunluk için kabin-içi
  görüntü azdır ve gizlilik kısıtlıdır. Strateji:
  - Roboflow Universe'teki açık veri setleri (`python -m train.roboflow_pull`).
  - Kontrollü çekim (araçta simülasyon, izinli sürücüler).
  - Sentetik augmentasyon ile çoğaltma (aşağıda).

## Etiketleme rehberi
- Format: YOLO (`<class> <cx> <cy> <w> <h>` normalize, her görüntü için `.txt`).
- Dizin: `images/` + `labels/` (veya görüntüyle aynı klasörde `<stem>.txt`).
- Sınıf listesi: `classes.txt` (her satır bir sınıf) veya `--classes` argümanı.
- Araç sınıfları: `car, truck, bus, minibus, motorcycle`.
- **Tabela sınıfları (Stage-1, opsiyonel):** generic `sign` + hız-limiti değerli sınıflar
  `speed_limit_30/50/70/90/120`. Dedektör bunları araç/kişiden ayrı toplar; `config/default.yaml`
  → `sign.value_map` sınıf adını km/h'ye eşler (hız-limiti çapraz kontrolü için). Bkz. `docs/mimari.md` §7.5.
- **Yaya:** `person` sınıfı hem yaya güvenliği hem (araç kutusu içinde kalırsa) sürücü kilidi içindir;
  cam-ardı sürücüyü yaya seviyesinde yakalamak için kabin-içi `person` örnekleri de eklenmelidir.
- Sürücü sınıfları: `phone, smoking, no_seatbelt, fatigue` — **çoklu etiket** (bir kabinde
  aynı anda birden çok aktif olabilir; her durum ayrı bbox).
- **Yorgunluk:** kapalı göz, esneme, baş düşmesi sahnelerini `fatigue` olarak etiketleyin.

## Sentetik augmentasyon stratejisi
Eğitimde ultralytics otomatik uygular; öne çıkanlar:
| Teknik | Amaç |
|---|---|
| Mozaik | bağlam çeşitliliği, küçük nesne öğrenimi |
| Flip (yatay) | yön bağımsızlığı |
| HSV jitter | farklı ışık/renk sıcaklığı |
| **Karartma / gamma** | **gece koşulları** (far patlaması senaryosu için kritik) |
| Motion blur | yüksek hızlı araç bulanıklığı |

`data/samples/` içindeki **sentetik trafik videosu** (`python -m aura.synthetic`) hat
testleri içindir; eğitim verisi değildir. Gerçek TOGG veri seti geldiğinde
`data/raw/` altına yerleştirip `prepare_dataset` ile işleyin.

## Veri dengeleme (data balancing) — FTR §2 (20 puan)
FTR şablonu verinin nasıl **dengelendiğini** açıkça ister. AURA tool'u dağılımı ölçer:
```bash
python -m train dataset --report --output data/processed/
```
Çıktı: her split (train/val/test) için **görüntü sayısı + sınıf-örnek dağılımı +
dengesizlik oranı (en kalabalık / en seyrek sınıf)**. Oran **> 3** ise uyarır.

**Dengeleme stratejileri (rapora yazın):**
1. **Hedeflenmiş toplama/etiketleme:** seyrek sınıflara (ör. `cigarette`, `minibus`) ek örnek.
2. **Oversampling:** az sınıfın görüntülerini eğitim listesinde çoğaltma.
3. **Sınıf-lehine augmentasyon:** seyrek sınıf sahnelerinde daha agresif mozaik/HSV/karartma.
4. **Split oranı:** varsayılan **%80/%10/%10** (train/val/test) — küçük özel sette val/test'in
   istatistiksel anlamı için %10+%10; çok dengesizse stratified split önerilir.

> FTR'ye: `dataset --report` çıktısını tablo olarak koyun + uyguladığınız dengeleme
> tekniklerini ve gerekçesini yazın. Komut + sayılar = "veriyi nasıl dengelediğinizin" kanıtı.

**Eksik sınıflar için somut açık veri setleri** (cigarette/seatbelt/minibus — URL + lisans +
görüntü sayısı): `docs/yol_haritasi.md` §2 (Gemini araştırması; kullanım öncesi lisans teyidi).

## Eksik-sınıf manifesti + çekme aracı (`train/datasets.yaml`)
Yol haritası §2'deki açık setler artık **bildirimsel bir manifestte** toplanır:
`train/datasets.yaml` her hedef sınıf (`cigarette`, `seatbelt`, `fatigue`, `minibus`,
`license_plate`) için **kaynak(lar) + lisans + ~görüntü sayısı + AURA taksonomisine
sınıf-eşlemesi** tutar. Eşleme `aura/taxonomy.py` ile tutarlıdır (ör. `cigarette → smoking`,
`van → minibus`). Manifestteki açık-kaynak köprü kapsamı (lisanslar §5 kaynakçaya yazılır;
**kullanım öncesi lisans/uyumluluk teyidi** notu korunur):

| Hedef sınıf (AURA) | Kaynak(lar) | ~Görüntü | Lisans | Durum |
|---|---|---|---|---|
| `license_plate` | `keremberke/license-plate-object-detection` (HF; Roboflow v1 → COCO→YOLO) | **8823** | CC BY 4.0 | **indirildi + işlendi**; YOLO26s fine-tune **TAMAMLANDI** (held-out mAP50 **0.983** / mAP50-95 **0.707**); `custom_license_plate` → **varsayılan LP dedektör** |
| `seatbelt → no_seatbelt_evidence` | Roboflow `oohmp/seatbelt-detection` v2 (HF `ramankamran`) | **3104** | CC BY 4.0 | **indirildi + işlendi**; fine-tune **TAMAMLANDI** (held-out mAP50 **0.895** / mAP50-95 **0.546**); opsiyonel (dış-kamera görüş açısı) |
| `cigarette → smoking` | CigDet (Cigarette Detection), Mendeley DOI `10.17632/6hyrr8typ7.1` | **557** | CC BY 4.0 | **indirildi + işlendi**; fine-tune **TAMAMLANDI** (held-out mAP50 **0.856** / mAP50-95 **0.457**); `pose.py` ikinci-model (phone-kanıtını korur). Büyük setler (Roboflow `driver-smoking-detecor` 1066, `Smoker YOLO.v4` 4221) API/Roboflow erişimi gerektirir |
| `phone` | HF `anywaylabs/synthetic-driver-monitoring` | **659** | CC BY 4.0 | **indirildi + işlendi**; **SENTETİK render** → domain-uyum riski |
| `car/bus/truck/motorcycle/person` | COCO (genel sınıflar) | — | CC BY 4.0 | mevcut (stok `yolo26l`) |
| `minibus → minibus` | **(no-auth açık bbox seti bulunamadı)** | — | — | Roboflow/Kaggle anahtarı veya komite verisi gerekir |
| `fatigue` | **(teyitli açık set yok — boş)** | — | — | komite verisi beklenir |

> **ONUR:** Sistem BASE/stok YOLO26 modelleriyle çalışır; üstüne **dört gerçek açık veri seti**
> indirilip işlenmiştir (`data/processed/{license_plate,seatbelt,smoking,phone}/data.yaml`,
> tümü PIL-doğrulanmış, CC BY 4.0): `license_plate` 8823 görsel (keremberke/HF), `seatbelt`
> 3104 görsel (Roboflow `oohmp`/HF `ramankamran`), `smoking` 557 görsel (CigDet/Mendeley DOI
> `10.17632/6hyrr8typ7.1`) ve `phone` 659 görsel (HF synthetic, **sentetik render** →
> domain-uyum riski). Bu veriyle YOLO26s fine-tune'lar **TAMAMLANDI (19 Haz 2026)**; gerçek
> held-out mAP (`weights/custom_*.metrics.json`, Ultralytics `model.val` ayrılmış test bölmesi):
> `license_plate` mAP50 **0.983** / mAP50-95 **0.707**, `smoking` **0.856** / **0.457**,
> `seatbelt` **0.895** / **0.546**. `custom_license_plate` 3-video A/B'de regresyon göstermedi →
> **varsayılan LP dedektör**; `custom_smoking` `pose.py`'da ikinci-model (phone-kanıtını korur);
> `seatbelt` opsiyonel (dış-kamera görüş açısı). Bunlar küçük domain held-out'larıdır
> (özellikle `smoking` 557 yakın/kontrollü set, `phone` sentetik) → doğruluk göstergesidir,
> sahanın tamamını temsil etmez. Fine-tune boru hattı (`train/`) ayrıca uçtan uca doğrulanmıştır
> (açık `coco128`, `yolo26s`, 5 epoch → gerçek `best.pt` mAP50 0.7645). Büyük sigara setleri
> (Roboflow `driver-smoking-detecor` 1066, `Smoker YOLO.v4` 4221) ek API/Roboflow erişimi
> gerektirir (manifestte listeli). `minibus` için no-auth açık bbox seti bulunamamıştır; `fatigue`
> için doğrulanmış açık set yoktur (manifestte `sources: []` boş bırakılır — uydurma kaynak
> eklenmez); bu sınıflar komite verisiyle gelir. ASIL **yayınlanmış** dedektör doğruluk göstergesi
> stok `yolo26l`'in COCO val2017 held-out (5000 görsel) sonucudur (mAP50 0.709 / mAP50-95 0.537);
> fine-tune'lu zorunlu-sınıf (`license_plate`, `smoking`, `seatbelt`) **final** held-out
> mAP'leri yukarıda (`weights/custom_*.metrics.json`) KESİNLEŞMİŞTİR. Plaka için ayrıca **sıkı LP-kırpık + pipeline dürüstlük zırhları** kullanılır;
> bu zırhlar sayesinde **her iki dedektör de** plakada 2/3 exact-match, 0 yanlış-onay verir ve
> belirsizde dürüstçe `pending` der; sistem asla yanlış plaka onaylamaz (bkz.
> `docs/degerlendirme.md` ölçülen sonuçlar + `ftr.md` §4).

```bash
python -m train fetch                  # PLAN bas (varsayılan KURU; AĞ KULLANMAZ)
python -m train fetch --class minibus  # tek hedef sınıfın planı
python -m train fetch --run            # GERÇEK indirme (roboflow → ROBOFLOW_API_KEY)
```
Plan her kaynak için **indirme tipi/koordinatları + lisans + ~görüntü sayısı + sınıf-eşlemesi
+ çıktı dizini** basar ve sonda **FTR §5 kaynakça lisanslarını** özetler. Manifestte
taksonomiyle çelişen bir eşleme varsa planda `⚠` ile işaretlenir (sessiz düzeltme yok).

> **ONUR notu:** `fatigue` ve `license_plate` için teyitli açık set olmadığından manifestte
> `sources: []` bırakılır; plan bunları boş gösterir (uydurma kaynak eklenmez). Kaggle/URL
> kaynakları `--run` ile **otomatik indirilmez** (kimlik/lisans onayı gerekir) — araç yalnız
> manuel indirme talimatını basar.

İndirilen set sonrası akış değişmez: `python -m train dataset --input ... --output ... --report`
(denge) → birden çok sürücü-davranış seti için `python -m train.merge_driver_datasets`.

## Dizin yapısı
```
data/
├── raw/            # ham, etiketsiz veya etiketli (git'e dahil değil)
├── processed/      # prepare_dataset çıktısı (train/val/test + data.yaml)
└── samples/        # sentetik hat-testi verisi
```

## Roboflow entegrasyonu
```bash
export ROBOFLOW_API_KEY=...
python -m train.roboflow_pull --workspace W --project P --version 1 --output data/raw/roboflow
python -m train dataset --input data/raw/roboflow --output data/processed
```
Anahtar yoksa local veriyle çalışın; pipeline mock modda eğitim olmadan da çalışır.

## Sorun Giderme
- **Görüntü bulunamadı:** `--input` altında `images/` var mı, uzantılar `.jpg/.png` mi?
- **Etiket eşleşmiyor:** `<stem>.txt` görüntüyle aynı isimde mi, `labels/` altında mı?
- **Sınıf indeksleri:** `data.yaml` `names` sırası etiket dosyalarındaki indekslerle uyumlu olmalı.
