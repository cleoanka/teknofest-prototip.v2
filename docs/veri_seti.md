# Veri Seti Stratejisi

## Ne yapar
AURA modellerini eğitmek için veri toplama, etiketleme ve sentetik augmentasyon
stratejisini tanımlar. İki ayrı veri seti: araç tespiti (Stage-1) ve sürücü durumu (Stage-2).

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
| `seatbelt → no_seatbelt_evidence` | `ramankamran/seatbelt-detection` (HF / Roboflow `oohmp`) | 3104 | CC BY 4.0 | **indirildi + kullanıldı** (denge 1.27); özel YOLO26 (s+l) fine-tune devam ediyor |
| `car/bus/truck/motorcycle/person/phone` | COCO (genel sınıflar) | — | CC BY 4.0 | mevcut |
| `cigarette → smoking` | **(no-auth açık bbox seti bulunamadı)** | — | — | Roboflow/Kaggle anahtarı veya komite verisi gerekir |
| `minibus → minibus` | **(no-auth açık bbox seti bulunamadı)** | — | — | Roboflow/Kaggle anahtarı veya komite verisi gerekir |
| `fatigue` | **(teyitli açık set yok — boş)** | — | — | komite verisi beklenir |
| `license_plate` | **(teyitli açık set yok — boş)** | — | — | komite verisi beklenir |

> **ONUR:** `seatbelt` için gerçek bir açık-kaynak YOLO seti (3104 görsel, CC BY 4.0) indirilip
> kullanılmış ve üzerinde özel YOLO26 (s+l) fine-tune EĞİTİMİ başlatılmıştır (held-out mAP eğitim
> bitince eklenecek). `cigarette` ve `minibus` için kimlik-doğrulaması gerektirmeyen (no-auth)
> açık bbox seti bulunamamıştır (Roboflow/Kaggle anahtarı veya komite verisi gerekir). `fatigue`
> ve `license_plate` için doğrulanmış açık set bulunamadığından bunlar manifestte `sources: []`
> boş bırakılır (uydurma kaynak eklenmez); bu sınıflar komite verisiyle gelir. v4 fine-tune'da
> (`yolguvenligi_types_v4`) `license_plate/cigarette/seatbelt/headphone` için eğitim verisi yoktu
> → bu sınıfların mAP'i güvenilir değildir; plaka için stok dedektör yerine **sıkı LP-kırpık +
> pipeline dürüstlük zırhları** kullanılır. Bu zırhlar sayesinde **her iki dedektör de** plakada
> 2/3 exact-match, 0 yanlış-onay verir ve belirsizde dürüstçe `pending` der; sistem asla yanlış
> plaka onaylamaz (bkz. `docs/degerlendirme.md` ölçülen sonuçlar + `ftr.md` §4).

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
