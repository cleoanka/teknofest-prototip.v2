# RoadGuard — Yenileme ve Sağlamlaştırma Planı (12 Haziran 2026)

> ⚠️ **ARŞİV / LEGACY (v2.1, 12.06.2026).** Bu plan v2.1 yenileme oturumunundur ve
> **tarihsel kayıt** olarak korunur (tüm maddeleri o gün uygulandı). İçeriği o güne aittir:
> OCR motoru o sırada **EasyOCR**'dı (artık **fast-plate-ocr**, plaka 3/3 exact / CER 0),
> dedektör varsayılanı sonradan **yolo26l** oldu, test sayısı 99/118 idi (güncelde **~600**;
> servis testleri sürüyor) ve FTR son teslimi 14.06 yazıyor (**28.06'ya ertelendi**).
> **Aktif/güncel planlar:** `plan1.md` (teknik yol haritası) · `plan2.md` (görev dağılımı) ·
> `ultraplan.md` (W1 yürütme). Güncel durum: `gozlem.md` · `CHANGELOG.md`.
>
> **Bu belge ne?** 12 Haziran 2026 gecesi yapılan otonom bakım/geliştirme oturumunun
> ana planı. Repo'nun önceki inşa planı `docs/plan_insa_v2.md`'ye arşivlendi
> (M1–M16 milestone'ları orada). Bu plan üç kaynaktan damıtıldı:
> **v1 prototip** (`teknofest-prototip` — 343 testli, gerçek videolarla ölçülmüş),
> **hidden_prototip** (YOLO26 cascade + gerçek video test dersleri) ve
> **v2/RoadGuard**'nın kendisi. Uygulanan her madde işaretlendi; sonuçların tam dökümü
> için `CHANGELOG.md`'ye bakın.

---

## 0. Durum Tespiti — "Neden Çalışmıyordu?"

Gece başlangıcında saptanan somut arızalar (hepsi kod/araçla doğrulandı):

| # | Arıza | Kanıt | Kök Neden |
|---|---|---|---|
| D1 | `bootstrap.py` paket kurulumunda çöküyordu | `pip install -e .[dev]` → `ruff (from versions: none)` | PyPI'a geçici ağ/önbellek hatası; bootstrap'te yeniden-deneme yok |
| D2 | Ağırlık indirmeleri yarıda kesiliyordu | `Connection reset by peer` / `read timed out` | Tek denemeli indirme; `weights/` boş kalınca pipeline sessizce mock'a düşüyor |
| D3 | `--source` ile verilen gerçek video, `ai_mode: auto`'da **mock**'ta işlenebiliyordu | `roadguard/__main__.py` `--source`'u cfg'ye yazmıyor; `resolve_ai_mode` config'teki sentetik örneğe bakıyor | CLI argümanı ile config'in kopukluğu |
| D4 | `--device auto` macOS'ta MPS'i hiç seçmiyordu | `roadguard/device.py` auto yolu yalnız CUDA'yı deniyor | Eksik MPS dalı → 4K videoda CPU'da sürünme |
| D5 | Sürücü davranışı (telefon/sigara) eventi **hiç** üretilmiyordu | Stok COCO `yolo26l`'de sınıf adları `cell phone` vb.; config `phone/smoking/...` bekliyor | Eğitilmiş driver-state ağırlığı yok + sınıf adı eşleşmesi sessizce boş dönüyor |
| D6 | Plaka hiç `confirmed` olamıyordu | 150-kare koşuda 11 × `PLATE_REJECTED`; oylar `041C8532 / 34TC8532 / 8532` varyantlarına dağılıyor | OCR karakter hataları (3→0, T→1) düzeltilmeden ham metinle oylama; 7'lik buffer her turda sıfırlanıyor |
| D7 | Swerving/dikkatsiz sürüş tespiti **yok** | Kod taraması: yanal hareket hiçbir yerde izlenmiyor (`SpeedEstimator` yalnız dikey `cy` saklıyor) | Modül hiç yazılmamış |
| D8 | Git karmaşası | Ev dizininde kazara `git init` edilmiş remote'suz repo; v1'de LFS eksikliği yüzünden yarım kalan checkout (136 staged-deletion); v2'de merge edilmiş 2 eski remote branch | `git-lfs` makinede kurulu değildi |
| D9 | v1'in eğitilmiş modeli (`yolguvenligi_types_v4.pt`, held-out mAP50 .788) diskte yoktu | `models/` boş; LFS pointer çekilememiş | D8 ile aynı |

---

## 1. Hedefler

1. **Çalışırlık:** `setup → run → video işle` zinciri tek komutla, gerçek modda, macOS/MPS'te sorunsuz.
2. **Üç test videosunda gerçek tespit** (hile/hardcode YOK — K-004 ilkesi: tüm eşikler oran-bazlı/ölçek-bağımsız, tek videoya özel sabit yok):
   - video_1 → sürücü **sigara** kullanıyor
   - video_2 → sürücü **telefonla** konuşuyor
   - video_3 → **swerving** (dikkatsiz sürüş)
   - hepsinde → plaka (son 4 hanesi **8532**) okunmalı
3. **Üç prototipin en iyi fikirlerinin birleşimi** (aşağıda §4–§5) + kendi eklediklerim.
4. **Git hijyeni:** tüm karmaşa çözülmüş, her şey `teknofest-prototip.v2` main'de, CI yeşil.
5. **Belgeler:** her yeni özellik `--help`'li, README/CLI referans/CHANGELOG güncel.

---

## 2. Git Onarımı

- [x] **Adli inceleme:** dört repoda (`v1`, `v2`, `hidden_prototip`, `~/roadguard`) reflog + fsck + unpushed taraması → **kayıp/pushlanmamış commit YOK**. `~/roadguard` (M1–M16) geçmişi v2'nin içinde birebir mevcut (ae8e62d v2'de ata).
- [x] **Ev dizini stray reposu:** `~/.git` (tek commit'lik kazara repo, remote'suz) güvenli yedeğe taşınır → `~/stray-home-git-backup.git`. Ev dizini artık repo değil.
- [x] **git-lfs kurulumu** (brew) + `git lfs install`.
- [x] **v1 onarımı:** index reset + çalışma ağacı restore + `git lfs pull` → 136 dosya ve **yolguvenligi_types_v4.pt** (52 MB, sha256 doğrulandı) geri geldi.
- [x] **v2 eski branch temizliği:** `fix/black-docstring-blank-line` ve `fix/dashboard-stream-bbox` main'e merge edilmiş (merge-base doğrulaması) → remote'tan silinir.
- [x] **Tüm değişiklikler v2 main'e** mantıklı commit'ler halinde push edilir; CI doğrulanır.

## 3. Kurulum Onarımı

- [x] `bootstrap.py` indirmelerine **yeniden-deneme** (3 deneme, artan bekleme) + `.part` artıklarının temizliği.
- [x] Ağırlıklar hazır: `yolo26s.pt` (20 MB), `yolo26l.pt` (53 MB), `yolo26s-pose.pt` (24 MB, **yeni**), `yolguvenligi_types_v4.pt` (52 MB, v1'den).
- [x] `bootstrap.py` ağırlık listesine pose modeli eklenir; v4 modeli için "komşu repodan kopyala" fallback'i.
- [x] 99 unit test yeşil; `ruff` + `black` temiz.

## 4. Çekirdek Hata Düzeltmeleri (P0)

- [x] **D3:** `roadguard/__main__.py` → `--source` değeri `cfg.runtime.source`'a yazılır (auto kararı artık gerçek kaynağa göre). Aynı düzeltme `services/inference_api` `StreamManager.start()` yoluna.
- [x] **D4:** `roadguard/device.py` → `auto`: CUDA → **MPS** → CPU sırası (Apple Silicon'da `torch.backends.mps.is_available()`).
- [x] **Ağırlık yolu CWD-bağımsız:** `detection/yolo.py`, `driver_state/yolo.py` model yolunu repo köküne göre çözer (hidden_prototip `_resolve_model_path` deseni).
- [x] **Sınıf adı takma-isim haritası:** `cell phone→phone`, `cigarette→smoking` vb. — stok/fine-tune model adları tek noktada normalize edilir (hidden'ın iki-uzaylı taksonomi dersi).

## 5. Yetenek Eklemeleri (üç prototipin sentezi + yeni fikirler)

### 5.1 Fine-tune dedektör entegrasyonu (v1'den)
- [x] `weights/yolguvenligi_types_v4.pt` (car/minibus/bus/truck/motorcycle/person/phone, held-out mAP50 .788) **birincil dedektör** yapılır (`config/default.yaml` `models.detector.path`); dosya yoksa stok `yolo26s.pt`'ye sessiz değil **loglu** fallback.
- [x] Eşik rehberi (hidden dersi): fine-tune model → conf 0.30+; ön-eğitimli → conf 0.05–0.10 + conf-bağımsız FP filtreleri. Config yorumlarına işlenir.

### 5.2 Pose-tabanlı sürücü davranışı (YENİ — v1 geometrisinin YOLO26'ya portu)
- v1 sigara/telefonu MediaPipe geometrisiyle çözmüştü (sigara recall %59, telefon %61, FP %0 — ölçülmüş). MediaPipe Python 3.13'te yok **ve** v2 mimari kararı "MediaPipe yasak" diyor. Çözüm: **`yolo26s-pose.pt`** (COCO 17 keypoint) ile aynı kanıtlanmış geometri — mimari karar korunur (saf YOLO26).
- [x] Yeni modül `roadguard/driver_state/pose.py`: sürücü ROI'sinde pose → bilek-ağız / bilek-kulak **göreli yakınlık** kıyası (v1 K-012 dersi: mutlak eşik değil oran; `d_ear < 0.40×yüz-genişliği` telefon, `d_mouth < d_ear` ve ağıza yakın → sigara adayı). **Uygulamada eklenen kural:** kulak keypoint'i görünmüyorsa karar YOK (dürüst çekimserlik) — kulaksız geometri video_2'de telefonu sigara sanıyordu.
- [x] **Hibrit ROI nesne kanıtı (uygulamada eklendi):** v4 dedektörü sürücü ROI'sinde de koşar; `phone` NESNESİ tespiti geometriden üstündür (hoparlörde konuşma = el ağız önünde → geometri tek başına 'sigara' derdi). Gerçek video_2 ölçümü: telefon bilek keypoint'i conf 0.21 (görünmez) ama v4 phone nesnesi 0.35-0.37 ✓.
- [x] v1'in **latch** dersi: phone nesnesi son görülmeden sonra 25 kare geçerli sayılır (nesne her karede yakalanmaz; sigara FP'sine dönüşmez). Sustain karşılığı pipeline'daki 16/8 süzgeci.
- [x] v4 dedektörünün `phone` sınıfı tam-kare tespitleri de araca düşüyorsa **füzyon** edilir (`fuse_detections`).
- [x] Config: `models.driver_state.backend: pose | yolo | auto` (+ tüm eşikler config'te, `--help` ve README'de açıklamalı).
- [x] ROI ön-işleme (v1 dersi): kısa kenarı 320px'e büyüt + gamma/CLAHE parlatma → camın arkasındaki sürücüde keypoint bulunabilirliği artar.

### 5.3 Swerving / dikkatsiz sürüş tespiti (YENİ — v1 fikrinin ölçek-bağımsız hali)
- [x] `roadguard/speed/estimator.py` yanal (cx) geçmiş tutar. **Nihai algoritma (3 iterasyonda rafine edildi):** ZigZag ekstremum sayacı — seri mevcut uç noktadan `amp_ratio × O ANKİ araç genişliği` kadar geri dönünce yön-değişimi sayılır; ≥`min_flips` (2) → `SpeedState.swerving`. Monoton hareket (yaklaşma perspektif kayması, tek şerit değişimi) yapısal olarak 0 üretir; pencere saniye cinsinden (fps-bağımsız). İlk iki deneme (adım-bazlı gürültü kapısı; doğrusal/parabolik trend çıkarma) gerçek 50fps verisinde sinyali kaçırdı/S-eğrisinde FP verdi — sentetik beş yörünge şekli + 3 gerçek video yörüngesiyle doğrulandı (video_3: 3 dönüş ✓, video_1/2: 0 ✓).
- [x] 16/8 kararlılık süzgecinden geçirilir; `Accumulator._cond`'a `speed.swerving` tokenı; `config` risk kuralı `swerving_vehicle` [high] → `RISK_ALERT`.
- [x] QoD tetikleyicisine swerving kritik koşulu eklenir.

### 5.4 Sigara risk kuralı (config-only boşluk)
- [x] `risk.rules`'a `smoking_driver = all_of:[driver.smoking]` [medium] eklenir → sigara artık `RISK_ALERT` üretir (önceden yalnız DRIVER_STATE'te gömülüydü).

### 5.5 Plaka hattı güçlendirme (v1 + hidden sentezi)
- [x] **Pozisyon-farkında karakter düzeltme** (v1 `plate_ocr.py`'den): TR plaka şablonuna göre rakam-bloklarında `O→0, I/L→1, B→8, S→5, Z→2, T→7…` değil — *blok bazlı*: il kodu (ilk 2) rakam, orta blok harf, son blok rakam; `041C8532→34TC8532` türü hatalar normalize edilir.
- [x] **Track-ömrü frekans oylaması** (v1 `PlateTracker` dersi): 7'lik buffer her redde sıfırlanmak yerine oylar track boyunca birikir; karakter-düzeyi çoğunluk oyu (aynı-uzunluk okumalar hizalanıp pozisyon başına mod alınır) tek-karakter hatalarını söndürür.
- [x] **Kısmi plaka raporu**: konsensüs tam-regex'i geçemese bile en güçlü aday `plate_partial` olarak TrackRecord + annotation'a yazılır (jüri kanıtı: "8532" bile görünür kalır).
- [x] **Conf-bağımsız FP filtreleri** (hidden dersi): plaka ROI'sinde aspect-ratio (1.5–9.5) + parlama testi (mean>210 & std<30 → far/ışık) OCR öncesi uygulanır.
- [x] EasyOCR'a çok-varyant ön-işleme (CLAHE + 2× upscale) — küçük/karanlık plakada okuma sayısını artırır; düşük güvenli okumada ikinci varyant oy havuzuna ek kanıt verir.
- [x] **Sıkı plaka kırpma (uygulamada eklendi):** özel LP dedektörü (`lp_yolo11n.pt`, HuggingFace, ~5MB — v1'in kanıtlanmış yolu) plakayı araç-altı geniş crop içinde bulup sıkı kırpar; OCR karakter doğruluğu belirgin arttı (gerçek ölçüm: geniş crop hiç doğru "34TC8532" üretmezken sıkı kırpma üretti).
- [x] **Oylar OCR güveniyle ağırıklanır** ve karar yalnız ikamesiz format-geçerli ham okumalarla verilir (min ağırlık + ikinciye 1.5 fark + oran) — karanlık çekimdeki sistematik ilk-karakter hatası (3→0/2) YANLIŞ onaya dönüşemez; aday `partial` alanında kanıt olarak taşınır.
- [x] Sweet spot x aralığı genişletildi (0.18–0.85): yanal/çapraz yaklaşan araç (video_3) eski 0.30–0.70 bölgesine hiç girmiyordu.

### 5.6 QoD yaklaşma tetiği (şartname boşluğu — FTR izlenebilirlik §1)
- [x] Şartnamenin asıl senaryosu "TOGG aracının **yaklaştığını** algılayınca QoD": bbox alan büyüme oranı sürekli pozitif + alan eşiği → `vehicle_approach` kritik tetiği (`roadguard/qod/client.py` + config). v1 qod_trigger A-koşulunun portu.

### 5.7 Video test aracı (hidden'dan port — jüri/demo kanıtı)
- [x] `tools/test_video.py`: videoyu pipeline'dan geçirir → **annotated mp4** + **JSON özet** (eventler, plaka oyları, sürücü bayrak süreleri, swerving kareleri, FPS). `--help`'li, README'li. Ham plaka kutuları çizilmez (hidden görselleştirme hijyeni), yalnız onaylı sonuçlar.
- [x] `python -m roadguard`'ya `--save-events PATH` (JSONL) bayrağı.

### 5.8 Değerlendirme
- [x] `data/samples/` altına 3 test videosu için GT iskeleti (`video_N_gt.json`) — plaka + davranış etiketi; `roadguard.eval` ile koşulabilir.
- [x] Hız: `value_kmh`'ın güvenilmez olduğu durumda (kalibrasyon ısınmamış / scale_confidence düşük) eventlerde `calibrated:false` etiketi netleştirilir.

## 6. Doğrulama Protokolü (hile yok — K-004)

- [x] Üç videoda uçtan uca koşu; beklenen sonuç matrisi:

| Video | Plaka | Sigara | Telefon | Swerving |
|---|---|---|---|---|
| video_1 | 34TC8532 ✓ | **✓ bekleniyor** | ✗ olmamalı (FP kontrolü) | ✗ |
| video_2 | 34TC8532 ✓ | ✗ | **✓ bekleniyor** | ✗ |
| video_3 | 34TC8532 ✓ | ✗ | ✗ | **✓ bekleniyor** |

- Çapraz-FP denetimi tabloya gömülü: bir davranış yalnız kendi videosunda çıkmalı.
- [x] Eşikler yalnızca **genel** gerekçeyle ayarlanır (model sınıfı, oran-bazlı geometri); video-özel sabit yasak.
- [x] `pytest -m "not integration"` yeşil; yeni modüllere unit test; `ruff` + `black` temiz; CI yeşil.

## 7. Dokümantasyon

- [x] README: yeni ağırlıklar, pose backend, swerving, test_video aracı, eşik rehberi.
- [x] `docs/cli_referans.md` + yeni `--help` çıktıları; `docs/mimari.md`'ye pose/swerving ekleri; `config/README.md` yeni anahtarlar.
- [x] `CHANGELOG.md` 2.1.0 girdisi; `docs/sartname_izlenebilirlik.md`'ye yaklaşma-tetiği/kanıt satırları.
- [x] En son: oturum dökümü `CHANGELOG.md`'ye işlendi.

## 7.5 Uygulama Sırasında Ortaya Çıkanlar (plan-dışı bulgular)

- **Kopya araç kutusu** (v4 aynı araca car+truck çift kutu üretiyor; NMS sınıf-bazlı) → sınıftan bağımsız IoU-dedup eklendi (`models.detector.dedup_iou`); hayalet ByteTrack track'leri 7→2'ye düştü.
- **Ağır aşama kapısı** (`tracking.min_track_frames`): tek-kare hayalet track'ler OCR/pose maliyeti üretmesin.
- **Takım arkadaşı branch'i:** Oturum sırasında Mustafa `feature/stage2-driver-state` pushladı (ID-merkezli driver-state motoru; eski main tabanlı, pipeline'la çakışıyor, işlevi mevcut 16/8 süzgeciyle örtüşüyor). Sonradan main çatısında değerlendirildi (katkı korunarak).
- **Plaka ilk-karakter belirsizliği:** karanlık otopark çekiminde EasyOCR "3"ü sistematik 0/2 okuyor (04/24/34TC8532 yarışıyor). Sistem yanlış onay VERMEZ (margin koruması); doğru sonek `8532` tüm adaylarda, tam aday `partial` alanında raporlanır. Kalıcı çözüm: perspektif düzeltme portu (v1 `plate_crop.py`) veya aydınlık çekim.

## 8. Bilinçli Kapsam Dışı (gerekçeli)

- **FTR raporunun kendisi** (teslim 14.06 17:00): içerik üretimi sahibinin kararlarını gerektirir; bu oturum repoyu rapora hazır hale getirir (metrikler, mimari, kanıt araçları hazır).
- **Model retrain** (cigarette/seatbelt sınıfları): veri etiketleme gerektirir; pose-geometri yolu bu gece ölçülebilir sonuç verir. Retrain altyapısı (`train/`) hazır ve belgelendi.
- **Mobil native build**: Expo iskeleti korunur; şartname finaline kadar gerekmiyor.
- **mediapipe bağımlılığı**: Python 3.13'te yok; bilinçli olarak pose-YOLO ile ikame edildi (mimari karar da korunmuş oldu).
