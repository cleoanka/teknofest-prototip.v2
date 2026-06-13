# fable.md — 12 Haziran 2026 Gece Oturumu: Yapılan Her Şey

> Bu belge, 12 Haziran 2026 gecesi Claude (Fable 5) tarafından otonom yürütülen
> bakım + yenileme oturumunun **tam dökümüdür**. Plan: [`plan.md`](plan.md).
> Sürüm: **2.1.0**. Her madde gerçekten yapıldı ve doğrulandı; hiçbir sonuç
> tek videoya özel sabitle (hile ile) üretilmedi — K-004 ilkesi: tüm eşikler
> oran-bazlı / ölçek-bağımsız / fps-bağımsız.

---

## 1. Görev

Kullanıcının isteği: *"prototip.v2 çalışmıyor, çalıştır; git karmaşasını
(branch/commit/pull/push) çöz; v1'deki iyi fikirleri ekle; üç test videosunda
gerçek tespit yap (plaka 8532, video_1 sigara, video_2 telefon, video_3 swerving
— hile yok); her şeyi main'e pushla; en sona fable.md yaz."*

## 2. Keşif — Üç Prototip + Belgeler Haritalandı

4 paralel inceleme ajanı ile:
- **v2 (AURA)** kod haritası: mock/real karar mantığı, event kuralları, kırılma noktaları.
- **v1 (`~/teknofest-prototip`)**: 343 testli, gerçek videolarla ölçülmüş 16 taşınabilir fikir
  (sustain+latch, sürücü kilidi, PlateTracker frekans oylaması, perspektif+keskinlik kapısı,
  PnP ölçek, metrik hız yığını, swerving, ani fren, QoD A-F, eğitim dersleri K-008/K-013...).
- **hidden_prototip**: YOLO26 cascade, conf eşik rehberi (ön-eğitimli 0.05-0.15 üretir!),
  conf-bağımsız FP filtreleri, plaka kilidi, test_video aracı dersleri.
- **Şartname + FTR şablonu** (PDF'ler okundu): puanlama (%40 YZ + %40 QoD + %20 mimari),
  birebir zorunlu tespitler (plaka, gerçek hız, araç, araç-içi nesne, yorgunluk, sigara,
  telefon, **TOGG yaklaşması**), kanıt yükümlülüğü (4.5), **FTR teslimi: 14.06.2026 17:00**.

## 3. Git Adli İncelemesi ve Temizlik

| Bulgu | İşlem |
|---|---|
| `~/.git` — ev dizini yanlışlıkla git reposu olmuş (5 Haz, tek commit, remote'suz) | `~/stray-home-git-backup.git`'e taşındı (geri alınabilir); ev dizini artık repo değil |
| v1 klonu `git-lfs` yokken yapılmış → checkout yarım, **136 dosya staged-deletion** görünüyordu | `brew install git-lfs` + `git lfs install` + index reset + restore + `git lfs pull` → **v1 tamamen onarıldı** |
| v1'in eğitilmiş modeli `yolguvenligi_types_v4.pt` diskte yoktu (LFS pointer) | LFS'ten çekildi (52 MB, sha256 doğrulandı: `6caaf19f…`) |
| v2'de merge edilmiş 2 eski remote branch | `fix/black-docstring-blank-line` + `fix/dashboard-stream-bbox` silindi (merge-base doğrulamasıyla) |
| `~/aura` reposu (M1-M16) | Geçmişi v2 main'in atası — dokunulmadı, gereksiz |
| Pushlanmamış / kayıp commit var mı? | 4 repoda reflog+fsck+upstream taraması: **YOK** |
| Kullanıcının önemli belgeleri (FTR şablonu, şartname PDF) | v1 köküne dokunulmadı; untracked olarak korunuyor |

⚠️ **Takım arkadaşı branch'i:** Oturum sırasında (13:28) **Mustafa**
`feature/stage2-driver-state` pushladı (ID-merkezli driver-state motoru).
**Main'e merge ETMEDİM**: eski main tabanlı, `pipeline.py`'de bu oturumla
çakışıyor ve işlevi mevcut 16/8 süzgeciyle örtüşüyor. Mustafa'nın yeni main'e
rebase etmesi ve takımın karar vermesi gerekiyor — kodu duruyor, kaybolmadı.

## 4. Kurulum Onarımı ("sonsuz sorun"ların kökleri)

- `pip install -e .[dev]` geçici PyPI hatasıyla çöküyordu → tekrar koşumla aşıldı;
  `bootstrap.py` indirmelerine **3'lü yeniden-deneme + artan bekleme + .part temizliği** eklendi.
- Ağırlıklar tamamlandı: `yolo26s.pt`, `yolo26l.pt`, **`yolo26s-pose.pt` (yeni)**,
  **`lp_yolo11n.pt` (yeni, HuggingFace plaka dedektörü)**, **`yolguvenligi_types_v4.pt`**
  (v1'den; bootstrap artık komşu klondan otomatik kopyalıyor).
- `.env`, sentetik örnek, smoke test ✓. Servisler doğrulandı: `./run.sh` → 8080
  (dashboard+OpenAPI 200), 8081/8082 (mock'lar 200).

## 5. Teşhis Edilen ve Düzeltilen Çekirdek Hatalar

| # | Hata | Düzeltme |
|---|---|---|
| D3 | `--source` config'e yazılmıyordu → `ai_mode=auto` **gerçek videoyu mock'ta** işleyebiliyordu (ana "çalışmıyor" nedeni) | `aura/__main__.py` + `services/inference_api/state.py` |
| D4 | `--device auto` macOS'ta **hiç MPS seçmiyordu** (hep CPU) | `aura/device.py` auto: CUDA→MPS→CPU |
| — | Model yolları CWD'ye bağlıydı | `resolve_repo_path` (repo-kökü çözümleme) |
| — | Stok COCO modeliyle sürücü sınıfları **sessizce hiç üretilmiyordu** (`cell phone` ≠ `phone`) | `aura/taxonomy.py` kanonik ad eşlemesi |
| — | v4 aynı araca **çift kutu** (car+truck) üretip hayalet track'ler doğuruyordu (7 hayalet) | Sınıftan bağımsız IoU-dedup (`dedup_iou: 0.80`) |
| — | Tek-kare hayalet track'ler OCR/pose maliyeti üretiyordu | `tracking.min_track_frames: 3` ağır-aşama kapısı |

## 6. Yeni Yetenekler (üç prototipin sentezi)

1. **Pose-tabanlı sürücü davranışı** (`aura/driver_state/pose.py`):
   YOLO26-pose ile bilek↔ağız/kulak **göreli yakınlık** geometrisi (v1'in gerçek
   videoda ölçülmüş MediaPipe geometrisinin saf-YOLO portu — "landmark kütüphanesi
   yok" mimari kararı korunur, fine-tune ağırlık gerekmez). Kulak görünmüyorsa
   **karar yok** (dürüst çekimserlik). + **Hibrit ROI nesne kanıtı**: v4 sürücü
   ROI'sinde `phone` NESNESİ arar; nesne kanıtı geometriden üstün. + **Bastırma
   latch'i** (assert etmeyen): güçlü telefon kanıtı sigara-geometrisini bastırır
   ama telefon iddiasını ileri taşımaz.
2. **Swerving tespiti** (`aura/speed/estimator.py`): ZigZag ekstremum sayacı —
   pencere saniye cinsinden, genlik o-anki araç genişliği biriminde; monoton
   hareket (yaklaşma, tek şerit değişimi) yapısal olarak 0 üretir.
   → `speed.swerving` risk tokenı → `swerving_vehicle` kuralı → `RISK_ALERT` + QoD.
3. **Plaka hattı**: sıkı LP kırpma (YOLOv11n) + aynı-satır segment birleştirme
   ("34"+"TC"+"8532" artık tek okuma) + parlama testi + CLAHE/2x ikinci varyant +
   **kalıcı, format-öncelikli, OCR-güveni-ağırlıklı oy havuzu**
   (`aura/plate/normalize.py`) + `PlateState.partial` kanıt alanı.
4. **QoD yaklaşma tetiği** (`qod.approach`): bbox alan büyümesi →
   `reason=vehicle_approach` — şartnamenin birincil "TOGG yaklaşınca QoD" senaryosu.
5. **Kanıt araçları** (şartname 4.5): `tools/test_video.py` (annotated mp4 + JSON
   özet: event sayıları, oy dökümü, bayrak süreleri, yörünge, FPS) ve
   `python -m aura --save-events` (JSONL denetim izi).
6. Yeni risk kuralları: `smoking_driver`, `swerving_vehicle`; `SPEED` event'ine
   `swerving` + `calibrated`; annotation'a `swerving`, `plate_partial`,
   `speed_calibrated`; sweet-spot yanal yaklaşımlar için genişletildi.

## 7. Doğrulama — 7 Tur, Hile Yok

Üç 4K/50fps gerçek video (kapalı otopark, TOGG) uçtan uca **7 kez** işlendi;
her turda bir zayıflık ölçülüp **genel** (videoya-özel olmayan) çözümle giderildi:

1. Baseline: araç+takip ✓, OCR ham varyantlara bölünüyor, sürücü eventi 0, hız ~1 km/h.
2. v4+pose: **sigara ✓** ama hayalet truck'lar; → dedup + ağır-aşama kapısı.
3. Kulak-şartı + nesne-hibrit öncesi: video_2 telefonu sigara sanıyordu (kulak yok →
   geometri kör) + plaka yanlış onay riski (04TC8532) → kulak-zorunluluğu +
   güven-ağırlıklı oylama + margin koruması.
4. Hibrit nesne kanıtı: **telefon ✓ (122 kare)**; swerving algoritması 2 kez
   yeniden tasarlandı (adım-kapısı → trend-çıkarma → **ZigZag sayacı**; beş sentetik
   yörünge + üç gerçek yörüngede doğrulandı) → **swerving ✓ (119 kare)**.
5-6. Latch regresyonları ölçüldü (assert eden latch FP amplifikasyonu yapıyor) →
   **bastırma-latch** tasarımına dönüldü.
7. **NİHAİ MATRİS** (çapraz yanlış-pozitif SIFIR):

| Video | GT | Plaka | Sigara | Telefon | Swerving | QoD yaklaşma |
|---|---|---|---|---|---|---|
| video_1 | sigara | partial **34TC8532** | **✓** (23 kare + 2 RISK_ALERT) | 0 ✓ | 0 ✓ | ✓ |
| video_2 | telefon | **34TC8532 CONFIRMED** | 0 ✓ | **✓** (122 kare) | 0 ✓ | ✓ |
| video_3 | swerving | partial (sonek 8532) | 0 ✓ | 0 ✓ | **✓** (119 kare + RISK_ALERT) | ✓ |

Dürüst notlar:
- **Plaka ilk karakteri**: karanlık otopark çekiminde EasyOCR "3"ü sistematik
  0/2 okuyor; sistem **yanlış onay vermez** (margin koruması), doğru sonek `8532`
  tüm adaylarda, video_2'de tam plaka onaylandı. Kalıcı iyileştirme: v1
  `plate_crop.py` perspektif-düzeltme portu (yol haritasında).
- **video_1 sigara recall'u**: v4, sigara tutan eli ara ara "phone" sanıyor
  (cigarette sınıfının eğitim verisi yok); bastırma-latch gerçek sigara karelerinin
  bir kısmını da törpülüyor (94→23). Tespit ve risk alarmı sağlam; kökten çözüm
  cigarette sınıfına veri toplamak (`train/` hazır).
- Hız `value_kmh` otopark sahnesinde kalibre değil — sistem değer uydurmuyor
  (`calibrated:false` etiketi), tasarım gereği.

## 8. Kalite

- **118 unit test** yeşil (`pytest -m "not integration"`); yeni testler:
  `test_plate_normalize` (9), `test_swerving` (4), `test_taxonomy` (4).
- `ruff` + `black` temiz; CI (GitHub Actions) pushlarla yeşil doğrulandı.
- Dokümantasyon güncel: README, CHANGELOG (2.1.0), `docs/mimari.md`,
  `docs/sartname_izlenebilirlik.md` (TOGG yaklaşması 4b, swerving 3c, kanıt 9b),
  `docs/cli_referans.md`, `config/README.md`; eski inşa planı `docs/plan_insa_v2.md`.

## 9. Commit Dökümü (main'e pushlandı)

```
dc46cdc fix: gerçek video koşumunu engelleyen üç çekirdek hata (D3/D4/CWD)
4fd4784 feat: üç prototipin sentezi — gerçek videoda doğrulanmış tespit çekirdeği (v2.1.0)
5450756 feat: gerçek video test aracı + GT iskeletleri (şartname 4.5 kanıt izi)
c6dfb8c docs: 12 Haziran yenileme oturumu — plan, mimari, izlenebilirlik, referanslar
+ bu dosya (fable.md) — son commit
```

## 10. Nasıl Çalıştırılır

```bash
./setup.sh                                                  # tek komut kurulum
./run.sh                                                    # dashboard: http://localhost:8080/
.venv/bin/python tools/test_video.py --source ~/video_1.mp4 --device mps
# → eval_results/video_1_annotated.mp4 + video_1_summary.json (kanıt)
```

## 11. Sıradaki İşler (öneri, öncelik sırasıyla)

1. **FTR raporu** (⏰ 14.06 17:00, KYS): tüm metrik/mimari/kanıt malzemesi hazır —
   `docs/sartname_izlenebilirlik.md` + `eval_results/*_summary.json` + bu dosya.
2. Mustafa'nın `feature/stage2-driver-state` branch'i: yeni main'e rebase +
   takım kararı (işlevi 16/8 ile örtüşüyor; testleri değerli).
3. `cigarette`/`seatbelt`/`fatigue` sınıflarına veri toplayıp v5 fine-tune
   (`train/` + `docs/egitim.md` hazır) — sigara recall'u ve kemer/yorgunluk için kökten çözüm.
4. Plaka perspektif-düzeltme + keskinlik kapısı portu (v1 `plate_crop.py`).
5. Aydınlık/açık-hava çekimle hız kalibrasyonu doğrulaması (GT hızlı veri komiteden gelince).

*Gece boyu tüm kararların gerekçeleri ve ara ölçümler `plan.md` §0/§7.5'te;
bu oturum "ölç → düzelt → yeniden ölç" döngüsüyle yürütüldü (v1'in ölçüm-önce kültürü).*

---

# 13 Haziran 2026 — Geri Bildirim Turu (v2.2.0)

Kullanıcı, eval kanıtlarına dayalı **5 somut madde** verdi; hepsi köklerinden çözüldü.
Yöntem yine "ölç → düzelt → yeniden ölç", hile yok (`sakın hile yapma` ilkesi).

## 12. Geri Bildirim → Kök Çözüm

### 12.1 "cabin değil sürücü ROI olmalı; modele giden alan minimum; yolo26l yap"
- **Sürücü-içi sıkı kırpma** (`pose.py:_driver_crop`): pose + nesne kanıtı artık tüm
  kabin yerine yalnız **sürücünün kişi kutusu (+%10)** üzerinde koşar. Kutu track başına
  önbelleğe alınır (`driver_crop.redetect_every`), kare başına TEK pose geçişi korunur.
- **Pose modeli `yolo26s-pose` → `yolo26l-pose`** (bootstrap indirir, lock'a yazıldı;
  diskte yoksa s-pose'a loglu fallback). Alan minimum olduğundan büyük model affordable.
- **Sonuç:** sigara tespiti video_1'de **23 → 118 kare** (önceki turun phone-FP
  bastırma tradeoff'u da ortadan kalktı — gerçek sigara artık net görünüyor).

### 12.2 "stabilite sorunlarını çöz" (car↔truck titremesi)
- **Track başına sınıf oylaması** (`stability/class_vote.py`): güven-ağırlıklı çoğunluk
  + hafif unutma (`decay=0.98`). Sınıf pipeline'da TEK noktada (`det.bbox.cls`) güncellenir
  → hız genişlik-önseli, accumulator, annotation, event'ler aynı kararlı sınıfı görür.
- **`min_track_frames` artık ÇIKTI kapısı**: 2-karelik phantom `truck` track'leri
  (video_3'te ByteTrack parçalanmasından) artık annotation/event üretmiyor.
- **Sonuç:** üç videoda da araç kalıcı `car`; `class_changes` izi (JSON özetinde) titremenin
  bastırıldığını kanıtlıyor (video_1: kare 37'de truck→car, sonra sabit).

### 12.3 "plakanın ilk harfi 0 okunuyor" (3→0, ve T→I)
- **Boyut-farkında kanıt** (`reader.py`): okuma ağırlığı = OCR güveni × kaynak kalitesi
  (LP kırpık yüksekliği). Çok küçük LP oylamaya girmez; küçük LP görüldüğü an
  `plate_too_small` QoD tetiği (consensus_fail beklemeden — havuz zehirlenmeden).
- **QoD erken bırakma**: plaka onaylanır onaylanmaz HIGH_THROUGHPUT bırakılır.
- **Pozisyon-hizalı karakter füzyonu** (`normalize.py:_char_consensus`): birden çok
  format-geçerli okuma pozisyon pozisyon birleşip en olası tahmini verir. İlk turda
  yalnız `partial`'a koymuştum; **ikinci turda güvenli şekilde ONAYA getirildi** (bkz. §15)
  — her pozisyonda kazanan ikinciyi `char_margin` mutlak farkla geçmeli, belirsizse pending.
- **Dürüstlük kararı (önemli):** video_1/3'te OCR plakayı varyantlara bölüyor (3→0, T→I).
  İlk turda char-füzyonu yanlış uyguladım (anchor + grup oranı) ve yanlış CONFIRMED ürettim
  (34IC8532 / 24IC8532) — geri aldım. İkinci turda **pozisyon-margin** ile doğru çözüldü:
  video_1 doğru `34TC8532` CONFIRMED, video_3 dürüst `pending` (uzaktan I↔T ayrılamıyor).
  Yanlış plakayı kesinleştirmek "okuyamadım"dan kötüdür — belirsiz pozisyon → pending.

### 12.4 "aracın doğru tanımlanmasını sağla"
- 12.2'deki sınıf oylaması + çıktı kapısı ile çözüldü; üç videoda da `car`.

### 12.5 "windows betiğini kontrol et"
- `run.ps1`/`setup.ps1` PS 5.1 hataları: `$ErrorActionPreference='Stop'` + native stderr
  modül-probe'u çökertiyordu (geçici `Continue` ile sarıldı); `Wait-Process` boş/çökmüş
  süreçte tüm servisleri öldürüyordu (`-EA SilentlyContinue` + guard); bare `python`
  (Store stub) → `py -3` fallback + çıkış-kodu doğrulama; `Push/Pop-Location`; UTF-8 BOM;
  `.env` yükleme (run.sh parite). Yeni **`dev.ps1`** (test/lint/format/eval/video-test —
  Makefile eşleniği). `bootstrap.py` git-lfs ipucu platforma göre (`%USERPROFILE%`/`~`).

## 13. Doğrulama Matrisi (13 Haz nihai, gerçek video, hile yok)

| Video | Araç | Plaka | Sürücü | Swerving | RISK_ALERT |
|---|---|---|---|---|---|
| video_1 | **car** ✓ | **34TC8532 CONFIRMED** ✓ | **sigara 118 kare** ✓ | 0 ✓ | 4 |
| video_2 | **car** ✓ | **34TC8532 CONFIRMED** ✓ | **telefon 110 kare** ✓ | 0 ✓ | — |
| video_3 | **car** ✓ | `pending` (uzak/bulanık) | temiz ✓ | **119 kare** ✓ | 1 |

Çapraz-FP sıfır (video_1'de telefon 0, video_2'de sigara 0). JSON özetlerine karar
şeffaflığı için `plate_raw_valid_weighted` (ağırlıklı format-geçerli dağılım) ve
`class_changes` (sınıf izi) eklendi — jüri "neden bu sonuç?" diye sorabilir.

**Dürüst sınırlar:** video_3 plakası uzaktan/bulanık olduğundan OCR `3→2` ve `T→I`
yapıyor ve hiçbir pozisyon güvenli ayrılmıyor → sistem `pending` der (yanlış pozitif yok).
Plaka kimliği **video_1 ve video_2'de CONFIRMED** (`34TC8532`). video_1/2'de araç ilk
~1-2 sn uzaktayken model ham tespitte `truck` görüyor (ham-tespit sınırı); sistem
yakınlaşınca `car`'a yakınsayıp sabit kalıyor — kalıcı çözüm `cigarette`/araç-tipi
dengeli v5 fine-tune (`train/` hazır).

## 14. Kalite (13 Haz)
- **137 unit test** yeşil; yeni/genişletilmiş: `test_class_vote` (9, alan-ağırlığı dahil),
  `test_plate_normalize` (pozisyon-margin füzyonu + regresyon koruması).
- `ruff` + `black` temiz. `tools/show_cabin_rois.py` → `show_driver_rois.py` (pipeline-eş).
- Docs güncel: CHANGELOG (2.2.1), `docs/mimari.md`, `config/README.md`, `pyproject` 2.2.1.

## 15. İkinci Geri Bildirim Düzeltmesi ("bozdun; stabilite düzelmedi; plaka daha iyiydi")

Kullanıcının ikinci geri bildirimi gerçek eval ölçümüyle kök çözüme kavuşturuldu —
tahmin değil, ölçüm: her video için ham per-frame sınıf + ağırlıklı plaka dağılımı çıkarıldı.

- **Plaka regresyonu (ilk-karakter `3↔0`):** ÖLÇÜM — OCR aynı plakayı `34TC8532`/`04TC8532`/
  `34IC8532` diye bölüyor; hiçbiri tek başına `ratio`'yu geçmiyor, ayrı-aday hangisi
  baskınsa onu (koşuma göre yanlış `04`) seçiyordu. ÇÖZÜM — **pozisyon-margin füzyonu onaya**:
  pos0'da çoğunluk `3` (34TC+34IC > 04), pos2'de `T` net → **`34TC8532` CONFIRMED**.
  Belirsiz pozisyon → pending (video_3). `consensus_ratio` 0.6'ya geri alındı.
- **Stabilite (sınıf salınımı):** ÖLÇÜM — model video_2'de ilk 53 kareyi ham tespitte
  `truck` görüyor (uzak araç); ilk denememdeki `decay<1` + alan-ağırlığı video_3'te
  GEÇ büyük-alan tespitine **salınım** (car→truck→car) yaratmıştı. ÇÖZÜM — **alan-ağırlıklı
  oy** (yakın/büyük araç daha güvenilir) + **`decay=1.0`** (saf kümülatif): video_3 tek
  sınıf `car` (salınım yok); video_1/2 car'a yakınsayıp sabit. İlk uzak-araç truck dönemi
  modelin ham-tespit sınırı (dürüstçe belgelendi).
- **Eval kanıtı:** kullanıcının isteğiyle `tools/test_video.py` (3 video, annotated + JSON)
  ve `tools/show_driver_rois.py` (3 video, sürücü ROI grid) yeniden koşuldu — `eval_results/`.
