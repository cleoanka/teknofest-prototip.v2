# GÖZLEM — TEKNOFEST 2026 "AURA" Projesi (Kapsamlı İnceleme)

> Salt-gözlem belgesi · 18.06.2026 · Hazırlayan: Claude (Opus 4.8)
> Kapsam: 4 repo / 277 commit + tüm plan + mimari + bizim katkımız + **internetteki en iyi çözüm yolları (SOTA) ile karşılaştırma**.
> Bu belge yalnızca GÖZLEMDİR — projede hiçbir değişiklik yapılmamıştır.

---

## 1. Proje & Yarışma Bağlamı

**TEKNOFEST 2026 "5G & Yapay Zekâ ile Akıllı Yol Güvenliği"** (Turkcell). Senaryo: yol-kenarı kamera akışı → YZ analiz → **TOGG aracı yaklaştığında 5G CAMARA QoD** ile geçici yüksek ağ kalitesi → daha başarılı analiz. **Number Verification** ile sessiz kullanıcı doğrulama. Mobil uygulamada gösterim.

- **Tespit hedefleri (şartname 4.4):** araç, plaka (+OCR), gerçek hız, araç-içi nesne = **sigara/telefon**, kemer, yorgunluk, riskli sürüş.
- **Puanlama (Tablo 1):** %40 YZ doğruluk/hassasiyet · %40 QoD entegrasyonu (yalnız kritik anda bant↑) · %20 rapor+sunum.
- **Kural 4.5:** her hedefin OTOMATİK üretildiği kanıtlanmalı (kanıtlanamayan sayılmaz).
- **Takvim:** FTR son teslim **28.06.2026** (ertelendi); finalistler 31.07; final Ağu–Eyl 2026.
- **FTR rubriği (100p):** Özet 5 / Veri Seti 20 / YZ Çözümü 50 / Sınama 20 / Kaynakça 5. Format: 3–10 sayfa, Arial 12.

---

## 2. Repo Soy-Ağacı (lineage) — Tüm Commitlerin Hikâyesi

Dört repo, tek projenin evrimi (GitHub: cleoanka):

### 2.1 `~/teknofest-prototip` — **v1, full-stack orijinal** (84 commit)
Projenin en zengin köküdür. Üç bağımsız kol:
- **Mobil:** React Native/Expo, v1→v2 yeniden yazım, kurulabilir Android APK, SDK54, NV-login + canlı kamera (taglar `v1.0.0-mobile`, `v2.0.0-mobile`).
- **Backend:** FastAPI; JWT **RS256** auth, rate-limiting, Prometheus, WS token-auth, pagination, plate-search, heatmap — **~248 test**e kadar büyüdü.
- **YZ:** YOLOv8; iki-aşamalı plaka (araç-crop→LP dedektör); **MediaPipe** sürücü geometrisi (telefon/sigara); **metrik hız** (ByteTrack + homografi + plaka-köşe PnP + Kalman, jitter −%40); eğitim hattı (açık-veri fetch/manifest, COCO→YOLO, aşamalı müfredat); **fine-tune `yolguvenligi_types_v4` (yolov8m@768, held-out mAP50 .788, 11 sınıf)**; tip-ayrımlı taksonomi.
- Branch'ler: `cleodemo`, `feat/ai-gercek-hiz`, `feat/mobile-v2`, `feat/mobile-sdk54-features`.

### 2.2 `~/hidden_prototip` — **YOLO26 cascade ara-iterasyon** (91 commit)
v1 ile ortak erken tarih (~`7dad35d`'ye kadar), sonra ayrışır: **kademeli (cascade) onay** deneyi — YOLO26n gate → YOLO26s confirm, **4-durumlu QoD tetik makinesi**, NMS-free uyumu, cascade latency ölçümü, bant verimliliği metrikleri. Kritik ders: **stok YOLO gerçek trafikte araçlara 0.05-0.15 conf verir → `conf_gate` 0.08'e indirildi**.

### 2.3 `~/aura` — **temiz M1-M16 yeniden inşa** (16 commit)
Sıfırdan, disiplinli mimari (M1 iskelet → M16 test+CI+izlenebilirlik). v2'nin geliştirme reposu; tarihi v2 main'in içindedir (ayrı tutmak gereksiz).

### 2.4 `~/teknofest-prototip.v2` — **AKTİF monorepo** (86 commit + bizim 11)
aura M1-M16'yı yutmuş + **v2.1/2.2/2.3** evrimi:
- v2.1: üç prototipin sentezi, **pose-tabanlı sürücü** (MediaPipe yerine saf YOLO26-pose geometri), **swerving** (ZigZag), plaka format-öncelikli kalıcı oy havuzu, QoD yaklaşma tetiği.
- v2.2: sürücü-içi sıkı kırpma, l-pose, **araç-sınıfı oylaması** (car↔truck titreme), boyut-farkında plaka kanıtı.
- v2.3: **YOLO26l varsayılan dedektör** + config profilleri (server/laptop/v4-finetune), **iki-katmanlı sürücü motoru** (Katman A model + Katman B per-ID 16/8), FTR metrik harness'ı, tabela/hız-limiti sahne katmanı, pozisyonel sürücü/yolcu kilidi.
- Takım dalları: `plate-speed-calibration` (LP→hız oto-kalib), `stage2-driver-state` (kemer iki-katman, domain model).

### 2.5 `feat/ultraplan-w1` — **bizim katkımız** (11 commit, origin'e push'lu)
`04c024c` W1 (plaka dewarp/enhance/PaddleOCR + mAP harness + hız-GT MAE/MAPE + veri manifesti) → `d1543bd` inceleme fixleri → `7b4d517` WP-E (docs 256-test + dashboard cila + bench.py) → `8fef7f6` **dewarp/enhance OFF** (ölçülen regresyon: 34TC→34IC) → `810cd1d` B3 Mermaid diyagramlar → `ef56954` FTR rapor taslağı → `2c3be7b`+`7c8aa1e` **stabilite fixleri** (yanlış-onay sıfır + phantom/track_id=-1 çıktı kapısı + orphan-persons + devasa-ROI sınırı) → `0c33d64`+`c1d71d4` honesty/şartname fixleri → **`d1b7e48` fast-plate-ocr (video_3 kurtarıldı, plaka 3/3 exact)**.

---

## 3. Mimari (mevcut, v2 + W1)

```
[Kamera/RTSP] → [Ön-İşleme] → [YOLO26l + ByteTrack + alan-ağırlıklı sınıf-oyu] ─┬→ [Sürücü ROI] → Katman A: YOLO26-pose geometri + hibrit nesne
                                  ↑ (çıktı kapısı: min_output_frames,            │                Katman B: per-ID 16/8 zaman-oylaması (engine)
                                     track_id=-1 guard, phantom bastırma)         └→ [Plaka ROI] → YOLO11n LP + fast-plate-ocr + format-öncelikli oy havuzu
                                                                                                    + dürüstlük zırhları (pozisyon-veto, zemin-koşulu, confirm_min_char_margin)
                              [ID-merkezli Accumulator + risk kuralları] ← [Hız metrik oto-kalib + swerving ZigZag]
                                          ↓                                          ↓
                              [Event/Annotation stream] → Dashboard + Mobil + JSONL kanıt        [QoD tetik: yaklaşma/kalite/anomali + histerezis]
```
- **Cascade + 2 katman**; **ID-merkezli** birikim; **16/8 kararlılık**; **kalibrasyon-bağımlı hız** (yoksa flag); **landmark kütüphanesi YOK** (saf YOLO26-pose); **sahne katmanı** (hız-limiti tabelası).
- **Gerçek/Mock sınırı:** YZ çekirdeği (preprocessing/detection/tracking/driver/plate/speed/eval/train) GERÇEK; telekom (qod_mock CAMARA, nv_mock NV, 5G, TOGG beslemesi) MOCK → finalde yalnız endpoint/credential değişir.
- ~6000 satır `aura/` + servisler (inference_api 18 uç, qod_mock, nv_mock) + dashboard (vanilla JS) + mobil (Expo/RN temel uygulama, tsc-temiz) + train (YOLO26 fine-tune hattı).

---

## 4. Ölçülen Gerçek Metrikler (W1 sonrası, dürüst)

| Metrik | Değer | Not |
|---|---|---|
| Davranış makro-F1 | **1.0** (her iki dedektör) | phone/smoking/swerving P=R=F1=1.0; 3 gerçek video |
| Araç sınıfı doğruluğu | **%100** | |
| **Plaka** | **3/3 exact (CER 0)** | fast-plate-ocr; video_1/2/3 = 34TC8532 (video_3 eski: 24IC8532 pending) |
| Held-out dedektör mAP | yolo26l COCO val2017: **mAP50-95 0.537 / mAP50 0.709** | 5000 görsel, gerçek held-out |
| QoD A/B | plaka **+33.3pp**, küçük-nesne **+51.4pp**, tespit **+25.5pp** | sentetik kontrollü set (kare-düzeyi GT); `qod_comparison=true` |
| FPS (MPS, M4 Pro) | ~5-6 | CUDA sunucuda daha yüksek |
| Kalite | **~600 birim test** + ruff + black + CI | `tests/` ~604 `def test_`; servis testleri sürüyor | |

**Stabilite/doğruluk zırhları (W1, gerçek-videoda doğrulandı):** ilk-harf-0 yok, track_id=-1/phantom/orphan-sürücü yok, bbox/swerving-FP yok, devasa-ROI sınırı. Belirsizde sahte-onay yerine **dürüst pending**.

---

## 5. İnternetteki EN İYİ Çözüm Yolları (SOTA) ↔ AURA Karşılaştırması

### 5.1 Araç tespiti + takip
- **SOTA (2025-26):** **YOLO26** (Eyl 2025) — NMS-free uçtan-uca, DFL kaldırıldı, **mAP50-95 ~51%(m)/53%(l)**, CPU'da YOLO11'den ~%43 hızlı, RT-DETR'den düşük gecikme. Alternatif: RF-DETR, YOLOv13. Takip: ByteTrack/BoT-SORT.
- **AURA:** varsayılan **stok yolo26l** + ByteTrack + alan-ağırlıklı sınıf-oyu. → **SOTA-uyumlu.** Held-out mAP50-95 0.537 (yayınlanan ~53% l ile tutarlı). Artı: kopya-kutu dedup, çıktı kapısı.

### 5.2 Plaka tespiti + OCR (en kritik)
- **SOTA:** iki-aşamalı **YOLO (LP) + güçlü OCR**. OCR'da **PaddleOCR PP-OCRv5** ve **fast-plate-ocr** (hafif, plakaya-özel ONNX) öne çıkıyor; PARSeq/LPRNet alternatif. Karanlık/küçük için dewarp + süper-çözünürlük tartışılıyor ama OCR motoru en büyük kaldıraç.
- **AURA:** YOLO11n LP dedektör + **fast-plate-ocr** (bizim W1 ile getirildi) + format-öncelikli oy havuzu + dürüstlük zırhları. → **SOTA-uyumlu ve KANITLI:** fast-plate-ocr, EasyOCR'ın çözemediği video_3'ü (uzak/karanlık) **34TC8532 exact** okudu (ölçülü A/B; dewarp+enhance ve Lanczos-SR teknikleri ölçülüp KIRMIZI çıktı → kapalı). Kalan sınır: onnxruntime CPU (MPS yok) — etki düşük (plaka per-onay, her kare değil).

### 5.3 Sürücü davranışı (sigara/telefon/kemer/yorgunluk)
- **SOTA:** distraction veri setlerinde fine-tune YOLO (ME-YOLOv8 3660 görsel; **YOLOv12-LAD** 2025; YOLOv5 mAP@50 ~%93.6). Sınıflar phone/smoking/seatbelt/fatigue çoklu-etiket.
- **AURA:** **pose-hibrit** (eğitim gerektirmez; bilek↔ağız/kulak geometrisi + nesne füzyonu) → 3 videoda **makro-F1 1.0**. **Yeni veri toplandı** (CC BY 4.0, PIL-doğrulanmış): license_plate 9123, seatbelt 3104, smoking 557, phone 659; minibus toplanamadı (auth'suz kaynak yok). **Fine-tune (YOLO26s) ŞU AN SÜRÜYOR — final mAP'ler henüz kesinleşmedi:** license_plate ~epoch 12/35'te `mAP50≈0.97` (`runs/detect/.../license_plate_s`), seatbelt erken epoch (sırada), smoking sırada. → Bu, "zorunlu sınıf held-out mAP" boşluğunu kapatan en yüksek-etkili adımdır; eğitim bitince §4'e gerçek mAP girer. *Dürüstlük: bunlar SÜREN eğitimin ara değerleridir, nihai değil.*

### 5.4 Hız tahmini (monoküler CCTV)
- **SOTA:** tek kameradan ölçek/homografi/plaka-genişliği (PPM) + Kalman; kalibrasyon-bağımlı.
- **AURA:** metrik oto-kalibrasyon (TR plaka 520mm + araç-genişliği önseli + Kalman/EMA + ivme-aykırı reddi); kalibrasyon yoksa **dürüst göreli-bayrak** (mutlak hız iddia etmez). MAE/MAPE harness'ı hazır (komite gerçek-hız verisi gelince). → **SOTA-uyumlu + dürüst.**

### 5.5 5G CAMARA QoD + Number Verification
- **SOTA:** CAMARA (Linux Foundation) + GSMA Open Gateway standart API'leri. **QoD:** QoS-profil keşfi + session CRUD (4G/5G). **NV:** SIM↔numara sessiz doğrulama (etkileşimsiz). Oct-2025 meta-release: 60 API (10 stabil); NV genel kullanımda.
- **AURA:** `qod_mock` (session CRUD + histerezis + LOW_LATENCY/HIGH_THROUGHPUT profilleri) + `nv_mock` (sessiz /verify) — **CAMARA sözleşmesini birebir taklit**; yaklaşma-tetiği şartnamenin birincil senaryosu. → **SOTA-uyumlu;** finalde yalnız endpoint/credential. **Sınır:** gerçek CAMARA sandbox (Vodafone/Telefónica/Orange) ile test edilmedi; QoD A/B yalnız sentetik sette ölçülü (gerçek videoda kare-düzeyi GT yok).

---

## 6. Güçlü Yanlar / Zayıf Yanlar / Öneriler (gözlem)

**Güçlü:**
- Mimari **SOTA-uyumlu** (YOLO26 + fast-plate-ocr + CAMARA-sözleşmesi) ve **mühendislik-disiplinli** (~600 birim test, ruff/black, CI, tek-config, profiller, cross-platform).
- **Gerçek/Mock sınırı:** gerçek 5G olmadan uçtan-uca demo; finalde sadece endpoint değişir.
- **Dürüstlük (K-004):** belirsizde pending, sahte-onay yok; ölçülen sayılar (uydurma yok); izlenebilirlik (şartname↔modül).
- **W1 kanıtlı kazanımlar:** plaka 2/3+pending → **3/3 exact**; stabilite gediği (orphan/phantom/ilk-harf-0) kapatıldı; FTR rapor taslağı + diyagramlar hazır.

**Zayıf / Risk:**
- **Özel-model eğitimi SÜRÜYOR (henüz bitmedi)** → zorunlu sınıfların (license_plate, smoking, seatbelt) **nihai** held-out mAP'leri kesinleşmedi; şu an ara değer (license_plate `mAP50≈0.97` ep12/35) var ama §4'e nihai sayı eğitim bitince girer. Bugünkü §4 hâlâ ağırlıklı stok COCO + 3-video davranış + boru-hattı doğrulamasına dayanıyor.
- **QoD A/B** yalnız sentetik kontrollü sette (gerçek 3 videoda kare-düzeyi GT yok) — %40 ağırlıklı kriter; gerçek CAMARA ile teyit edilmeli.
- **Mobil** temel uygulama hazır (NV giriş + canlı WS panosu + QoD histerezis; §8) — final için gerçek cihaz testi + gerçek Turkcell CAMARA bağlama kaldı.
- Held-out set küçük (3 video) — istatistiksel mAP değil; geniş etiketli set lazım.
- fast-plate-ocr **CPU** (onnxruntime MPS sağlamaz) — etki düşük ama not.

**Öneriler (yarışma için, etki sırasına göre):**
1. **Süren fine-tune'u tamamla + held-out ölç** (license_plate / seatbelt / smoking; veri + manifest + boru-hattı hazır, eğitim koşuyor) → nihai mAP'leri §4'e işle; baseline'ı (pose-hibrit / stok) geçenleri varsayılan/profil yap.
2. **Gerçek CAMARA sandbox** (Vodafone/Telefónica/Orange) ile QoD+NV testi → §40 QoD kanıtını gerçek ağda göster.
3. **Mobil tam uygulama** (final 3. aşama): NV sessiz giriş + canlı kamera → QoD-tetikli yüksek çözünürlük + tespit gösterimi.
4. Plaka için **fastplate kalıcı** + (opsiyonel) dewarp/SR'yi komite footage'ında A/B (bu footage'da KIRMIZI çıktılar — overfit etme).
5. FTR §4'e **hız** ve **kanıt-izi (4.5)** alt-bölümlerini güçlendir (W1'de eklendi); zorunlu-sınıf mAP boşluğunu dürüstçe belirt.

---

## 7. Gemini + Codex Tartışması (GÜNCELLENDİ)
> **Gemini — ÇALIŞIYOR (`gemini-2.5-flash`):** İlk denemedeki `403 SUBSCRIPTION_REQUIRED` YALNIZCA `gemini-3.1-pro-preview` (enterprise Code-Assist lisansı isteyen) modelindeydi; **`gemini-2.5-flash` sorunsuz.** Gemini ile **mobil-RN best-practice** (CAMARA QoD = backend-proxy deseni; RN yerleşik WebSocket + FlatList; MJPEG → `react-native-webview`; Card/Badge panosu) + **final/CAMARA stratejisi** araştırıldı. §5 SOTA = WebSearch + repo-ölçüm + Gemini.
> **Codex:** salt-okunur danışma çağrıldı ama **0-çıktı ile takıldı** (yanıt vermedi) — bu oturumda kullanılamadı.

## 8. Bu Oturumun Yeni Katkısı — Mobil Uygulama (final 3. aşama)
> §6'daki "mobil iskelet" boşluğu KAPATILDI. `mobile/` (Expo/RN/TS) geliştirildi (commit `1bbbf8c`, **`npx tsc --noEmit` exit 0**): NV sessiz giriş (POST /verify) · canlı WS tespit panosu (/stream/annotations → araç kartları + plaka CONFIRMED/pending rozet + sürücü ikonları + risk/swerving) · /stream/events akışı · **QoD histerezis** (QOD_TRIGGER/RISK → 'quality' PATCH /config → 6sn histerezis → baseline) · MJPEG video (WebView varsa). Finalde kalan: gerçek Turkcell CAMARA + cihaz testi. Detay: `plan1.md` Faz 2 · `plan2.md` Ö5.

---

## 9. Kaynakça (WebSearch)
- fast-plate-ocr: https://github.com/ankandrew/fast-plate-ocr
- PaddleOCR PP-OCRv5: https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det · https://www.tenorshare.com/ocr/paddleocr.html
- YOLO+PaddleOCR plaka pipeline: https://medium.com/@ggulsumkayhann/license-plate-detection-and-recognition-with-yolo-and-paddleocr-9c39baecce87
- YOLO26 benchmark: https://arxiv.org/html/2509.25164v5 · https://docs.ultralytics.com/models/yolo26 · https://blog.roboflow.com/best-object-detection-models/
- Sürücü distraction: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/itr2.12560 · https://www.mdpi.com/2079-9292/15/9/1838 (YOLOv12-LAD)
- CAMARA QoD/NV: https://camaraproject.org · https://developers.opengateway.telefonica.com/docs/qod · https://developers.opengateway.telefonica.com/docs/numberverification · https://developer.vodafone.com/camara

---
*main `d34ed83` intact; tüm çalışmalar `feat/ultraplan-w1`'de. Bu oturumda mobil uygulama geliştirildi (commit `1bbbf8c`, tsc-temiz; yalnız `mobile/`). Gözlem belgesi sonu.*
