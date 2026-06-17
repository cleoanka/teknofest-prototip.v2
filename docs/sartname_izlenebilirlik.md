# Şartname İzlenebilirlik

TEKNOFEST 2026 "5G & Yapay Zekâ ile Akıllı Yol Güvenliği" şartnamesindeki her zorunlu
madde, onu karşılayan bileşene/dosyaya bağlanır. (Kaynak: şartname PDF + `plan.md` §0/§17.)

## Zorunluluk → Bileşen eşlemesi

> **v2.3 notu:** Dedektör omurgası artık varsayılan **stok YOLO26l** (sunucu); `v4` fine-tune
> seçilebilir profil (`--profile v4-finetune`). Sürücü durumu **iki katman**: Katman A model
> (pose-hibrit/YOLO26l) + Katman B per-ID 16/8 zaman-oylaması (`DriverStateEngine`). Metrik
> kanıtı: `python -m aura.eval --metrics-report` (P/R/F1+CER+FPS, dedektör A/B). Detay: `ftr.md`.

| # | Şartname zorunluluğu | Karşılayan bileşen | Puan | Durum |
|---|---|---|---|---|
| 1 | Araç tespiti | `aura/detection` (**varsayılan stok YOLO26l** + ByteTrack; `v4` 11-sınıf fine-tune profili; sınıf-bağımsız kopya-kutu dedup + alan-ağırlıklı sınıf oyu) | %40 | ✅ (3 gerçek videoda araç %100; dedektör A/B `metrics_report.md`) |
| 2 | Plaka tespiti + okuma | `aura/plate` (sweet spot + **sıkı LP kırpma** + güven-ağırlıklı kalıcı oylama + OCR + Türk regex + `partial` kanıt) | %40 | ✅ (gerçek videoda doğrulandı: **her iki dedektör** 2/3 exact-match, CER 0.083, **0 yanlış-onay**; belirsizde dürüstçe `pending`, asla yanlış plaka onaylamaz) |
| 3 | Hız tespiti | `aura/speed` (tripwire/ipm/disabled/metric + relative flag; `calibrated` etiketi) | %40 | ✅ |
| 3b | Hız-limiti tabelası + ihlal (yol güvenliği) | `aura/scene` (SignTracker) + `accumulator` `speed.over_limit` → `SPEED_LIMIT_VIOLATION` | %40 | ✅ (custom dataset retrain bekliyor) |
| 3c | Dikkatsiz sürüş / swerving (risk unsuru) | `aura/speed` ZigZag yanal yörünge analizi → `speed.swerving` → `RISK_ALERT` + QoD | %40 içinde | ✅ (gerçek videoda doğrulandı) |
| 4 | Araç-içi nesne / sürücü davranışı — **sigara, telefon** (şartname 4.4 birebir) + kemer/yorgunluk | `aura/driver_state` (pose-geometri + hibrit nesne kanıtı **veya** fine-tune YOLO26l; **no-landmark-lib**) | %40 içinde | ✅ (sigara+telefon gerçek videoda doğrulandı; kemer/yorgunluk fine-tune bekliyor) |
| 4b | **TOGG aracının yaklaştığını algılama** (QoD birincil senaryosu, Bölüm 1-2) | `aura/pipeline` bbox alan-büyüme yaklaşma tetiği → `QOD_TRIGGER reason=vehicle_approach` | %40 | ✅ (gerçek videoda tetiklendi) |
| 5 | QoD yalnızca kritik anda | `aura/qod` (histerezis; yaklaşma + kalite + anomali/swerving tetikleri) | %40 | ✅ |
| 6 | QoD başarım artışı **kanıtı** | `aura/eval` A/B harness + `GET /eval/results` + dashboard paneli | %40 | ✅ (delta: plaka +33.3pp, küçük nesne +51.4pp, tespit oranı +25.5pp; `eval_results/report.json` `qod_comparison=true`, yeniden-üretilebilir. **Kontrollü sentetik sette** `data/samples/ornek.mp4` ölçülür — gerçek videoda kare-düzeyi GT olmadığından A/B orada ölçülemez) |
| 7 | Number Verification sessiz doğrulama | `services/nv_mock` + `POST /verify` + `mobile/` | — | ✅ |
| 8 | Tespitlerin mobil ekranda gösterimi | `mobile/` + `WS /stream/events` | — | ✅ |
| 9 | Doğruluk / hassasiyet / model hızı | `train/` (YOLO26 fine-tune→`model.val`→mAP/P/R/F1 export) + `aura/eval --metrics-report` (video-düzeyi P/R/F1+CER+FPS, dedektör A/B) + `tools/test_video.py` FPS | %40 | ✅ (**her iki dedektör** makro-F1 1.0; plaka 2/3 exact, 0 yanlış-onay; araç %100. Stok `yolo26l` COCO val2017 held-out mAP50 0.709 / mAP50-95 0.537 — zorunlu-sınıf held-out mAP komite verisi gelene dek YOK) |
| 9b | **Her hedefin otomatik üretildiğinin kanıtı** (Bölüm 4.5) | `--save-events` JSONL izi + `tools/test_video.py` (annotated mp4 + JSON özet + oy dökümü) + `PlateState.partial` | — | ✅ |
| 10 | Modern mimari / rapor | repo yapısı + `docs/` + CI (`.github/workflows/ci.yml`) | %20 | ✅ |

## Mimari kararlar (şartname uyumu)
| Karar | Şartname gerekçesi |
|---|---|
| Cascade YOLO26s→YOLO26l | Edge-first verim; ağır modeli yalnızca ROI'de çalıştır |
| CAMARA QoD + histerezis | "QoD yalnızca kritik anda" + 5G-native kaynak yönetimi |
| 16/8 state machine | Kararlı tespit (yanlış alarm engelleme) |
| ID-merkezli accumulator | Tutarlı karar üretimi |
| Sahne katmanı (tabela) | Tabela araca değil sahneye ait → ID-merkezli akışın yanında ayrı katman |
| No-MediaPipe yorgunluk | Trafik kamerası koşullarında dayanıklılık (detection sınıfı) |
| Kalibrasyon-bağımlı hız | Sistemin sınırını tanıması (kalibrasyon yoksa flag) |

## Gerçek / Mock sınırı
- **Gerçek (YZ çekirdeği):** preprocessing, detection, tracking, stability, driver_state, plate/OCR, speed, accumulator, eval, train.
- **Mock (sözleşme taklidi):** `qod_mock` (CAMARA QoD), `nv_mock` (Number Verification), 5G şebekesi, TOGG video beslemesi.
- Final ortamda **yalnızca endpoint/credential** değişir; sözleşme ve YZ çekirdeği aynı kalır.

## Kanıt komutları
```bash
python -m aura.eval --source data/samples/ornek.mp4 --qod-comparison   # QoD A/B delta
python tools/test_video.py --source <video.mp4> --device mps           # annotated mp4 + JSON kanıt
python -m aura --source <video.mp4> --save-events kanit.jsonl          # event JSONL izi
curl -s localhost:8080/eval/results                                    # JSON metrik
pytest -m "not integration"                                            # 256 unit test
```
