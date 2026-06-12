# Şartname İzlenebilirlik

TEKNOFEST 2026 "5G & Yapay Zekâ ile Akıllı Yol Güvenliği" şartnamesindeki her zorunlu
madde, onu karşılayan bileşene/dosyaya bağlanır. (Kaynak: şartname PDF + `plan.md` §0/§17.)

## Zorunluluk → Bileşen eşlemesi

| # | Şartname zorunluluğu | Karşılayan bileşen | Puan | Durum |
|---|---|---|---|---|
| 1 | Araç tespiti | `aura/detection` (fine-tune `v4` 11-sınıf + ByteTrack; stok YOLO26s fallback; kopya-kutu dedup) | %40 | ✅ (gerçek 4K videoda doğrulandı) |
| 2 | Plaka tespiti + okuma | `aura/plate` (sweet spot + **sıkı LP kırpma** + güven-ağırlıklı kalıcı oylama + OCR + Türk regex + `partial` kanıt) | %40 | ✅ (gerçek videoda doğrulandı) |
| 3 | Hız tespiti | `aura/speed` (tripwire/ipm/disabled/metric + relative flag; `calibrated` etiketi) | %40 | ✅ |
| 3b | Hız-limiti tabelası + ihlal (yol güvenliği) | `aura/scene` (SignTracker) + `accumulator` `speed.over_limit` → `SPEED_LIMIT_VIOLATION` | %40 | ✅ (custom dataset retrain bekliyor) |
| 3c | Dikkatsiz sürüş / swerving (risk unsuru) | `aura/speed` ZigZag yanal yörünge analizi → `speed.swerving` → `RISK_ALERT` + QoD | %40 içinde | ✅ (gerçek videoda doğrulandı) |
| 4 | Araç-içi nesne / sürücü davranışı — **sigara, telefon** (şartname 4.4 birebir) + kemer/yorgunluk | `aura/driver_state` (pose-geometri + hibrit nesne kanıtı **veya** fine-tune YOLO26l; **no-landmark-lib**) | %40 içinde | ✅ (sigara+telefon gerçek videoda doğrulandı; kemer/yorgunluk fine-tune bekliyor) |
| 4b | **TOGG aracının yaklaştığını algılama** (QoD birincil senaryosu, Bölüm 1-2) | `aura/pipeline` bbox alan-büyüme yaklaşma tetiği → `QOD_TRIGGER reason=vehicle_approach` | %40 | ✅ (gerçek videoda tetiklendi) |
| 5 | QoD yalnızca kritik anda | `aura/qod` (histerezis; yaklaşma + kalite + anomali/swerving tetikleri) | %40 | ✅ |
| 6 | QoD başarım artışı **kanıtı** | `aura/eval` A/B harness + `GET /eval/results` + dashboard paneli | %40 | ✅ (delta: plaka +33pp, küçük nesne +51pp) |
| 7 | Number Verification sessiz doğrulama | `services/nv_mock` + `POST /verify` + `mobile/` | — | ✅ |
| 8 | Tespitlerin mobil ekranda gösterimi | `mobile/` + `WS /stream/events` | — | ✅ |
| 9 | Doğruluk / hassasiyet / model hızı | `train/` (fine-tune; held-out mAP50 .788) + `aura/eval` metrikleri + `tools/test_video.py` FPS raporu | %40 | ✅ |
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
pytest -m "not integration"                                            # 118 unit test
```
