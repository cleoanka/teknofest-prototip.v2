# Şartname İzlenebilirlik

TEKNOFEST 2026 "5G & Yapay Zekâ ile Akıllı Yol Güvenliği" şartnamesindeki her zorunlu
madde, onu karşılayan bileşene/dosyaya bağlanır. (Kaynak: şartname PDF + `plan.md` §0/§17.)

## Zorunluluk → Bileşen eşlemesi

| # | Şartname zorunluluğu | Karşılayan bileşen | Puan | Durum |
|---|---|---|---|---|
| 1 | Araç tespiti | `aura/detection` (YOLO26s + ByteTrack) | %40 | ✅ |
| 2 | Plaka tespiti + okuma | `aura/plate` (sweet spot + voting + OCR + Türk regex) | %40 | ✅ |
| 3 | Hız tespiti | `aura/speed` (tripwire/ipm/disabled/metric + relative flag) | %40 | ✅ |
| 3b | Hız-limiti tabelası + ihlal (yol güvenliği) | `aura/scene` (SignTracker) + `accumulator` `speed.over_limit` → `SPEED_LIMIT_VIOLATION` | %40 | ✅ (custom dataset retrain bekliyor) |
| 4 | Araç-içi nesne / sürücü davranışı (telefon/sigara/kemer/yorgunluk) | `aura/driver_state` (YOLO26l, 4 sınıf, **no-landmark**) | %40 içinde | ✅ |
| 5 | QoD yalnızca kritik anda | `aura/qod` (histerezis; kalite + optimizasyon tetiği) | %40 | ✅ |
| 6 | QoD başarım artışı **kanıtı** | `aura/eval` A/B harness + `GET /eval/results` + dashboard paneli | %40 | ✅ (delta: plaka +33pp, küçük nesne +51pp) |
| 7 | Number Verification sessiz doğrulama | `services/nv_mock` + `POST /verify` + `mobile/` | — | ✅ |
| 8 | Tespitlerin mobil ekranda gösterimi | `mobile/` + `WS /stream/events` | — | ✅ |
| 9 | Doğruluk / hassasiyet | `train/` (YOLO26 fine-tune) + `aura/eval` metrikleri | %40 | ✅ |
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
curl -s localhost:8080/eval/results                                    # JSON metrik
pytest -m "not integration"                                            # 58 unit test
```
