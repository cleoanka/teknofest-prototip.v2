# Değişiklik Günlüğü

Bu projedeki tüm önemli değişiklikler bu dosyada belgelenir.
Format [Keep a Changelog](https://keepachangelog.com/tr/1.0.0/) temellidir.

## [Unreleased]

### Milestone 1 — Repo iskeleti + bootstrap + config + weights + smoke
- Tam monorepo dizin iskeleti oluşturuldu (`aura/`, `services/`, `dashboard/`, `train/`, `docs/`, `tests/`).
- `bootstrap.py` (saf stdlib): venv, torch backend otomatik tespiti, paket kurulumu, model ağırlığı otomatik indirme (SHA256 trust-on-first-use), örnek video üretimi, smoke test.
- `config/default.yaml` (§14 tam şema) + kalibrasyon örneği.
- `setup.sh`/`setup.ps1` + `run.sh`/`run.ps1` cross-platform sarmalayıcılar.
- `pyproject.toml` (aura paketi + core/dev bağımlılık grupları), `Makefile`, `.gitignore`, `.gitattributes`, `.env.example`, `LICENSE` (MIT).

### Milestone 2 — Pydantic v2 sözleşmeleri + pipeline iskeleti
- `aura/schema.py`: §6.0 sözleşmeleri (`PlateState`, `DriverState`, `SpeedState`, `BBox`, `TrackRecord`, `AuraEvent`, `AnnotationFrame`) + `make_event` yardımcısı.
- Pipeline iskeleti: `preprocessing` → `detection`(+ROI geometri) → `stability` ⊗ (`driver_state` ∥ `plate`) → `speed` → `accumulator` → `events`. Model-bağımlı modüller dürüst stub, akış ve sözleşmeler doğru.
- `accumulator` (gerçek): ID-merkezli `TrackRecord` + durum-değişimi event'leri + config'ten risk kuralları.
- `events.EventEmitter` (gerçek): event/annotation halka tamponu + callback kayıt defteri (M7 WS köprüsü için).
- `python -m aura` CLI (§4.1 argparse) + `Pipeline.run_video/frames`.
- `tests/test_contracts.py`: sözleşme + accumulator + risk testleri (5 geçti).

### Milestone 3 — detection + ByteTrack + ROI crop → accumulator (en kısa uçtan-uca)
- `aura/detection/yolo.py`: `YOLO26Detector` (gerçek) — ultralytics YOLO26 + ByteTrack, araç sınıf filtresi, ROI crop.
- `aura/detection/mock.py`: `MockDetector` (deterministik) — parlak araç bloklarını eşikler + `SimpleIoUTracker` ile kalıcı ID; ağırlık olmadan tüm hat çalışır.
- `build_detector` + `resolve_ai_mode`: real/mock/auto çözümlemesi (lazy import).
- Uçtan-uca doğrulandı: sentetik videoda 3 araç kalıcı ID ile takip, DETECTION_UPDATE event'leri, 90 annotation karesi.
- `tests/test_detection.py`: mock dedektör + IoU + ROI testleri (CI-uyumlu).

### Milestone 4 — stability (16/8) + driver_state
- `aura/stability/state_machine.py` (gerçek): per `track×alan` kayar pencere (16), ≥8 tutarlılıkta commit; flicker'da önceki yüksek-güvenli değer korunur.
- `aura/driver_state/yolo.py`: `YOLO26lDriverClassifier` (gerçek) — cabin ROI çoklu-etiket detection.
- `aura/driver_state/mock.py`: `MockDriverClassifier` — cabin baskın rengini senaryo durumuna eşler (phone / smoking+no_seatbelt / fatigue).
- Uçtan-uca: 3 DRIVER_STATE event + 1 RISK_ALERT (unbelted), 16/8 süzgecinden geçerek.
- `tests/test_stability.py` (7/16→ret, 8/16→kabul, flicker), `tests/test_driver_state.py` → toplam 20 test geçti.

### Milestone 5 — plate (sweet spot + voting + OCR) + QoD kalite tetiği
- `aura/plate/ocr.py`: `RealOCR` (EasyOCR) + `MockOCR` (renk→senaryo plakası); `build_ocr` çözümlemesi.
- `aura/plate/voting.py`: `VotingBuffer` (konsensüs/ret).
- `aura/plate/reader.py` (gerçek): sweet-spot gating → voting → Türk plaka regex → konsensüs (PLATE_CONFIRMED + erken çıkış) / ret (PLATE_REJECTED + QoD kalite tetiği + yeniden okuma); yetersiz piksel → QoD.
- `aura/qod/client.py`: `QoDController` — histerezis (min_active + cooldown), QOD_TRIGGER/RELEASE; pipeline'a entegre.
- accumulator plaka snapshot fix'i (aliasing → geçiş event'leri).
- Uçtan-uca: sweet-spot içi 2 PLATE_CONFIRMED, sağ-şerit araç gating ile pending.
- `tests/test_plate.py` + `tests/test_qod.py` → toplam 31 test geçti.

### Milestone 6 — speed (disabled + relative flag, tripwire, ipm fallback)
- `aura/speed/estimator.py` (gerçek): `disabled` (relative_velocity_flag), `tripwire` (iki çizgi × gerçek mesafe / frame-delta), `ipm` (M12 opsiyonel modüle güvenli düşüş).
- Göreli hız bayrağı 16/8 süzgecinden geçirildi (eşik civarı salınım önlendi).
- Speed anomalisi → QoD optimize tetiği (LOW_LATENCY).
- Uçtan-uca senaryo: 3 DETECTION + 3 DRIVER_STATE + 2 PLATE_CONFIRMED + 1 SPEED + 1 QOD_TRIGGER + 1 RISK_ALERT.
- `tests/test_speed.py` → toplam 35 test geçti.

### Milestone 7 — events + inference_api + qod_mock + nv_mock
- `services/inference_api/` (FastAPI :8080): `StreamManager` (arka plan pipeline worker), tüm router'lar (system/cameras/stream/tracks/eval/config), MJPEG `GET /stream/video`, `WS /stream/annotations` + `WS /stream/events` — iki-kanal tasarım.
- `services/qod_mock/` (:8081): CAMARA QoD sözleşmesi (sessions CRUD).
- `services/nv_mock/` (:8082): Number Verification sessiz doğrulama.
- `GET /cameras`: OpenCV enum + platform isim çözümleme (macOS AVFoundation), `AURA_CAMERA_PROBE=0` ile atlanır.
- Canlı doğrulama: 3 servis kalktı, pipeline autostart (67 kare/3 track/1 QoD session), MJPEG 917KB aktı, mock'lar yanıt verdi, OpenAPI 15 endpoint.
- `tests/test_events.py` + `tests/test_api_contracts.py` → toplam 46 test geçti.

### Milestone 8 — dashboard (kamera seçici + MJPEG+Canvas + bbox toggle + event log + track panel)
- `dashboard/` (vanilla HTML5 + ES6 modules + Canvas + WS + Chart.js CDN, npm/build yok).
- İki-kanal video: MJPEG `<img>` + `<canvas>` overlay; **bbox toggle client-side** (MJPEG kesilmez).
- `camera-selector.js` (webcam/iPhone/video/RTSP), `video-renderer.js` (bbox+ikon+QoD çizimi), `event-stream.js` (auto-reconnect WS), `qod-panel.js` (Chart.js A/B), `app.js` (orkestratör + track kartları + event log + tema).
- `style.css`: dark/light CSS custom properties, grid düzen.
- `inference_api` `/` ve `/assets` üzerinden serve eder.
- Doğrulandı: `/` 200 (html), 5 ES modül + CSS 200, `node --check` 5/5.

### Milestone 9 — QoD A/B paneli (eval harness + /eval/results + Chart.js)
- `aura/eval/metrics.py`: Levenshtein/CER, plaka doğruluğu, tespit oranı, küçük-nesne oranı.
- `aura/eval/harness.py`: aynı video QoD ON (tam çözünürlük) vs QoD OFF (düşük çözünürlük) → GT'ye karşı delta tablosu + report.md/json.
- `python -m aura.eval` CLI (§4.3) + `/eval/run`/`/eval/results`/`/eval/results/export` bağlandı; dashboard Chart.js paneli tüketir.
- Ölçülen delta (şartname %40 kanıtı): Plaka +33pp, Küçük nesne +51pp, Tespit +25pp.
- `tests/test_eval.py` → toplam 50 test geçti.

### Milestone 10 — train modülü + egitim/veri_seti dokümanları
- `python -m train` (§4.2): `detector` (YOLO26s) / `driver-state` (YOLO26l 320px) / `dataset` subcommand'ları; ultralytics lazy import (`--help` torch gerektirmez).
- `train/prepare_dataset.py` (gerçek): train/val/test split + `data.yaml` üretimi (deterministik).
- `train/roboflow_pull.py` (ROBOFLOW_API_KEY), `train/utils.py` (custom ağırlık swap), `configs/` data.yaml örnekleri.
- `docs/egitim.md` + `docs/veri_seti.md`.
- `tests/test_train.py` → toplam 53 test geçti.

### Milestone 11 — mobil Expo iskeleti
- `mobile/` (Expo SDK 51 + React Native + TypeScript): NV sessiz giriş, canlı `WS /stream/events` listesi, QoD rozeti.
- `src/api/client.ts` (verifyNumber + connectEvents + setSource), `LoginScreen`/`DashboardScreen`, `src/config.ts` (EXPO_PUBLIC_API_URL/NV_URL).
- `mobile/README.md`: `npx expo start`, mock↔gerçek geçişi (yalnızca adres), Android emülatör notu.
- Doğrulandı: JSON config'ler valid, TS import yolları tutarlı (Expo build emülatör gerektirir).

### Milestone 12 — §8 opsiyonel modüller (toggle + lazy import)
- `aura/optional/loader.py`: `get_optional(cfg, name)` — flag kapalıyken **import yapmaz** (lazy).
- `zero_waste_payload.py` (ROI+yapısal metin payload), `super_resolution.py` (OCR öncesi upscale), `homography_ipm.py` (`ipm_speed` piksel→dünya).
- Pipeline/PlateReader/SpeedEstimator'a lazy hook'lar; kapalıyken davranış değişmez.
- `docs/mimari_ek_moduller.md` (ana mimari yalnızca referans verir).
- `tests/test_optional.py`: kapalıyken `sys.modules`'te yok + işlevsellik → toplam 58 test geçti.

### Milestone 13 — CLI --help her yerde + docs/cli_referans.md
- Tüm entry point'ler argparse `--help` (§4 şablonları); `docs/cli_referans.md` gerçek çıktılardan (10 komut).

### Milestone 14 — docs/api_referans.md
- Tüm endpoint'ler (inference_api + mock'lar): curl + httpx + canlı response örnekleri.

### Milestone 15 — docs/mimari.md v2.0 + doküman tamamlama
- `docs/mimari.md`: v1.1 YZ katmanı (§1–7) korundu + sistem katmanı (§8: topoloji, event/annotation sözleşmesi, NV/QoD akışı, mock↔gerçek sınırı), yorgunluk/MediaPipe gerekçesi (§9), kamera enumerasyonu (§10).
- `docs/kurulum.md`, `docs/calistirma.md`, `docs/kalibrasyon.md`, `docs/degerlendirme.md`, `docs/README.md` eklendi.
