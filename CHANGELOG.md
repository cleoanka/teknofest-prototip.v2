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
