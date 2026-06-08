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
