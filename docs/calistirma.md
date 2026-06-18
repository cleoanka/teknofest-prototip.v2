# Çalıştırma — Uçtan Uca Demo Senaryosu

## Ne yapar
Kurulu sistemde tipik bir demo akışını adım adım gösterir.

## Gereksinimler
`./setup.sh` tamamlanmış olmalı.

> **Önce sağlık kontrolü (önerilir):**
> ```bash
> python tools/doctor.py        # bağımlılık, cihaz (MPS/CUDA/CPU), ağırlık, config ✓
> ```
> Tüm çekirdek bileşenler ✓ ise sistem **gerçek modda** hazırdır. Ağırlık eksikse pipeline
> sessizce `mock` moda düşer (çökmez) — `python bootstrap.py` ile ağırlıkları indirin.

## Senaryo

### 1. Servisleri kaldır
```bash
./run.sh
```
inference_api (:8080), qod_mock (:8081), nv_mock (:8082) kalkar. Dashboard otomatik
örnek videoyla başlar (`AURA_AUTOSTART=1`).

### 2. Dashboard'u aç
```
http://localhost:8080/
```
- **Kaynak seç** (sol): webcam / iPhone / video / RTSP.
- **Canlı video** (orta): MJPEG + Canvas bbox overlay.
- **BBox: ON/OFF** — client-side toggle (akış kesilmez).
- **Event log** — PLATE_CONFIRMED, RISK_ALERT, QOD_TRIGGER… renk-kodlu.
- **Track listesi** (sağ): plaka, hız, sürücü ikonları, QoD rozeti; tıkla → detay.

### 3. QoD A/B kanıtı
Alt panelde **Eval Çalıştır** → harness aynı videoyu QoD OFF/ON koşar; Chart.js delta
gösterir (plaka doğruluğu, küçük nesne, tespit oranı).

### 4. CLI ile pipeline
```bash
.venv/bin/python -m aura --source data/samples/ornek.mp4 --max-frames 90 --log-level INFO
.venv/bin/python -m aura.eval --source data/samples/ornek.mp4 --qod-comparison
```

### 5. Gerçek video testi (kanıt üretimi)
Gerçek 4K test videosunu pipeline'dan geçirip **annotated mp4 + JSON kanıt** üretir
(şartname 4.5 izi: plaka kararı, sürücü bayrak süreleri, swerving, FPS):
```bash
.venv/bin/python tools/test_video.py --source ~/video_1.mp4 --device mps
#   → eval_results/video_1_annotated.mp4 + eval_results/video_1_summary.json
```

### 6. Mobil (opsiyonel)
```bash
cd mobile && EXPO_PUBLIC_API_URL=http://<LAN-IP>:8080 npx expo start
```
Sessiz NV girişi → canlı event listesi + QoD rozeti.

## Örnekler
```bash
# Webcam ile canlı
curl -X POST localhost:8080/stream/start -H 'content-type: application/json' -d '{"source":"0"}'
# RTSP
curl -X POST localhost:8080/stream/start -H 'content-type: application/json' \
  -d '{"source":"rtsp://192.168.1.10:8554/stream"}'
```

## Sorun Giderme
| Belirti | Çözüm |
|---|---|
| Dashboard boş video | `/stream/status` `running:true` mı? Kaynak doğru mu? |
| Event akmıyor | WS bağlantısı (tarayıcı konsolu); inference :8080 çalışıyor mu? |
| Mobil bağlanmıyor | `localhost` yerine LAN IP; emülatörde `10.0.2.2` |
| Eval "no_results" | Önce **Eval Çalıştır** / `POST /eval/run` |
