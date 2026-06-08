# Çalıştırma — Uçtan Uca Demo Senaryosu

## Ne yapar
Kurulu sistemde tipik bir demo akışını adım adım gösterir.

## Gereksinimler
`./setup.sh` tamamlanmış olmalı.

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

### 5. Mobil (opsiyonel)
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
