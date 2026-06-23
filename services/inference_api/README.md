> 📂 **aura/** · Gerçek YZ Mikroservisi · [⬅ repo kökü](../../README.md)

# `inference_api/` — Gerçek YZ Mikroservisi (:8080)

<div align="center">

![Servis](https://img.shields.io/badge/servis-inference__api-blue?style=flat-square)
![Port](https://img.shields.io/badge/port-8080-success?style=flat-square)
![Framework](https://img.shields.io/badge/framework-FastAPI-009688?style=flat-square)
![Akış](https://img.shields.io/badge/akış-iki--kanal-orange?style=flat-square)

</div>

FastAPI. AURA pipeline'ını arka plan thread'inde koşturur ve **iki-kanal** akış yayar.

> [!IMPORTANT]
> **Onur zırhı K-004** — Bu dosya yalnızca görsel olarak zenginleştirilmiştir. Hiçbir sayı, komut, dosya-yolu, bağlantı veya iddia değiştirilmemiş; tüm bölümler aynen korunmuştur.

---

## 🧠 İki-kanal mimari

- `GET /stream/video` → MJPEG (ham veya `?bbox=true` ile server-side çizimli)
- `WS /stream/annotations` → kare başına bbox koordinatları (dashboard canvas çizer)
- `WS /stream/events` → `AuraEvent` stream'i (durum değişimleri)

Dashboard bbox toggle'ı **client-side** yapar (canvas temizle/çiz) → sunucuya gidiş-geliş yok.

```mermaid
flowchart LR
    P["AURA pipeline<br/>(arka plan thread)"] --> SM["StreamManager<br/>(state.py)"]
    SM -->|MJPEG| V["GET /stream/video<br/>ham veya ?bbox=true"]
    SM -->|bbox koordinatları| A["WS /stream/annotations"]
    SM -->|AuraEvent| E["WS /stream/events"]
    A --> C["Dashboard canvas<br/>(client-side toggle)"]
    V --> C
    classDef src fill:#009688,stroke:#00695c,color:#fff;
    classDef sink fill:#1565c0,stroke:#0d47a1,color:#fff;
    class P,SM src;
    class V,A,E,C sink;
```

---

## 🔌 Endpoint grupları

| Grup | Endpoint'ler |
|---|---|
| Sistem | `GET /health`, `GET /info` |
| Kamera | `GET /cameras` (OpenCV enum + platform isimleri) |
| Akış | `POST /stream/start\|stop`, `PATCH /stream/config`, `GET /stream/status`, `GET /stream/video`, `WS /stream/annotations\|events` |
| Track | `GET /tracks`, `GET /tracks/{id}`, `GET /tracks/{id}/history` |
| Eval | `POST /eval/run`, `GET /eval/results`, `GET /eval/results/export` |
| Config | `GET /config`, `PATCH /config` |

---

## 🗂️ Yapı

| Dosya | Sorumluluk |
|---|---|
| `main.py` | App assembly (lifespan, router'lar, statik dashboard serve) |
| `state.py` | `StreamManager` — pipeline worker + MJPEG buffer + WS push |
| `models.py` | İstek/yanıt DTO'ları |
| `routers/` | system, cameras, stream, tracks, eval, config |

---

## ⚙️ Env

| Değişken | Etki |
|---|---|
| `AURA_AUTOSTART=0` | Başlangıçta otomatik stream başlatma (varsayılan 1) |
| `AURA_CAMERA_PROBE=0` | `/cameras` donanım taramasını atla (CI/başsız) |

---

## 🚀 Çalıştırma

```bash
uvicorn services.inference_api.main:app --port 8080 --reload
```
