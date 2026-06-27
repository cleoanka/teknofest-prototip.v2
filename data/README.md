> 📂 **data/** · Veri Katmani · [⬅ repo koku](../README.md)

<div align="center">

# 🗂️ `data/` — Veri

![samples](https://img.shields.io/badge/samples-deterministik%20sentetik-blue?style=flat-square)
![GT%20plaka](https://img.shields.io/badge/GT%20plaka-34TC8532-success?style=flat-square)
![ornek.mp4](https://img.shields.io/badge/ornek.mp4-90%20kare%20%C2%B7%203%20ara%C3%A7-informational?style=flat-square)
![raw%2Fprocessed](https://img.shields.io/badge/raw%20%2F%20processed-.gitignore-lightgrey?style=flat-square)

</div>

---

## 📁 Dizin Yapisi

```mermaid
flowchart TD
    DATA["data/"]
    DATA --> SAMP["samples/<br/>sentetik + gercek-video GT"]
    DATA --> RAW["raw/<br/>ham, etiketsiz (.gitignore)"]
    DATA --> PROC["processed/<br/>islenmis/etiketli setler (.gitignore)"]

    SAMP --> ORN["ornek.mp4 + ornek_gt.json"]
    SAMP --> VID["video_{1,2,3}_gt.json"]
    PROC --> CLS["license_plate/ · phone/<br/>smoking/ · seatbelt/"]

    classDef tracked fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef ignored fill:#f3f4f6,stroke:#9ca3af,color:#374151;
    class SAMP,ORN,VID tracked;
    class RAW,PROC,CLS ignored;
```

| Dizin | Icerik | Durum |
|---|---|---|
| `samples/` | Sentetik senaryo + gercek-video GT | ✅ izlenir |
| `raw/` | Ham/etiketlenmemis egitim verisi | 🟡 `.gitignore` |
| `processed/` | Islenmis/etiketli egitim setleri | 🟡 `.gitignore` |

---

## 🎬 `samples/`

Bootstrap tarafından üretilen **deterministik sentetik trafik senaryosu**:
- `ornek.mp4` — 90 kare, 3 araç, farklı şeritler/zamanlar, plaka varyasyonları (`.gitignore`'lu, yeniden üretilebilir).
- `ornek_gt.json` — kare-bazlı ground-truth (bbox, vehicle_class, plaka, sürücü durumu, hız).

Yeniden üret:
```bash
python -m roadguard.synthetic --out data/samples --frames 90
```

Ek olarak `samples/` gerçek test videolarının **video-düzeyi** ground-truth'unu da tutar:
- `video_{1,2,3}_gt.json` — gerçek test videosu GT'si (12 Haz 2026). Kare-kare bbox
  etiketi yoktur; **plaka ve davranış video-düzeyinde** verilir (GT plaka `34TC8532`).
  Videoların kendisi (`video_1.mp4` …) repo kökündedir.

> [!NOTE]
> Gerçek test videoları için **kare-kare bbox etiketi yoktur**; plaka ve davranış yalnızca **video-düzeyinde** verilir (GT plaka `34TC8532`).

<details>
<summary><b>📄 GT yapısı (<code>ornek_gt.json</code>)</b></summary>

```json
{ "video": "ornek.mp4", "fps": 30, "width": 640, "height": 360,
  "frames": [ { "frame": 0, "objects": [
    { "id": 1, "bbox": [x1,y1,x2,y2], "vehicle_class": "car",
      "plate": "34ABC123", "driver": {"phone": true, ...}, "speed_kmh": 40.0 } ] } ] }
```

</details>

> [!IMPORTANT]
> Gerçek TOGG veri seti geldiğinde `data/samples/` üzerine yazılır; `data/raw/` (ham, etiketsiz veri) `.gitignore`'ludur.

---

## 🧱 `raw/`

Ham/etiketlenmemiş eğitim verisi (git'e dahil edilmez). Eğitim akışı: `docs/veri_seti.md`.

---

## ⚙️ `processed/`

İşlenmiş/etiketli eğitim setleri (sınıf-başına klasör: `license_plate/`, `phone/`,
`smoking/`, `seatbelt/`). `train fetch` ile toplanır, `train dataset` ile YOLO formatına
hazırlanır. `.gitignore`'ludur (büyük). Kaynak/lisans manifesti: `train/datasets.yaml`.

```mermaid
flowchart LR
    FETCH["train fetch<br/>(toplama)"] --> DSET["train dataset<br/>(YOLO formati)"]
    DSET --> READY["processed/<br/>license_plate · phone<br/>smoking · seatbelt"]
```
