> 📂 **services/** · Mikroservisler · [⬅ repo kökü](../README.md)

<div align="center">

# 🧩 `services/` — Mikroservisler

![inference_api](https://img.shields.io/badge/inference__api-8080-2ea44f?style=flat-square)
![qod_mock](https://img.shields.io/badge/qod__mock-8081-blue?style=flat-square)
![nv_mock](https://img.shields.io/badge/nv__mock-8082-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-uvicorn-009688?style=flat-square)

</div>

---

## 🗂️ Servisler

| Servis | Port | Gerçek/Mock | Açıklama |
|---|---|---|---|
| `inference_api/` | 8080 | ✅ **Gerçek YZ** | Pipeline'ı koşturur; MJPEG + WS yayar; dashboard'u serve eder |
| `qod_mock/` | 8081 | 🟡 Mock (CAMARA sözleşmesi) | QoD session aç/sorgula/sil |
| `nv_mock/` | 8082 | 🟡 Mock (NV sözleşmesi) | Sessiz numara doğrulama |

---

## 🧠 Gerçek/Mock sınırı

> [!IMPORTANT]
> `inference_api` tüm CV/YZ hattını gerçek çalıştırır. `qod_mock` ve `nv_mock` gerçek telekom API sözleşmesini birebir taklit eder — final ortamında yalnızca endpoint/credential CAMARA/operatör gateway'ine çevrilir, sözleşme değişmez.

```mermaid
flowchart TD
    A["inference_api · :8080<br/>✅ Gerçek YZ"] -->|"MJPEG + WS · dashboard"| D["İstemci / Dashboard"]
    B["qod_mock · :8081<br/>🟡 CAMARA sözleşmesi"] -->|"QoD session aç/sorgula/sil"| G["Final: CAMARA gateway"]
    C["nv_mock · :8082<br/>🟡 NV sözleşmesi"] -->|"Sessiz numara doğrulama"| G

    classDef real fill:#2ea44f,stroke:#1a7f37,color:#fff;
    classDef mock fill:#fff3cd,stroke:#d4a72c,color:#000;
    class A real;
    class B,C mock;
```

---

## 🚀 Çalıştırma

```bash
./run.sh          # üçünü birden kaldırır (macOS/Linux)
.\run.ps1         # Windows
```

<details>
<summary>veya tek tek</summary>

```bash
uvicorn services.inference_api.main:app --port 8080
uvicorn services.qod_mock.main:app --port 8081
uvicorn services.nv_mock.main:app --port 8082
```

</details>

---

> [!TIP]
> Tüm endpoint'ler: `docs/api_referans.md` · OpenAPI: http://localhost:8080/docs
