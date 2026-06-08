# `nv_mock/` — Number Verification Mock (:8082)

Operatör **Number Verification** (sessiz doğrulama) sözleşmesini taklit eder.
SMS/OTP yoktur; SIM/şebeke bağı kontrol edilir. Mobil uygulama sessiz girişte bunu çağırır.

## Endpoint'ler
| Method | Path | Açıklama |
|---|---|---|
| `POST` | `/verify` | `{phone_number, sim_token?}` → `{verified, latency_ms, phone_number}` |
| `GET` | `/health` | Servis durumu |

**Mock kuralı:** geçerli `sim_token` + Türk numarası (`+90`/`0`) → `verified: true`.
Final ortamında endpoint operatör NV API'sine çevrilir.

## Örnek
```bash
curl -X POST localhost:8082/verify -H 'content-type: application/json' \
  -d '{"phone_number":"+905551112233","sim_token":"abc"}'
# → {"verified": true, "latency_ms": 40, "phone_number": "+905551112233"}
```

Çalıştır: `uvicorn services.nv_mock.main:app --port 8082`
