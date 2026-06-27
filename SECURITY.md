# Güvenlik Politikası — RoadGuard

## Desteklenen sürümler

RoadGuard aktif geliştirme aşamasındadır (TEKNOFEST 2026). Güvenlik düzeltmeleri
`main` dalı üzerinde uygulanır.

| Sürüm | Destek |
|---|---|
| `main` (en güncel) | ✅ |
| Önceki etiketler | ❌ |

## Açık bildirimi

Bir güvenlik açığı bulduysan **lütfen herkese açık bir issue açma.** Bunun yerine:

- Depo sahibine **özel** olarak ulaş (GitHub üzerinden özel güvenlik danışmanlığı /
  Security Advisory veya doğrudan iletişim).
- Açığı yeniden üretecek minimum adımları, etkilenen dosya/uç noktayı ve olası etkiyi paylaş.

Makul sürede yanıt verip düzeltmeyi `main`'e alacağız.

## Güvenlik duruşu (tasarım)

RoadGuard'ın inference API'si **ENV-GATED** sertleştirmelerle gelir — varsayılan (yerel
demo) davranışı bozmadan, üretimde ilgili ortam değişkenleri set edilince koruma devreye girer:

| Ortam değişkeni | Etki |
|---|---|
| `ROADGUARD_API_TOKEN` | Mutasyon uçlarına `X-RoadGuard-Token` başlık-auth'u (set ise zorunlu). |
| `ROADGUARD_API_PROTECT_READS` | Token ile birlikte set ise PII okuma uçlarını da korur (`/tracks*`, `/info`, MJPEG `?token=`). |
| `ROADGUARD_ALLOW_NET_SOURCE` | Set değilse `http://`/`file://`/serbest şema kaynaklar reddedilir (SSRF guard). |
| `ROADGUARD_CORS_ORIGINS` | CORS allowlist (varsayılan localhost; asla wildcard değil). |

Ayrıntı: [`docs/dagitim.md`](docs/dagitim.md) §3.5. Üretimde PII (canlı plaka/görüntü) için
ek olarak ters-proxy / ağ ACL / VPN katmanı önerilir.

## Kapsam

- **Kapsam içi:** inference API auth/CORS/SSRF, plaka/PII sızıntısı, DoS (eş-zamanlı akış sınırı),
  bağımlılık açıkları.
- **Kapsam dışı:** mock telekom servislerinin (QoD/NV) gerçek operatör auth'u (final ortamında
  Turkcell gateway'e bağlanır), yerel demo varsayılanları.
