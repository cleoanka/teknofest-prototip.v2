# AURA Benchmark — cuda0

- **Tarih:** 2026-06-26 00:54:34
- **Kaynak:** `video_1.mp4`
- **Cihaz (istek → çözülen):** `cuda` → `cuda:0`
- **Profil:** `v4-finetune`
- **AI modu:** `real`
- **İşlenen kare:** 150 (ısınma 5, ölçülen 145)

## Sonuçlar (ısınma sonrası)

| Metrik | Değer |
|---|---|
| Ortalama FPS (kararlı-hal) | **12.80** |
| Uçtan-uca FPS (ısınma dahil) | 11.78 |
| Kare-süresi ortalama | 78.11 ms |
| Kare-süresi p50 | 76.96 ms |
| Kare-süresi p95 | 103.75 ms |
| Toplam süre | 12.74 s |

> Not: Apple Silicon (MPS) sayıları **alt sınırdır**; sunucu CUDA'da FPS
> belirgin yüksektir. Gerçek dağıtım FPS'i için bu aracı CUDA sunucuda
> `--device cuda --profile server` ile koşun (bkz. docs/dagitim.md §5).
