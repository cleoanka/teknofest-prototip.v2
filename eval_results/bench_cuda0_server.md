# AURA Benchmark — cuda0 (server profili)

- **Tarih:** 2026-06-26
- **Kaynak:** `video_1.mp4`
- **Cihaz (istek → çözülen):** `cuda` → `cuda:0`
- **GPU:** NVIDIA GeForce RTX 5070 Laptop GPU — 36 SM × 128 = **4.608 CUDA çekirdeği**, 8 GB VRAM, Compute 12.0 (Blackwell)
- **Profil:** `server`
- **Dedektör:** yolo26l · imgsz 960
- **AI modu:** `real` (EasyOCR kurulu olmadığından mock OCR; OCR yalnızca plaka kırpığında çalıştığından FPS üzerindeki etkisi küçüktür)
- **İşlenen kare:** 150 (ısınma 5, ölçülen 145)

## Sonuçlar (ısınma sonrası)

| Metrik | Değer |
|---|---|
| Ortalama FPS (kararlı-hal) | **12.31** |
| Uçtan-uca FPS (ısınma dahil) | 10.29 |
| Kare-süresi ortalama | 81.23 ms |
| Kare-süresi p50 | 80.02 ms |
| Kare-süresi p95 | 93.09 ms |
| Toplam süre | 14.57 s |

> **Donanım notu:** RTX 5070 Laptop GPU — 4.608 CUDA çekirdeği (36 SM × 128), 8 GB VRAM, Compute Capability 12.0 (NVIDIA Blackwell). Torch 2.8.0+cu128.
> Bu sayılar GERÇEK ölçümlerdir (2026-06-26, `python tools/bench.py --source video_1.mp4 --device cuda --profile server --warmup 5 --max-frames 150`).
