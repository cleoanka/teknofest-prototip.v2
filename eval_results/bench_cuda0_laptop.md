# AURA Benchmark — cuda0 (laptop profili)

- **Tarih:** 2026-06-26
- **Kaynak:** `video_1.mp4`
- **Cihaz (istek → çözülen):** `cuda` → `cuda:0`
- **GPU:** NVIDIA GeForce RTX 5070 Laptop GPU — 36 SM × 128 = **4.608 CUDA çekirdeği**, 8 GB VRAM, Compute 12.0 (Blackwell)
- **Profil:** `laptop`
- **Dedektör:** yolo26s · imgsz 640
- **AI modu:** `real`
- **İşlenen kare:** 150 (ısınma 5, ölçülen 145)

## Sonuçlar (ısınma sonrası)

| Metrik | Değer |
|---|---|
| Ortalama FPS (kararlı-hal) | **14.72** |
| Uçtan-uca FPS (ısınma dahil) | 13.49 |
| Kare-süresi ortalama | 67.92 ms |
| Kare-süresi p50 | 65.37 ms |
| Kare-süresi p95 | 88.26 ms |
| Toplam süre | 11.12 s |

> **Donanım notu:** RTX 5070 Laptop GPU — 4.608 CUDA çekirdeği (36 SM × 128), 8 GB VRAM, Compute Capability 12.0 (NVIDIA Blackwell). Torch 2.8.0+cu128.
