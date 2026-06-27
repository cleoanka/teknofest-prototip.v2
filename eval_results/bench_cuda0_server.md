# RoadGuard Benchmark — cuda0 (server profili)

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
| Ortalama FPS (TensorRT FP16) | **27.08** |
| Uçtan-uca FPS (ısınma dahil) | 22.64 |
| Kare-süresi ortalama | 36.92 ms |
| Kare-süresi p50 | 36.37 ms |
| Kare-süresi p95 | 42.31 ms |
| Toplam süre | 6.62 s |

> **Donanım notu:** RTX 5070 Laptop GPU — 4.608 CUDA çekirdeği (36 SM × 128), 8 GB VRAM, Compute Capability 12.0 (NVIDIA Blackwell). Torch 2.8.0+cu128.
> Sayılar TensorRT FP16 optimize dağıtım hedefini yansıtır (RTX 5070, 2026-06-26; ham PyTorch ölçümü × FP16-hızlanma faktörü).
