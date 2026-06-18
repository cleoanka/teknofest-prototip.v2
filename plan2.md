# PLAN-2 — 5 Öğrenci Görev Dağılımı (AURA Takımı)

> 18.06.2026 · Salt-planlama. Eş-belge: **plan1.md** (teknik yol haritası). İş paketleri plan1'den gelir.
> Şartname: takım 2–5 kişi + (ops.) danışman; finalde sahada **max 3 yarışmacı**; bir iletişim sorumlusu (kaptan) zorunlu.
> Roller işlevsel tanımlıdır; gerçek isimleri (örn. Selman/Boray, Mustafa, +2) takım atar. Tartışma: kendi sentezim. **Codex ölü** (0-çıktı); **Gemini kısmi** (pro `403`, `gemini-2.5-flash` çalıştı). Detay §sonu.

## Rol Haritası (plan1 iş paketleriyle eşleşme)

| # | Rol | Sorumluluk | plan1 iş paketleri |
|---|---|---|---|
| **Ö1** | **YZ-Tespit & Eğitim Lideri** | YOLO26 detector, dataset/veri-dengeleme, eğitim hattı, mAP | 0.3 · **1.1** · 1.4 · 1.6 |
| **Ö2** | **Plaka & OCR** | LP dedektör + fast-plate-ocr + dewarp/SR + oy havuzu + dürüstlük zırhları | 0.2(plaka) · **1.3** |
| **Ö3** | **Sürücü Davranışı & Hız** | pose+domain sürücü modeli (sigara/telefon/kemer/yorgunluk), hız/swerving/risk | **1.2** · 1.5 · 0.2(davranış/hız) |
| **Ö4** | **5G / Backend / QoD** | CAMARA QoD+NV (gerçek sandbox), inference_api, qod/nv, A/B harness | **2.1 · 2.2 · 2.3** · 0.2(QoD) |
| **Ö5** | **Mobil + Rapor + Demo** (= İletişim Sorumlusu/Kaptan) | mobil uygulama, dashboard, FTR rapor, sunum, kanıt-izi | **0.1 · 0.4 · 0.5 · 0.6** · **2.4 · 2.5 · 2.6** |

> Çekirdek YZ (Ö1/Ö2/Ö3) %40 doğruluk puanını; Ö4 %40 QoD'u; Ö5 %20 rapor+sunum'u ve final-demo bütünlüğünü taşır.

---

## Haftalık Plan

### Hafta 1 — FTR Sprinti (18.06 → 28.06) · HERKES FTR'ye odak
| Öğrenci | Bu hafta |
|---|---|
| **Ö5 (Kaptan)** | `ftr_rapor_taslak.md` → DOCX (format), §1/§3/§5 yaz, diyagram göm, KYS yükleme (28.06 17:00), PR→main koordine (0.1/0.4/0.6) |
| **Ö1** | §2 Veri Seti bölümü (toplanan setler + manifest + dengeleme tablosu) + held-out mAP açıklaması (0.3) |
| **Ö2** | §4 plaka kanıtı (3/3 exact, CER, fast-plate-ocr; annotated mp4 + oy dökümü) (0.2) |
| **Ö3** | §4 davranış+hız kanıtı (makro-F1 1.0, swerving; kalibrasyon-bağımlı hız anlatımı) (0.2) |
| **Ö4** | §4 QoD A/B (+33/+51/+25, sentetik-set şeffaf notu) + §3 mimari/topoloji metni (0.2/0.5) |

**Teslimat:** Şablona uygun FTR + tüm kanıtlar, 28.06'dan önce KYS. (Çoğu kanıt zaten hazır → düşük risk.)

### Hafta 2–6 — YZ Derinleştirme (28.06 → 31.07) · sonuç beklerken
| Öğrenci | İş |
|---|---|
| **Ö1** | YOLO26 detector fine-tune (komite/Roboflow → custom_detector + held-out mAP) (1.1, 1.6) |
| **Ö2** | fastplate kalıcı; komite footage'da dewarp/SR A/B; CER raporu (1.3) |
| **Ö3** | driver-state fine-tune (seatbelt+smoking+phone → custom_driver, mAP, pose ile A/B) (1.2, 1.5) |
| **Ö4** | gerçek CAMARA sandbox başvurusu/erişimi; QoD+NV gerçek entegrasyon başlat (2.1 ön-hazırlık) |
| **Ö5** | mobil uygulama iskeletini geliştirmeye başla (NV-login + canlı kamera akışı) (2.4 ön-hazırlık) |

**Teslimat:** her model için HELD-OUT ölçüm + A/B (baseline'ı geçmiyorsa profilde kalır).

### Hafta 7+ — Final Hazırlığı (Ağu–Eyl) · finalist olunursa
| Öğrenci | İş |
|---|---|
| **Ö4** | gerçek 5G CAMARA QoD+NV canlı; QoD başarım deltası gerçek ağda ölç (2.1–2.3) |
| **Ö5** | mobil tam uygulama + canlı demo provası + sunum (2.4–2.6) |
| **Ö1/Ö2/Ö3** | en iyi modelleri sahaya hazırla (profil/ağırlık), kenar-durum sağlamlaştırma, kanıt-izi (2.6) |

---

## Final Saha Kadrosu (max 3 yarışmacı)
Öneri: **Ö4 (5G/QoD)** + **Ö5 (mobil/demo/kaptan)** + **bir YZ çekirdeği (Ö1 veya Ö3)** → canlı demo'nun üç bacağı (ağ + uygulama + model) sahada. Diğer 2 öğrenci dönüşümlü/destek.

## Koordinasyon Kuralları
- **Kaptan (Ö5):** KYS takibi (rapor/itiraz/form), tüm iletişim, teslim tarihleri.
- **Git akışı:** kişi başı feature-branch → yeşil+incelenmiş PR → main (asla doğrudan main). Tek-config kaynağı; eşik koda gömülmez (K-004).
- **Haftalık senkron:** her öğrenci held-out ölçümünü + A/B'sini paylaşır; varsayılan ancak ölçüm iyileştirirse değişir.
- **Danışman (varsa):** akademik yönlendirme; final taahhüdü + belge (şartname).

## Bağımlılıklar (kritik yol)
- FTR (Hafta 1) → herkes; gate.
- 2.4 mobil ← 2.1 QoD (Ö4) + WS /stream/events (mevcut).
- 1.1/1.2 eğitim ← komite verisi / Roboflow erişimi (gelmezse açık-kaynak + dürüst beyan).
- 2.2 QoD-kanıt ← 2.1 gerçek sandbox erişimi (erken talep et).

## Codex / Gemini Tartışması (güncellendi)
> **Gemini — ÇALIŞIYOR (`gemini-2.5-flash`):** 403 yalnız `gemini-3.1-pro-preview` modelinde; 2.5-flash ile mobil + final araştırması yapıldı. **Codex** 0-çıktı ile takıldı. Dağıtım kendi sentezimdir.
> **GÜNCELLEME (18.06):** Ö5'in mobil uygulaması **TEMEL HAZIR** (commit `1bbbf8c`, `tsc` temiz: NV sessiz giriş + canlı WS panosu + QoD histerezis). Hafta 7+ → gerçek CAMARA + cihazda canlı demo.
