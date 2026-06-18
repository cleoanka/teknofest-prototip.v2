# PLAN-1 — Yapılabilecekler (AURA Teknik Yol Haritası)

> 18.06.2026 · Salt-planlama (projeye dokunulmadı). Kaynak: `gozlem.md` (4-repo + SOTA analizi) + repo-içi ölçümler.
> Tartışma: kendi sentezim + Codex ikinci-görüş (planlama-Codex koşuyor; geldiğinde §6'ya eklenecek). **Gemini lisans-403 nedeniyle kullanılamadı.**
> Eş-belge: **plan2.md** (5 öğrenciye görev dağılımı).

## 0. Öncelik İlkesi & Takvim
- **Puanlama:** %40 YZ doğruluk · %40 QoD (yalnız kritik anda bant↑) · %20 rapor+sunum.
- **18.06 → 28.06 (10 gün): FTR teslimi = MUTLAK gate.** Çoğu kanıt hazır → riski düşük.
- **28.06 → 31.07:** sonuç bekleme + YZ derinleştirme (komite verisi gelirse eğitim).
- **Ağu–Eyl: Final** (mobil + gerçek 5G/QoD/NV canlı demo, sahada max 3 yarışmacı).
- **K-004 (değişmez):** ölçülen sayı; uydurma yok; belirsizde dürüst `pending`; videoya-özel sabit yok.

## 1. Mevcut Durum (W1 sonrası — neyin HAZIR / EKSİK olduğu)
**Hazır (ölçülü):** plaka **3/3 exact** (fast-plate-ocr), davranış **makro-F1 1.0**, araç %100, held-out yolo26l mAP50-95 0.537, QoD A/B +33/+51/+25pp (sentetik), 259 test, FTR rapor taslağı (`ftr_rapor_taslak.md`) + Mermaid diyagramlar + metrik harness, mock CAMARA (QoD/NV), stabilite zırhları.
**Eksik / Riskli:** ① özel-model EĞİTİMİ ertelendi → zorunlu sınıf (plaka/sigara) held-out mAP yok; ② QoD A/B yalnız sentetik sette (gerçek 3 videoda kare-düzeyi GT yok); ③ mobil İSKELET; ④ gerçek CAMARA sandbox test edilmedi; ⑤ küçük held-out (3 video).

---

## FAZ 0 — FTR'yi Geçmek (ŞİMDİ → 28.06) · EN YÜKSEK ÖNCELİK
| # | İş | Kabul kriteri | Durum |
|---|---|---|---|
| 0.1 | `ftr_rapor_taslak.md` → **DOCX** (Arial 12, başlık Arial Black 14, 1.15, 3–10 sayfa, kapak+içindekiler ayrı) | Şablona uyumlu .docx; KYS'ye yüklenebilir | taslak hazır → biçimlendir |
| 0.2 | §4 Sınama tabloları (P/R/F1, plaka 3/3 CER0, held-out mAP 0.537, QoD A/B, FPS) + **PR-curve görseli** | Gerçek sayılar + grafik | metrikler hazır |
| 0.3 | §2 Veri Seti: toplanan setler (seatbelt 3104/smoking 36/phone 659) + manifest + dengeleme + açık-kaynak lisansları | `dataset --report` tablosu + kaynakça | hazır |
| 0.4 | §3 Mimari diyagramları göm (docs/diagrams Mermaid → PNG) | Kuşbakışı + topoloji + plaka-karar | hazır |
| 0.5 | **Dürüst sınır beyanı:** zorunlu-sınıf mAP'i komite verisine bağlı; QoD sentetik-sette ölçülü | §4.1/§4.2'de açık | hazır |
| 0.6 | Son kontrol: 259 test + ruff/black + CI yeşil; `feat/ultraplan-w1` → **PR → main** (yeşil+incelenmiş) | PR açık, gözden geçirilmiş | branch push'lu |

**Çıktı:** Şablona uygun FTR + tüm kanıtlar; 28.06 17:00'dan önce KYS.

---

## FAZ 1 — YZ Derinleştirme (28.06 → 31.07) · %40 YZ puanını yükseltir
| # | İş | Kabul kriteri |
|---|---|---|
| 1.1 | **YOLO26 detector fine-tune** (komite TOGG verisi / Roboflow araç+plaka setleri → `custom_detector.pt`, taban=yolo26 s+l) | held-out mAP raporu (`model.val`); v4-baseline'ı geçer |
| 1.2 | **Driver-state fine-tune** (seatbelt 3104 + smoking/phone setleri → `custom_driver.pt`, backend=yolo) | zorunlu sınıf (phone/smoking/seatbelt) held-out mAP; pose-hibrit ile A/B |
| 1.3 | **Plaka:** fast-plate-ocr kalıcı; komite footage'ında dewarp+SR yeniden A/B (bu footage'da KIRMIZI çıktı — overfit etme) | CER düşüşü ölçülü VEYA dürüst "gerekmedi" |
| 1.4 | **İstatistiksel mAP/PR** (geniş etiketli set → `aura.eval --map`) | mAP50/50-95 + PR-eğrisi; §4 güçlenir |
| 1.5 | **Hız doğrulama** (komite gerçek-hız GT gelince MAE/MAPE; harness hazır) | hız MAE/MAPE tablosu |
| 1.6 | Tabela/hız-limiti (speed_limit_* sınıfları) custom dataset retrain | `SPEED_LIMIT_VIOLATION` gerçek üretir |

**İlke:** her eğitim sonrası HELD-OUT ölçüm + A/B (baseline'ı geçmiyorsa profilde kalır, varsayılan değişmez).

---

## FAZ 2 — Final Hazırlığı (Ağu–Eyl) · %40 QoD + canlı demo
| # | İş | Kabul kriteri |
|---|---|---|
| 2.1 | **Gerçek CAMARA QoD entegrasyonu** (Turkcell/operatör sandbox): mock → gerçek endpoint/credential | gerçek QoS-profil session aç/sorgula/sil |
| 2.2 | **QoD başarım kanıtı** (gerçek ağda A/B: TOGG yaklaşınca bant↑ → analiz doğruluğu↑) | ölçülen delta (gerçek, sentetik değil) |
| 2.3 | **Number Verification** gerçek entegrasyon (mobil veri hattı, sessiz) | gerçek SIM↔numara doğrulama |
| 2.4 | **Mobil tam uygulama** (Expo): NV-login → canlı kamera → QoD-tetikli yüksek çözünürlük → tespit gösterimi (WS /stream/events) | sahada telefonda çalışır demo |
| 2.5 | **Canlı demo provası** + sunum (kısa anlatım + yarışma etabı) | kesintisiz uçtan-uca prova |
| 2.6 | **Kanıt-izi (4.5):** her hedef otomatik üretildi (`--save-events` JSONL + annotated mp4) | jüriye gösterilebilir iz |

---

## 3. Çapraz-Kesen İyileştirmeler (her faza paralel)
- Dashboard cila (plaka CONFIRMED/pending rozet, QoD Δ paneli, swerving/risk vurgusu) — W1'de yapıldı, finalde genişlet.
- Docs güncel (mimari, izlenebilirlik 256-test, dağıtım).
- Perf: **CUDA sunucuda gerçek FPS** ölç (`tools/bench.py`) — MPS sayıları alt-sınır.
- Test coverage genişlet; CI pinli; packaging (tek-komut setup).

## 4. Risk Yönetimi
| Risk | Önlem |
|---|---|
| Komite verisi gelmezse | Açık-kaynak köprü + boru-hattı (tek komut retrain) + dürüst beyan |
| Gerçek CAMARA erişimi gecikirse | Mock sözleşme birebir → finalde yalnız endpoint; sandbox erken talep |
| Plaka komite-footage'da zor | fastplate + (gerekirse) SR/dewarp A/B; asla yanlış-onay (pending) |
| Mobil yetişmezse | Dashboard'u mobil-uyumlu yedek; çekirdek demo web |

## 5. Onur / Kalite İlkeleri
Ölçülen sayı (uydurma yok) · belirsizde pending · videoya-özel sabit yok · her değişiklik test+A/B · main'e yalnız yeşil+incelenmiş PR.

## 6. Codex / Gemini Tartışması
> **Gemini:** lisans `403 SUBSCRIPTION_REQUIRED` → kullanılamadı. **Codex** (salt-okunur planlama danışması) koşuyor; takeaway'leri bittiğinde buraya eklenecek. Karar (bu plan) kendi sentezimdir; Codex teyit/zenginleştirme amaçlı.
