<div align="center">

# 📋 FTR Rehberi — RoadGuard Final Tasarım Raporu

**Gönderilecek FTR'nin içinde ne var, bizden ne bekleniyor, neler eklenebilir?**

![puanlama](https://img.shields.io/badge/puanlama-%2540_YZ_%C2%B7_%2540_QoD_%C2%B7_%2520_rapor-1b5e20?style=flat-square)
![format](https://img.shields.io/badge/format-3--10_sayfa_%C2%B7_Arial_12-0d47a1?style=flat-square)
![durum](https://img.shields.io/badge/FTR_PDF-10_sayfa_%E2%9C%93-success?style=flat-square)

</div>

> [!NOTE]
> Bu dosya, **gönderilecek FTR'yi** (`FTR_GONDERILECEK.pdf`) açıklar: her bölümde şartnamenin
> ne istediği, bizde **şu an ne olduğu** (hangi kanıt/figür/sayı) ve **ne eklenebileceği**.
> FTR'nin kaynağı `FTR_GONDERILECEK.tex`'tir; "biz düzenleriz" derken onu düzenleyip PDF'i
> yeniden üretebilirsiniz (aşağıda komut).

---

## 🎯 Bir bakışta

| | |
|---|---|
| **Puanlama** (şartname Tablo 1) | **%40** YZ doğruluk/hassasiyet · **%40** 5G QoD entegrasyonu · **%20** rapor + sunum |
| **FTR teslimi** | KYS sistemi, saat 17:00'a kadar (repo bağlayıcı tarihi **28.06.2026**; şartname PDF'i 14.06 yazar — ertelendi) |
| **Format** (zorunlu) | 3–10 sayfa · Arial 12 / başlık 14 bold · 1.15 satır · iki yana yaslı · kenar üst 2.8 / diğer 2.5 cm · **Kapak + İçindekiler ayrı 2 sayfa** |
| **Onur kuralı** (şartname 4.5) | "Raporladığı her hedefin otomatik üretildiğini kanıtlamakla yükümlü; **kanıtlanamayan hedef değerlendirilmez**" → RoadGuard'ın K-004 zırhı + kanıt izi bunu karşılar |

---

## 📑 Bölüm bölüm: ne bekleniyor · bizde ne var · ne eklenebilir

### 1. Proje Özeti `(5p)`
- **Beklenen:** yürütülen faaliyetlerin özeti.
- **Bizde var:** FTR §1 — 5 ana faaliyet + gerçek/mock sınırı + kalite güvencesi (815 geçen / 1 atlanan test, CI).
- **Eklenebilir:** takım adı/ID, bir-cümlelik "yenilikçi yan" vurgusu (poz-hibrit + dürüstlük zırhı).

### 2. Veri Seti Oluşturulması `(20p)`
- **Beklenen:** toplama, etiketleme, **dengeleme (data balancing)**, **augmentation**, train/val/test oranları **gerekçeli**; açık veri setleri **kaynakçada**.
- **Bizde var:** FTR §2.1–§2.5 + **2 grafik** — `fig_veri_dengesi` (4 set dağılımı), `fig_split` (%80/%10/%10). Dengesizlik oranı ölçülü (seatbelt 1,27). 4 set CC BY 4.0, kaynakça §5'te + tam liste `kaynakca.md`.
- **Eklenebilir:** komite TOGG verisi gelince gerçek domain dağılımı; 4 setin hepsi için ayrı denge tablosu (şu an örnek seatbelt).

### 3. Yapay Zekâ Çözümü `(50p)`
- **3.1 Problemin Analizi `(15p)`** — *Beklenen:* ışık/blur/oklüzyon vb. + neden bu yol? *Bizde:* Tablo 1 (problem→çözüm) + ID-merkezli karar gerekçesi.
- **3.2 Çözüm Mimarisi `(15p)`** — *Beklenen:* kuşbakışı diyagram, ham video→çıktı. *Bizde:* `fig_mimari` (kaskad boru hattı) + gerçek/mock sınırı. *Eklenebilir:* render edilmiş `docs/diagrams/*.mmd` (yüksek çözünürlük), mobil arayüz ekran görüntüsü.
- **3.3 Çözüm Detayları `(20p)`** — *Beklenen:* algoritmalar, ağ mimarileri, ön/son işleme, çerçeveler. *Bizde:* YOLO26 + ByteTrack + fast-plate-ocr + YOLO26-pose hibrit + Kalman/EMA + CLAHE + dürüstlük zırhları + yazılım/donanım yığını.

### 4. Çözümün Sınanması `(20p)`
- **Beklenen:** **Accuracy / Precision / Recall / F1 / FPS** tablo **ve grafik**; "neden güveniyoruz?" verilerle.
- **Bizde var:** FTR §4.1–§4.7 — **held-out mAP/P/R/F1 tablosu + 4 grafik** (`fig_tespit_map`, `fig_custom_prf1`, `fig_plaka_ab`) + davranış P/R/F1 (makro-F1 1,0) + plaka A/B (**baseline 2/3 → production 3/3, CER 0,0, 0 yanlış-onay**) + FPS + **§4.9 "neden güveniyoruz"** (5 nicel gerekçe). Tüm sayılar `eval_results/` + `weights/custom_*_s.metrics.json`'dan **türetilmiş** (elle yazılmamış).
- **Eklenebilir:** QoD A/B'yi güncel `--qod-comparison` koşusuyla doldur (delta koşuya bağlı); komite gerçek-hız GT'siyle mutlak hız MAE/MAPE; geniş etiketli sette istatistiksel mAP.

### 5. Kaynakça `(5p)`
- **Beklenen:** tüm akademik/teknik kaynaklar (repo, makale, doküman) detaylı.
- **Bizde var:** FTR §5 (14 çekirdek kaynak) + **tam `kaynakca.md`** (47 kaynak, izlenebilirlik tablosu, lisans envanteri).

---

## 🗂️ Dosya haritası

| Dosya | Ne |
|---|---|
| **[`FTR_GONDERILECEK.pdf`](FTR_GONDERILECEK.pdf)** | **Gönderilecek FTR** (10 sayfa, format-uyumlu) |
| [`FTR_GONDERILECEK.tex`](FTR_GONDERILECEK.tex) | PDF'in **düzenlenebilir** LaTeX kaynağı |
| [`tools/make_ftr_figures.py`](../../tools/make_ftr_figures.py) | grafikleri **gerçek ölçümlerden** üretir |
| [`docs/figures/`](../figures/) | Üretilen grafikler (`fig_*.png`) |
| [`kaynakca.md`](kaynakca.md) | Tam kaynakça (FTR §5 havuzu) |
| [`ftr.md`](../../ftr.md) | FTR doldurma rehberi + format notu |

---

## 🔧 PDF'i yeniden üretme

```bash
# 1) Grafikleri üret (gerçek ölçümlerden — sayı uydurmaz)
python tools/make_ftr_figures.py

# 2) PDF'i derle (TOC için iki kez; Arial + Türkçe için xelatex)
xelatex FTR_GONDERILECEK.tex && xelatex FTR_GONDERILECEK.tex
```

> [!TIP]
> Metni düzenlemek en kolay `FTR_GONDERILECEK.tex` üzerinden. Sayı/figür değiştirmek için
> önce `tools/make_ftr_figures.py`'yi (veya kaynak JSON'ları) güncelleyip grafikleri yeniden
> üretin — böylece **rapor ile kanıt her zaman tutarlı** kalır (K-004).

---

## ✅ Teslim öncesi kontrol listesi

- [ ] 3–10 sayfa (kapak+içindekiler+kaynakça dahil) — *şu an **10** ✓*
- [ ] Arial 12 / başlık 14 bold · 1.15 satır · iki yana yaslı
- [ ] Kenar üst 2.8 / alt-sağ-sol 2.5 cm
- [ ] Kapak ve İçindekiler **ayrı 2 sayfa** ✓
- [ ] **Takım Adı / Takım ID / Başvuru ID** dolduruldu (kapakta boş alanlar)
- [ ] Tüm sayılar `eval_results/` + `weights/custom_*_s.metrics.json`'dan (uydurma yok) ✓
- [ ] Açık veri setleri kaynakçada (CC BY 4.0) ✓
- [ ] KYS'ye teslim tarihinden önce yüklendi

---

## ➕ Neler eklenebilir (final etabına doğru)

| Fırsat | Etki | Durum |
|---|---|---|
| QoD A/B'yi baskılı OFF koşusuyla doldur (`--qod-comparison`) | %40 QoD kanıtı güçlenir | harness hazır |
| Komite gerçek-hız GT → mutlak hız MAE/MAPE | hız doğruluğu sayısı | harness hazır |
| Geniş etiketli set → istatistiksel mAP | tespit doğruluğu | komite verisi |
| Mobil app ekran görüntüsü + canlı 5G QoD/NV demo | final sunumu | iskelet hazır |
| `docs/diagrams/*.mmd` yüksek-çözünürlük render → §3.2 | mimari görseli | mermaid kaynak hazır |

---

*Hazırlık: ULTRAPLAN W2 (rapor + cila). Tüm grafikler ve sayılar depo ölçüm artefaktlarından
yeniden-üretilebilir. Onur zırhı (K-004 = şartname 4.5): kanıtlanamayan hedef raporlanmaz.*
