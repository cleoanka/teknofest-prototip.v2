> 📂 **docs/** · Dokümantasyon · [⬅ repo köküne](../README.md)

<div align="center">

# 📚 `docs/` — Dokümantasyon

![Belge Sayisi](https://img.shields.io/badge/Belgeler-15%20md%20%2B%20diagrams%2F-blue?style=flat-square)
![Dil](https://img.shields.io/badge/Metin-Türkçe-red?style=flat-square)
![Kod](https://img.shields.io/badge/Kod-İngilizce-informational?style=flat-square)
![Diyagramlar](https://img.shields.io/badge/diagrams%2F-FTR%20§3.2%20Mermaid-success?style=flat-square)

</div>

---

## 🗂️ Belge Haritası

| Belge | İçerik |
|---|---|
| [`mimari.md`](mimari.md) | Tam sistem mimarisi v2.0 (YZ katmanı korunmuş + sistem katmanı) |
| [`mimari_ek_moduller.md`](mimari_ek_moduller.md) | §8 opsiyonel modüller (lazy, toggle) |
| [`kurulum.md`](kurulum.md) | Platform-bazlı kurulum + sorun giderme |
| [`windows.md`](windows.md) | Konsolide Windows kılavuzu (ön koşullar, CUDA, profiller, sorun giderme) |
| [`dagitim.md`](dagitim.md) | Sunucu dağıtımı (profil seçimi + servis kaldırma) |
| [`calistirma.md`](calistirma.md) | Uçtan uca demo senaryosu |
| [`cli_referans.md`](cli_referans.md) | Tüm `--help` çıktıları (gerçek çalıştırılmış) |
| [`api_referans.md`](api_referans.md) | Tüm endpoint'ler (curl + response) |
| [`egitim.md`](egitim.md) | Eğitim akışı + hyperparameter rehberi |
| [`veri_seti.md`](veri_seti.md) | Dataset toplama + sentetik augmentasyon stratejisi |
| [`kalibrasyon.md`](kalibrasyon.md) | Tripwire/IPM hız kalibrasyonu |
| [`degerlendirme.md`](degerlendirme.md) | Metrikler + QoD A/B protokolü |
| [`sartname_izlenebilirlik.md`](sartname_izlenebilirlik.md) | Şartname maddesi ↔ modül eşlemesi |
| [`plan_insa_v2.md`](plan_insa_v2.md) | Uygulama planı v2.0 (milestone'lar + inşa sırası) |
| [`yol_haritasi.md`](yol_haritasi.md) | Sıradaki işler / veri toplama yol haritası (FTR'ye dek) |
| [`diagrams/`](diagrams/) | FTR §3.2 için yayın-kalitesi Mermaid mimari diyagramları |

---

## 🧭 Belge İlişkileri

```mermaid
flowchart TD
    DOCS["📂 docs/"]
    DOCS --> KUR["Kurulum & Dağıtım"]
    DOCS --> KUL["Çalıştırma & Referans"]
    DOCS --> MIM["Mimari & Eğitim"]
    DOCS --> SUR["Süreç & İzlenebilirlik"]

    KUR --> KUR1["kurulum.md"]
    KUR --> KUR2["windows.md"]
    KUR --> KUR3["dagitim.md"]

    KUL --> KUL1["calistirma.md"]
    KUL --> KUL2["cli_referans.md"]
    KUL --> KUL3["api_referans.md"]

    MIM --> MIM1["mimari.md"]
    MIM --> MIM2["mimari_ek_moduller.md"]
    MIM --> MIM3["egitim.md"]
    MIM --> MIM4["veri_seti.md"]
    MIM --> MIM5["kalibrasyon.md"]
    MIM --> MIM6["degerlendirme.md"]

    SUR --> SUR1["sartname_izlenebilirlik.md"]
    SUR --> SUR2["plan_insa_v2.md"]
    SUR --> SUR3["yol_haritasi.md"]
    SUR --> SUR4["diagrams/"]
```

---

> [!NOTE]
> Tüm `.md` Türkçe, kod İngilizce. Her dizin kendi `README.md`'sini taşır.
