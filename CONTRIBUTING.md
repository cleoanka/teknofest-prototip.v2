# Katkı Rehberi — RoadGuard

RoadGuard'a katkıda bulunmak istediğin için teşekkürler. Bu rehber, geliştirme
ortamını kurmanı ve değişikliklerini projenin kalite standartlarına uygun şekilde
göndermeni sağlar.

## 🚀 Geliştirme ortamı

```bash
# 1. Bağımlılıklar + model ağırlıkları (tek komut, CPU/MPS/CUDA otomatik)
python bootstrap.py

# 2. Paketi düzenlenebilir kur
pip install -e .

# 3. Hızlı sağlık kontrolü
python -m roadguard.smoke          # uçtan uca pipeline smoke testi
python tools/doctor.py             # ortam/ağırlık hazırlık raporu
```

## ✅ Göndermeden önce (kalite kapısı)

CI bu üç adımı çalıştırır; PR açmadan önce yerelde geçmeleri gerekir:

```bash
ruff check .                       # statik denetim
black --check .                    # biçim
pytest -m "not integration"        # birim/perf testleri (~815 test, model gerektirmez)
```

- Yeni davranış eklediysen **test ekle** (mevcut testlerin desenini izle, `tests/` altında).
- Yeni CLI bayrağı/komut eklediysen `--help` çıktısını ve ilgili `docs/` sayfasını güncelle.
- Performans-kritik kodda mikro-benchmark mantığını koru (sıcak döngüde gereksiz tahsis yok).

## 🛡️ Onur ilkesi (K-004)

RoadGuard'ın temel tasarım ilkesi **kanıtlanamayan sonucun raporlanmamasıdır**:

- Videoya-özel sabit **kullanma**; karar eşikleri oran/ölçek-temelli olmalı.
- Belirsizlikte sistem **uydurmak yerine çekimser kalır** (ör. plaka `pending`).
- Rapora/koda giren her metrik izlenebilir bir ölçüme dayanmalı.

## 🌿 Dal ve commit düzeni

- `main` korunur; çalışmanı bir **özellik dalında** yap (`feat/...`, `fix/...`).
- Commit başlıkları kısa ve kapsam-önekli olsun: `fix(plate): ...`, `feat(train): ...`,
  `docs(ftr): ...`, `test(api): ...`.
- PR açıklamasında **ne / neden** ve test sonuçlarını belirt; şablon otomatik gelir.

## 🐞 Hata / öneri

Sorun bildirmek veya özellik önermek için [issue şablonlarını](.github/ISSUE_TEMPLATE/)
kullan. Güvenlik açıkları için lütfen [`SECURITY.md`](SECURITY.md)'yi izle (herkese açık
issue açma).
