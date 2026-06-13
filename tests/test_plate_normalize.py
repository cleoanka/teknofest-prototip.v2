"""TR plaka normalizasyonu + format-öncelikli oy havuzu (model gerektirmez).

Senaryolar gerçek baseline koşusundan alınmıştır (12 Haz 2026, video_1):
EasyOCR '34TC8532' plakasını kareler arasında '041C8532' (3→0, T→1) ve '8532'
(kesik) olarak da okudu; eski 7'lik buffer konsensüsü hiç oluşturamadı.
"""

from __future__ import annotations

from aura.plate.normalize import PlateVotePool, normalize_tr


# --- normalize_tr --------------------------------------------------------- #
def test_valid_plate_passes_through():
    assert normalize_tr("34TC8532") == ("34TC8532", 0)


def test_lowercase_and_punctuation_stripped():
    assert normalize_tr("34 tc 8532") == ("34TC8532", 0)


def test_single_fix_in_letter_block():
    # '1' harf bloğunda → 'I' (1 ikame)
    cand, fixes = normalize_tr("041C8532")
    assert cand == "04IC8532" and fixes == 1


def test_letter_in_province_block_fixed():
    # 'O4' il kodunda O→0
    cand, fixes = normalize_tr("O4TC8532")
    assert cand == "04TC8532" and fixes == 1


def test_unfixable_returns_none():
    assert normalize_tr("INVALID")[0] is None
    assert normalize_tr("8532")[0] is None  # kesik okuma aday OLUŞTURMAZ


def test_invalid_province_rejected():
    assert normalize_tr("99ABC123")[0] is None  # TR il kodu 01-81


# --- PlateVotePool --------------------------------------------------------- #
def _pool() -> PlateVotePool:
    return PlateVotePool(min_weight=2.0, margin_weight=1.5, ratio=0.6)


def test_pool_confirms_dominant_raw_valid_read():
    p = _pool()
    # Gerçek dağılım taklidi: ham-geçerli doğru okuma + düzeltilebilir varyant + kesik
    for _ in range(6):
        p.add("34TC8532")
    for _ in range(9):
        p.add("041C8532")  # 1-ikameli varyant — karara katılamaz
    for _ in range(7):
        p.add("8532")  # kesik — karara katılamaz
    p.add("04TC8532")  # ham-geçerli ama azınlık
    value, conf = p.consensus()
    assert value == "34TC8532"
    assert conf > 0.6


def test_pool_does_not_confirm_competing_valid_reads():
    p = _pool()
    for _ in range(4):
        p.add("34ABC123")
    for _ in range(3):
        p.add("06XY999")
    value, _ = p.consensus()
    assert value is None  # margin_votes=2 sağlanmıyor (4-3=1)


def test_pool_partial_reports_strongest_candidate():
    p = _pool()
    p.add("041C8532")
    p.add("041C8532")
    p.add("8532")
    assert p.consensus()[0] is None
    assert p.best_partial() == "04IC8532"  # düzeltilmiş aday kanıt izi olarak görünür


def test_pool_votes_never_reset():
    p = _pool()
    p.add("34TC8532", conf=0.9)
    assert p.consensus()[0] is None  # min_weight=2.0 henüz yok
    p.add("34TC8532", conf=0.9)
    p.add("34TC8532", conf=0.9)
    assert p.consensus()[0] == "34TC8532"  # birikim korunur, ağırlık 2.7 ile tamamlar


def test_pool_confidence_weighting_beats_systematic_misread():
    # Uzaktan sistematik 3→0 hatası: '04TC8532' ÇOK ama düşük güvenli;
    # yakın/net '34TC8532' az ama yüksek güvenli → doğru plaka kazanmalı.
    p = _pool()
    for _ in range(6):
        p.add("04TC8532", conf=0.35)  # toplam 2.1
    for _ in range(4):
        p.add("34TC8532", conf=0.95)  # toplam 3.8
    value, _ = p.consensus()
    assert value == "34TC8532"


def test_pool_size_weight_beats_far_format_valid_misread():
    # Gerçek video_1 kilidi (12 Haz akşamı): T→I misread'i '34IC8532' formatça
    # GEÇERLİ bir rakip aday üretir ve uzak karelerde sayıca üstünlük kurar
    # (42 vs 30) → ratio 0.38'e düşer, konsensüs HİÇ oluşmaz. Kaynak-kalitesi
    # ağırlığı (LP kırpık yüksekliği) uzak okumaları kısar: yakın/net okumalar
    # kararı dürüstçe verir.
    p = _pool()
    for _ in range(12):
        p.add("34IC8532", conf=0.55, weight=0.2)  # uzak kareler: toplam 1.32
    for _ in range(8):
        p.add("34TC8532", conf=0.85, weight=1.0)  # yakın kareler: toplam 6.8
    value, _ = p.consensus()
    assert value == "34TC8532"


def test_pool_low_weight_alone_cannot_confirm():
    # Yalnızca uzak/kalitesiz kanıt varken karar VERİLMEZ (erken-yanlış-kilit
    # koruması): ağırlık çarpanı min_weight eşiğini dolduramaz.
    p = _pool()
    for _ in range(10):
        p.add("34TC8532", conf=0.9, weight=0.15)  # toplam 1.35 < min_weight 2.0
    assert p.consensus()[0] is None


def test_pool_weight_default_is_backwards_compatible():
    p = _pool()
    for _ in range(3):
        p.add("34TC8532", conf=0.9)  # weight verilmedi → 1.0 (eski davranış)
    assert p.consensus()[0] == "34TC8532"


# --- Pozisyon-hizalı karakter füzyonu (CONFIRMED kararı, pozisyon-margin) ---
def test_position_fusion_confirms_split_correct_plate():
    # Gerçek video_1 dersi: OCR doğru plakayı varyantlara böler (34TC8532/04TC8532/
    # 34IC8532). Hiçbiri tek başına ayrı-aday onayını geçemez (ratio düşük), AMA her
    # pozisyonun çoğunluğu doğru → füzyon 34TC8532'yi onaylar.
    p = _pool()
    for _ in range(5):
        p.add("34TC8532", conf=0.7)  # doğru
    for _ in range(4):
        p.add("04TC8532", conf=0.5)  # 3→0 misread
    for _ in range(3):
        p.add("34IC8532", conf=0.5)  # T→I misread
    # pos0: 3 (34TC+34IC) net > 0 (04TC); pos2: T (34TC+04TC) net > I (34IC)
    assert p.consensus()[0] == "34TC8532"


def test_position_fusion_pending_on_ambiguous_leading_digit():
    # KRİTİK: ilk karakter 0↔3 neredeyse eşit okunuyorsa (kullanıcının 'kronik'
    # dediği sorun) o pozisyon belirsizdir → yanlış '04' ASLA onaylanmaz, pending.
    p = _pool()
    for _ in range(5):
        p.add("34TC8532", conf=0.6)
    for _ in range(5):
        p.add("04TC8532", conf=0.6)  # 0 ve 3 eşit ağırlık
    assert p.consensus()[0] is None  # pos0 belirsiz → dürüst pending
    assert p.best_partial() is not None  # ama tahmin (kanıt izi) verilir


def test_position_fusion_pending_on_far_letter_ambiguity():
    # Gerçek video_3 dersi: uzaktan I↔T ayrılamıyor (fark < char_margin) → pending.
    p = _pool()
    for _ in range(3):
        p.add("24IC8532", conf=0.8)
    for _ in range(2):
        p.add("24TC8532", conf=0.6)  # pos2 I vs T yakın
    assert p.consensus()[0] is None  # pos2 belirsiz → pending (yanlış plaka onaylanmaz)


def test_dominant_correct_read_still_confirms():
    # Doğru okuma NET baskınsa (ayrı-aday margin + ratio) normal onay sürer.
    p = _pool()
    for _ in range(8):
        p.add("34TC8532", conf=0.85)
    for _ in range(3):
        p.add("34IC8532", conf=0.6)  # azınlık misread
    assert p.consensus()[0] == "34TC8532"


def test_competing_distinct_plates_never_confirm():
    # İki GERÇEKTEN farklı plaka birleştirilmez, onaylanmaz (belirsizlik korunur).
    p = _pool()
    for _ in range(4):
        p.add("34ABC123")
    for _ in range(3):
        p.add("06XY999")
    assert p.consensus()[0] is None


def test_char_fuse_partial_picks_dominant_per_position():
    # best_partial füzyonu: pozisyon başına en baskın karakter (eşiksiz kanıt izi).
    p = _pool()
    for _ in range(5):
        p.add("34IC8532", conf=0.8)
    for _ in range(2):
        p.add("34IC0532", conf=0.7)  # pos5 azınlık (0)
    # pos5'te 8 baskın → birleşik partial 34IC8532
    assert p.best_partial() == "34IC8532"


def test_char_consensus_off_falls_back_to_weights():
    p = PlateVotePool(min_weight=2.0, margin_weight=1.5, ratio=0.6, char_consensus=False)
    for _ in range(3):
        p.add("34IC8532", conf=0.8)
    for _ in range(2):
        p.add("34IC0532", conf=0.7)
    # füzyon kapalı → en ağır tek aday (ağırlık) döner, birleştirme yok
    assert p.best_partial() == "34IC8532"
