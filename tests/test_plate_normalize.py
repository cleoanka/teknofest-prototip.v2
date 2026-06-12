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
