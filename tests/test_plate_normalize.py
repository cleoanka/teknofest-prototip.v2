"""TR plaka normalizasyonu + format-öncelikli oy havuzu (model gerektirmez).

Senaryolar gerçek baseline koşusundan alınmıştır (12 Haz 2026, video_1):
EasyOCR '34TC8532' plakasını kareler arasında '041C8532' (3→0, T→1) ve '8532'
(kesik) olarak da okudu; eski 7'lik buffer konsensüsü hiç oluşturamadı.
"""

from __future__ import annotations

from roadguard.plate.normalize import PlateVotePool, normalize_tr


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


def test_pool_full_high_quality_displaces_low_quality():
    """Codex bulgusu: havuz (max_reads) dolduğunda, yakın/net yüksek-ağırlık okuma,
    erken biriken uzak/bulanık düşük-ağırlık YANLIŞ okumayı değiştirip konsensüsü
    düzeltebilmeli (eski sürüm dolu havuzda yeni okumayı tamamen düşürüyordu)."""
    p = PlateVotePool(min_weight=2.0, margin_weight=1.5, ratio=0.6, max_reads=4)
    for _ in range(4):
        p.add("06AA1111", conf=0.2, weight=0.2)  # havuzu düşük-ağırlıkla doldur (yanlış)
    for _ in range(4):
        p.add("34TC8532", conf=0.95, weight=0.95)  # yakın/net doğru → düşükleri ezer
    value, _ = p.consensus()
    assert value == "34TC8532"


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


# --- v2.3 dürüstlük zırhları: ayrı-aday VETO + uzak-okuma zemin koşulu --------
def test_separate_candidate_vetoed_when_position_contested():
    # yolo26l video_1 dersi: '04TC8532' bütün-string baskın (margin+ratio GEÇER) AMA
    # pos0 0↔3 çekişmeli → ayrı-aday VETO eder, char füzyonu da belirsiz → dürüst pending.
    # (Eski sürüm bu durumda YANLIŞ '04TC8532' onaylıyordu.)
    p = _pool()
    for _ in range(4):
        p.add("04TC8532", conf=0.6)  # 2.4 — bütün-string lideri
    for _ in range(2):
        p.add("34TC8532", conf=0.45)  # 0.9
    for _ in range(2):
        p.add("34IC8532", conf=0.35)  # 0.7  → pos0 '3' toplam 1.6
    # ayrı-aday: w_top=2.4>=2.0, margin 2.4-0.9=1.5>=1.5, ratio 2.4/4.0=0.6>=0.6 → GEÇER
    # ama pos0: '0'=2.4 vs '3'=1.6, fark 0.8 < char_margin 1.5 → VETO → pending
    assert p.consensus()[0] is None
    assert p.best_partial() is not None  # kanıt izi yine verilir


def test_far_only_reads_stay_pending_without_clear_view():
    # video_3 dersi: UZAK plaka tutarlı okunur ('24IC8532') ama hiçbir okuma NET değil
    # (peak < confirm_peak_weight) → birikim min_weight'i geçse de dürüst pending.
    p = _pool()  # confirm_peak_weight=0.30 (varsayılan)
    for _ in range(22):
        p.add("24IC8532", conf=0.55, weight=0.2)  # eff 0.11 each → toplam ~2.4, peak 0.11
    assert p.consensus()[0] is None  # zemin koşulu sağlanmadı → pending
    assert p.best_partial() == "24IC8532"  # kanıt izi görünür


def test_peak_floor_disabled_allows_far_only_confirm():
    # confirm_peak_weight=0 → zemin koşulu kapalı (ayarlanabilirlik); uzak-birikim onaylar.
    p = PlateVotePool(min_weight=2.0, margin_weight=1.5, ratio=0.6, confirm_peak_weight=0.0)
    for _ in range(22):
        p.add("24IC8532", conf=0.55, weight=0.2)
    assert p.consensus()[0] == "24IC8532"


def test_clear_close_read_still_confirms_with_peak_floor():
    # En az bir NET/yakın okuma (peak >= floor) varsa onay normal sürer.
    p = _pool()
    for _ in range(4):
        p.add("34TC8532", conf=0.9, weight=1.0)  # peak 0.9 >= 0.30
    assert p.consensus()[0] == "34TC8532"


# --- SORUN 1: ONAY-sıkı pozisyon eşiği (confirm_min_char_margin) ---------------
# Gerçek video ölçümü (17 Haz, stok yolo26l): YANLIŞ ilk-harf '0' pos0-margin'i ~1.55,
# DOĞRU '3' margin'i ~1.52 — ikisi de char_margin=1.5'i geçip YANLIŞ onaya gidiyordu.
# Yeni 'confirm_min_char_margin' (vars. 2.0) ile belirsiz ilk-karakter dürüst PENDING;
# NET plaka (video_2, yüksek margin) onaylanmaya devam eder. (Yanlış-onay sıfır > çok-onay.)
def _strict_pool() -> PlateVotePool:
    # default.yaml ile aynı: confirm_min_char_margin=2.0
    return PlateVotePool(min_weight=2.0, margin_weight=1.5, ratio=0.6, confirm_min_char_margin=2.0)


def test_confirm_min_char_margin_blocks_wrong_first_digit_above_old_threshold():
    # KRİTİK regresyon kapanı: pos0 '0' margini ~1.55 (eski 1.5 eşiğinin ÜSTÜNDE) →
    # eski kod '04TC8532'yi YANLIŞ onaylıyordu. Sıkı eşik (2.0) → dürüst PENDING.
    p = _strict_pool()
    for _ in range(5):
        p.add("04TC8532", conf=0.62)  # 3.10
    for _ in range(2):
        p.add("34TC8532", conf=0.40)  # 0.80
    for _ in range(2):
        p.add("34IC8532", conf=0.375)  # 0.75 → pos0 '3' toplam 1.55, '0' 3.10, margin 1.55
    assert p.consensus()[0] is None  # 1.55 < confirm_min 2.0 → yanlış onay ENGELLENDİ
    assert p.best_partial() is not None  # kanıt izi yine raporlanır (partial korunur)


def test_confirm_min_char_margin_pends_correct_but_uncertain_first_digit():
    # Doğru '3' margini ~1.52 da (eşik altı) PENDING olur — KABUL: belirsizken
    # yanlış-onay engellemek > şüpheli-onay (kullanıcı sert direktifi).
    p = _strict_pool()
    for _ in range(4):
        p.add("34TC8532", conf=0.38)  # 1.52 — doğru ama belirsiz (rakipsiz değil)
    p.add("04TC8532", conf=0.0001)  # zayıf '0' rakibi (pos0 margin ~1.52)
    assert p.consensus()[0] is None  # belirsiz → pending


def test_confirm_min_char_margin_clear_plate_still_confirms():
    # Net/yüksek-margin plaka (video_2: 34TC8532 baskın) sıkı eşikle de ONAYLANIR.
    p = _strict_pool()
    for _ in range(8):
        p.add("34TC8532", conf=0.9)  # pos0 margin çok yüksek (>2.0)
    p.add("34IC8532", conf=0.5)  # azınlık T→I misread
    assert p.consensus()[0] == "34TC8532"


def test_confirm_min_char_margin_defaults_to_char_margin_when_none():
    # confirm_min_char_margin=None → char_margin'e düşer (geriye dönük uyum):
    # margin 1.55 eski davranışta onaylanır (yeni knob verilmeden eski test korunur).
    p = PlateVotePool(min_weight=2.0, margin_weight=1.5, ratio=0.6)  # confirm_min=None
    assert p.confirm_char_margin == p.char_margin == 1.5
    for _ in range(5):
        p.add("04TC8532", conf=0.62)
    for _ in range(2):
        p.add("34TC8532", conf=0.40)
    for _ in range(2):
        p.add("34IC8532", conf=0.375)
    # eski (gevşek) davranış: 1.55 >= 1.5 → onaylar (knob None iken regresyon yok)
    assert p.consensus()[0] == "04TC8532"


def test_confirm_min_char_margin_never_below_char_margin():
    # confirm_min < char_margin verilse bile char_margin'in altına düşmez (güvenlik).
    p = PlateVotePool(char_margin=1.8, confirm_min_char_margin=1.0)
    assert p.confirm_char_margin == 1.8


# --- PERF: normalize_tr önbelleği (add()'te bir kez; davranış-koruyan) ---------
def test_norm_cache_index_aligned_with_raw_reads():
    # _norm her okuma için cache'lenir ve raw_reads ile İNDEKS-HİZALI olmalı:
    # _norm[i] == normalize_tr(raw_reads[i][0]). (O(N²) yeniden-normalizasyonu önler.)
    p = _pool()
    samples = ["34TC8532", "041C8532", "8532", "INVALID", "O4TC8532"]
    for s in samples:
        p.add(s)
    assert len(p._norm) == len(p.raw_reads)
    for (raw, _), cached in zip(p.raw_reads, p._norm, strict=True):
        assert cached == normalize_tr(raw)


def test_norm_cache_does_not_change_consensus_result():
    # Önbellek davranışı DEĞİŞTİRMEZ: cache'li sonuç, her seferinde yeniden normalize
    # edilenle birebir aynı kazananı/güveni üretir (regresyon kapanı).
    p = _pool()
    for _ in range(6):
        p.add("34TC8532", conf=0.85)
    for _ in range(3):
        p.add("041C8532", conf=0.5)  # düzeltilebilir varyant (karara katılmaz)
    value, conf = p.consensus()
    assert value == "34TC8532"
    # ikinci kez çağırınca (cache yeniden kullanılır) aynı sonuç
    assert p.consensus() == (value, conf)
    assert p.best_partial() == "34TC8532"
