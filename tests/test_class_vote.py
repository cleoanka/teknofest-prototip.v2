"""Track başına sınıf oylaması (model gerektirmez).

Senaryolar gerçek video ölçümlerinden alınmıştır (12 Haz akşamı):
- video_1/2: araç İLK karede 0.79-0.84 güvenle 'truck', sonra kalıcı 'car'
- video_3: yakın plandaki otomobil tek tek karelerde 'truck'a dönüyor
"""

from __future__ import annotations

from roadguard.stability.class_vote import TrackClassVoter


def _voter(cfg, decay: float | None = None) -> TrackClassVoter:
    if decay is not None:
        cfg.data.setdefault("tracking", {}).setdefault("class_vote", {})["decay"] = decay
    return TrackClassVoter(cfg)


def test_single_class_passthrough(cfg):
    v = _voter(cfg)
    assert v.update(1, "car", 0.9) == "car"
    assert v.update(1, "car", 0.8) == "car"
    assert v.stable_class(1) == "car"


def test_single_frame_flip_is_outvoted(cfg):
    # video_3 senaryosu: yerleşik 'car' track'i tek karede yüksek güvenli 'truck'
    # görse bile çoğunluk sınıfı değişmez.
    v = _voter(cfg)
    for _ in range(20):
        v.update(3, "car", 0.6)
    assert v.update(3, "truck", 0.9) == "car"
    assert v.update(3, "car", 0.6) == "car"


def test_wrong_first_frame_corrects_quickly(cfg):
    # video_1/2 senaryosu: ilk kare uzak/bulanık → 'truck' 0.84; sonraki kareler 'car'.
    v = _voter(cfg)
    assert v.update(1, "truck", 0.84) == "truck"  # ilk karede tek kanıt o
    v.update(1, "car", 0.55)
    stable = v.update(1, "car", 0.60)
    assert stable == "car"  # 1.15 > 0.84 — iki karede düzelir


def test_untracked_detection_not_voted(cfg):
    v = _voter(cfg)
    assert v.update(-1, "truck", 0.9) == "truck"  # kimliksiz kutuya geçmiş bağlanmaz
    assert v.update(None, "bus", 0.9) == "bus"
    assert v.stable_class(-1) is None


def test_disabled_returns_raw(cfg):
    cfg.data.setdefault("tracking", {})["class_vote"] = {"enabled": False}
    v = TrackClassVoter(cfg)
    v.update(1, "car", 0.9)
    assert v.update(1, "truck", 0.1) == "truck"  # kapalıyken ham sınıf aynen döner


def test_decay_allows_late_correction(cfg):
    # Saf kümülatifte erken dönemde birikmiş yanlış oy sonsuza dek baskın kalabilir;
    # decay eski kanıtı söndürür → kalıcı yeni kanıt eninde sonunda kazanır.
    v = _voter(cfg, decay=0.9)
    for _ in range(30):
        v.update(7, "truck", 0.9)
    last = "truck"
    for _ in range(60):
        last = v.update(7, "car", 0.9)
    assert last == "car"


def test_area_weighted_near_overrides_far(cfg):
    # KİLİT (gerçek video_2 dersi): araç uzaktayken ONLARCA kare yüksek-güvenli
    # 'truck' tespit ediliyor; yakınlaşınca 'car'. Alan ağırlığı sayesinde az
    # sayıda YAKIN (büyük bbox) 'car' karesi, çok sayıda UZAK (küçük bbox) 'truck'
    # karesini devralır.
    v = _voter(cfg)
    last = "?"
    for _ in range(30):
        last = v.update(7, "truck", 0.84, area_norm=0.004)  # uzak, küçük bbox
    assert last == "truck"  # henüz yalnız uzak kanıt var
    for _ in range(10):
        last = v.update(7, "car", 0.70, area_norm=0.05)  # yakın, büyük bbox
    assert last == "car"  # az sayıda yakın kanıt, çok sayıda uzak kanıtı devraldı


def test_area_floor_keeps_far_vote_alive(cfg):
    # Çok uzak araç (area_norm ~0) bile oy üretmeli (area_floor) — tamamen yok sayılmaz.
    v = _voter(cfg)
    last = "?"
    for _ in range(5):
        last = v.update(8, "car", 0.5, area_norm=0.0)
    assert last == "car"


def test_prune_drops_dead_tracks(cfg):
    v = _voter(cfg)
    v.update(1, "car", 0.9)
    v.update(2, "bus", 0.9)
    v.prune({2})
    assert v.stable_class(1) is None and v.stable_class(2) == "bus"


def test_alphabetic_tiebreak_is_deterministic(cfg):
    # Eşit ağırlıkta (aynı conf, area yok) deterministik kazanan: max((w, ad)) →
    # ağırlık eşitse alfabetik BÜYÜK ad kazanır ('car' > 'bus'). Mevcut davranış —
    # önemli olan tutarlı/deterministik olması (giriş sırasından bağımsız).
    v1 = _voter(cfg)
    v1.update(1, "car", 0.5)
    assert v1.update(1, "bus", 0.5) == "car"
    # ters sırada da aynı kazanan → sıradan bağımsız determinizm
    v2 = _voter(cfg)
    v2.update(1, "bus", 0.5)
    assert v2.update(1, "car", 0.5) == "car"


def test_empty_cls_passthrough(cfg):
    # Boş sınıf adı oylanmaz, aynen döner.
    v = _voter(cfg)
    assert v.update(1, "", 0.9) == ""
    assert v.stable_class(1) is None


def test_zero_conf_uses_floor_weight(cfg):
    # conf<=0 → 1e-3 taban ağırlık (oy yine sayılır, sıfırlanmaz).
    v = _voter(cfg)
    assert v.update(1, "car", 0.0) == "car"
    assert v.stable_class(1) == "car"
    # taban ağırlık birikir: tek oydan sonra çoğunluk hâlâ 'car'
    assert v.update(1, "car", -5.0) == "car"


def test_decay_multiplies_existing_before_adding(cfg):
    # decay<1 iken GELEN oy eklenmeden ÖNCE mevcut oylar çarpılır; yeni oy tam ağırlıkta.
    v = _voter(cfg, decay=0.5)
    v.update(1, "truck", 1.0)  # truck=1.0
    # 'car' eklenirken truck 0.5'e söner, car=1.0 → tek karede car kazanır
    assert v.update(1, "car", 1.0) == "car"


def test_stable_class_unknown_returns_none(cfg):
    v = _voter(cfg)
    assert v.stable_class(999) is None
