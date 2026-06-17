"""Sürücü/Yolcu atama — pozisyonel sürücü (sağ-en-alt, her kare) + yolcu kilidi."""

from __future__ import annotations

from aura.detection.detector import Person
from aura.identity.driver_lock import DriverLock
from aura.schema import BBox


def _vehicle() -> BBox:
    return BBox(x1=0, y1=0, x2=100, y2=100, conf=0.9, cls="car")


def _person(track_id: int, cx: float, cy: float, half: float = 6.0) -> Person:
    return Person(
        bbox=BBox(x1=cx - half, y1=cy - half, x2=cx + half, y2=cy + half, conf=0.9, cls="person"),
        track_id=track_id,
    )


# --- sürücü seçimi: dikey öncelikli sağ-en-alt -------------------------------- #


def test_selects_bottom_right(cfg):
    lock = DriverLock(cfg)
    v = _vehicle()
    persons = [
        _person(1, 20, 20),  # sol-üst (yolcu)
        _person(2, 80, 80),  # sağ-alt (sürücü)
        _person(3, 20, 80),  # sol-alt
    ]
    cand = lock.select_candidate(v, persons)
    assert cand is not None and cand.track_id == 2


def test_selects_bottom_over_right(cfg):
    """Dikey öncelik: daha AŞAĞIDAKİ kişi, daha sağdaki ama yukarıdakine yeğlenir."""
    lock = DriverLock(cfg)
    v = _vehicle()
    lower_left = _person(1, 40, 95)  # ny=0.95 (en alt), nx=0.40
    higher_right = _person(2, 95, 60)  # ny=0.60, nx=0.95 (tam sağ ama yukarıda)
    cand = lock.select_candidate(v, [lower_left, higher_right])
    assert cand is not None and cand.track_id == 1


# --- sürücü POZİSYONEL (kilitlenmez) ----------------------------------------- #


def test_driver_is_positional_each_frame(cfg):
    """Sürücü her kare sağ-en-alta göre yeniden seçilir; kimliğe kilitlenmez."""
    lock = DriverLock(cfg)
    v = _vehicle()
    # Kare 0: 7 sağ-altta → sürücü, 8 yolcu
    a = lock.update(1, v, [_person(7, 85, 85), _person(8, 20, 20)], 0)
    assert a.driver_id == 7 and a.passenger_ids == [8]
    # Kare 1: 8 sağ-alta geçti (7 yukarı çıktı) → sürücü ANINDA 8 (kilit gecikmesi yok)
    a = lock.update(1, v, [_person(7, 20, 20), _person(8, 85, 85)], 1)
    assert a.driver_id == 8 and a.passenger_ids == [7]


def test_driver_established_after_confirm_frames(cfg):
    """Sürücü confirm_frames boyunca varsa 'kuruldu' (locked + tek-seferlik event)."""
    lock = DriverLock(cfg)  # confirm_frames = 3
    v = _vehicle()
    d = [_person(7, 85, 85)]
    for i in range(2):  # 1..2. kare → henüz kurulmadı
        a = lock.update(1, v, d, i)
        assert a.driver_id == 7 and a.locked is False and a.newly_locked is False
        assert a.streak == i + 1
    a = lock.update(1, v, d, 2)  # 3. kare → kuruldu
    assert a.locked is True and a.newly_locked is True
    assert lock.is_locked(1) and lock.driver_of(1) == 7


# --- yolcu kilidi ------------------------------------------------------------- #


def test_passenger_locks_after_confirm_frames(cfg):
    lock = DriverLock(cfg)  # confirm_frames = 3
    v = _vehicle()
    persons = [_person(7, 85, 85), _person(8, 20, 20)]  # 7 sürücü, 8 yolcu
    a = None
    for i in range(3):
        a = lock.update(1, v, persons, i)
        assert a.driver_id == 7 and a.passenger_ids == [8]
    # 3 kare yolcu kaldı → 8 YOLCU olarak kilitlendi
    assert 8 in lock.passengers_of(1)
    assert a.locked_passenger_ids == [8]


def test_locked_passenger_excluded_from_driver(cfg):
    """Kilitli yolcu, sağ-alta geçse bile sürücü olamaz."""
    lock = DriverLock(cfg)
    v = _vehicle()
    for i in range(3):  # 8'i yolcu olarak kilitle (7 sürücüyken)
        lock.update(1, v, [_person(7, 85, 85), _person(8, 20, 20)], i)
    assert 8 in lock.passengers_of(1)
    # Şimdi 8 sağ-altta, 7 yukarıda olsa bile: 8 kilitli yolcu → sürücü 7 kalır
    a = lock.update(1, v, [_person(7, 20, 20), _person(8, 85, 85)], 3)
    assert a.driver_id == 7
    assert 8 in a.passenger_ids


# --- kenar durumlar ----------------------------------------------------------- #


def test_empty_cabin_no_driver(cfg):
    lock = DriverLock(cfg)
    v = _vehicle()
    a = lock.update(5, v, [], 0)
    assert a.driver_id is None and a.locked is False and a.passenger_ids == []


def test_person_outside_vehicle_ignored(cfg):
    lock = DriverLock(cfg)
    v = _vehicle()
    outside = [_person(3, 500, 500)]  # araç kutusunun tamamen dışında
    assert lock.persons_in_vehicle(v, outside) == []
    a = lock.update(6, v, outside, 0)
    assert a.driver_id is None


def test_passengers_tagged(cfg):
    """Sürücü dışındaki herkes yolcu olarak işaretlenir (passenger_ids)."""
    lock = DriverLock(cfg)
    v = _vehicle()
    persons = [
        _person(2, 80, 80),  # sürücü = en alt-sağ
        _person(1, 20, 20),  # yolcu (sol-üst)
        _person(3, 20, 80),  # yolcu (sol-alt)
    ]
    a = lock.update(7, v, persons, 0)
    assert a.driver_id == 2
    assert sorted(a.passenger_ids) == [1, 3]


def test_prune_forgets_stale_vehicle(cfg):
    lock = DriverLock(cfg)
    v = _vehicle()
    for i in range(3):
        lock.update(11, v, [_person(7, 85, 85), _person(8, 20, 20)], i)
    assert 8 in lock.passengers_of(11)
    lock.prune(frame_idx=2 + lock.max_age + 1)  # max_age aşıldı
    assert lock.passengers_of(11) == set()
    assert not lock.is_locked(11)


# --- global / dışlamalı eşleştirme (assign_frame) --------------------------- #


def test_two_vehicles_distinct_drivers(cfg):
    """İki ayrı araç + iki ayrı kişi → her biri kendi aracının sürücüsü olur."""
    lock = DriverLock(cfg)
    a = BBox(x1=0, y1=0, x2=100, y2=100, conf=0.9, cls="car")
    b = BBox(x1=200, y1=0, x2=300, y2=100, conf=0.9, cls="car")
    pa = _person(7, 80, 80)  # yalnızca A içinde
    pb = _person(9, 280, 80)  # yalnızca B içinde
    out = None
    for i in range(3):
        out = lock.assign_frame([(1, a), (2, b)], [pa, pb], i)
    assert out[0].driver_id == 7
    assert out[1].driver_id == 9


def test_shared_person_drives_best_score_vehicle(cfg):
    """Örtüşen iki araçta tek kişi → en iyi skorlu (merkeze yakın) aracın sürücüsü."""
    lock = DriverLock(cfg)
    a = BBox(x1=0, y1=0, x2=100, y2=100, conf=0.9, cls="car")  # merkez (50,50)
    b = BBox(x1=50, y1=0, x2=150, y2=100, conf=0.9, cls="car")  # merkez (100,50)
    shared = _person(7, 60, 80)  # A'nın merkezine daha yakın
    out = lock.assign_frame([(1, a), (2, b)], [shared], 0)
    assert out[0].driver_id == 7
    assert out[1].driver_id is None


def test_locked_passenger_excluded_from_other_vehicle(cfg):
    """A'ya kilitli yolcu, B'ye yaklaşsa bile B'nin sürücüsü olamaz (global dışlama)."""
    lock = DriverLock(cfg)
    a = BBox(x1=0, y1=0, x2=100, y2=100, conf=0.9, cls="car")
    b = BBox(x1=50, y1=0, x2=150, y2=100, conf=0.9, cls="car")
    drv = _person(7, 40, 90)  # A'nın sürücüsü (en alt)
    pas = _person(8, 60, 30)  # A içinde, üstte → yolcu
    for i in range(3):  # 8'i A'da yolcu olarak kilitle
        lock.assign_frame([(1, a), (2, b)], [drv, pas], i)
    assert 8 in lock.passengers_of(1)
    # 8 artık B'ye daha yakın bir konumda; yine de A'nın kilitli yolcusu kalır.
    out = lock.assign_frame([(1, a), (2, b)], [_person(7, 40, 90), _person(8, 120, 30)], 3)
    assert 8 in out[0].passenger_ids  # hâlâ A'nın yolcusu
    assert out[1].driver_id != 8  # B 8'i sürücü yapamaz
