"""Sürücü kimlik kilidi — sağ-alt seçim, 5 kare kilit, kilit sonrası reddetme."""

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


def test_locks_after_confirm_frames(cfg):
    lock = DriverLock(cfg)  # confirm_frames = 5 (default)
    v = _vehicle()
    persons = [_person(7, 85, 85), _person(9, 15, 15)]

    last = None
    for i in range(4):  # 1..4. kare → henüz kilit yok
        last = lock.update(vehicle_id=42, vehicle=v, persons=persons, frame_idx=i)
        assert last.locked is False
        assert last.candidate_id == 7
        assert last.streak == i + 1

    last = lock.update(vehicle_id=42, vehicle=v, persons=persons, frame_idx=4)  # 5. kare
    assert last.locked is True
    assert last.newly_locked is True
    assert last.driver_id == 7
    assert lock.driver_of(42) == 7


def test_other_person_cannot_take_over_after_lock(cfg):
    lock = DriverLock(cfg)
    v = _vehicle()
    driver = [_person(7, 85, 85)]
    for i in range(5):  # araç 42'ye sürücü 7 kilitlenir
        lock.update(vehicle_id=42, vehicle=v, persons=driver, frame_idx=i)
    assert lock.is_locked(42)

    # Şimdi sağ-altta DAHA uygun başka biri (id=99) gelse bile sürücü değişmez
    intruder = [_person(99, 95, 95), _person(7, 50, 50)]
    a = lock.update(vehicle_id=42, vehicle=v, persons=intruder, frame_idx=5)
    assert a.locked is True
    assert a.driver_id == 7  # kilitli sürücü korunur
    assert lock.driver_of(42) == 7


def test_candidate_streak_resets_on_change(cfg):
    lock = DriverLock(cfg)
    v = _vehicle()
    # 3 kare aday 7, sonra aday 8 olursa streak sıfırlanır → kilit gecikir
    for i in range(3):
        lock.update(vehicle_id=1, vehicle=v, persons=[_person(7, 85, 85)], frame_idx=i)
    a = lock.update(vehicle_id=1, vehicle=v, persons=[_person(8, 85, 85)], frame_idx=3)
    assert a.locked is False
    assert a.candidate_id == 8
    assert a.streak == 1


def test_empty_cabin_no_lock(cfg):
    lock = DriverLock(cfg)
    v = _vehicle()
    a = lock.update(vehicle_id=5, vehicle=v, persons=[], frame_idx=0)
    assert a.driver_id is None and a.locked is False


def test_person_outside_vehicle_ignored(cfg):
    lock = DriverLock(cfg)
    v = _vehicle()
    outside = [_person(3, 500, 500)]  # araç kutusunun tamamen dışında
    assert lock.persons_in_vehicle(v, outside) == []
    a = lock.update(vehicle_id=6, vehicle=v, persons=outside, frame_idx=0)
    assert a.driver_id is None


def test_prune_forgets_stale_vehicle(cfg):
    lock = DriverLock(cfg)
    v = _vehicle()
    for i in range(5):
        lock.update(vehicle_id=11, vehicle=v, persons=[_person(7, 85, 85)], frame_idx=i)
    assert lock.is_locked(11)
    lock.prune(frame_idx=4 + lock.max_age + 1)  # max_age aşıldı
    assert not lock.is_locked(11)
