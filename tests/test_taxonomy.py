"""Sınıf adı taksonomisi — model-uzayı → kanonik RoadGuard adı eşlemesi."""

from __future__ import annotations

from roadguard.taxonomy import canonical


def test_coco_cell_phone_maps_to_phone():
    assert canonical("cell phone") == "phone"


def test_cigarette_maps_to_smoking():
    assert canonical("cigarette") == "smoking"


def test_case_insensitive():
    assert canonical("Cell Phone") == "phone"


def test_unknown_passthrough():
    assert canonical("car") == "car"
    assert canonical("speed_limit_50") == "speed_limit_50"
