"""dewarp iç fonksiyon kapsamı: _order_corners köşe sıralaması, _find_quad minAreaRect
fallback dalı (4-köşe yok ama büyük kontur var). Saf cv2/numpy, model GEREKTİRMEZ.
"""

from __future__ import annotations

import cv2
import numpy as np

from aura.plate.dewarp import _find_quad, _order_corners


# --- _order_corners ------------------------------------------------------- #
def test_order_corners_sorts_tl_tr_br_bl():
    # Karışık sıralı dört köşe → (TL, TR, BR, BL) düzenine oturmalı.
    pts = np.array([[10, 100], [100, 10], [100, 100], [10, 10]], dtype="float32")
    rect = _order_corners(pts)
    tl, tr, br, bl = rect
    assert tuple(tl) == (10.0, 10.0)  # toplam en küçük
    assert tuple(br) == (100.0, 100.0)  # toplam en büyük
    assert tuple(tr) == (100.0, 10.0)  # x-y en küçük
    assert tuple(bl) == (10.0, 100.0)  # x-y en büyük


def test_order_corners_returns_four_float_points():
    pts = np.array([[0, 0], [50, 2], [48, 30], [2, 28]], dtype="float32")
    rect = _order_corners(pts)
    assert rect.shape == (4, 2)
    assert rect.dtype == np.dtype("float32")


# --- _find_quad ----------------------------------------------------------- #
def test_find_quad_none_on_flat_image():
    # Kenar/kontur olmayan düz görüntü → None (kimlik korunur).
    gray = np.full((60, 180), 80, np.uint8)
    assert _find_quad(gray) is None


def test_find_quad_none_on_zero_area():
    gray = np.zeros((0, 0), np.uint8)
    assert _find_quad(gray) is None


def test_find_quad_detects_rectangle():
    # Büyük dolu dikdörtgen → 4-köşe dörtgen bulunur (alanın >%20'si).
    gray = np.zeros((100, 240), np.uint8)
    cv2.rectangle(gray, (20, 15), (210, 80), 255, -1)
    quad = _find_quad(gray)
    assert quad is not None
    assert quad.shape == (4, 2)


def test_find_quad_minarearect_fallback_on_non_quad_blob():
    # 4-köşe approx VERMEYEN ama yeterince büyük (alanın >%20) bir kontur (daire benzeri)
    # → minAreaRect fallback dalı çalışır, 4 köşe döner (None değil).
    gray = np.zeros((120, 200), np.uint8)
    cv2.circle(gray, (100, 60), 55, 255, -1)  # büyük daire: approxPolyDP 4 vermez
    quad = _find_quad(gray)
    assert quad is not None  # minAreaRect fallback ile köşe üretildi
    assert quad.shape == (4, 2)


def test_find_quad_none_when_blob_too_small():
    # Küçük kontur (alanın %20 altında) → ne 4-köşe ne fallback → None.
    gray = np.zeros((120, 200), np.uint8)
    cv2.circle(gray, (100, 60), 8, 255, -1)  # küçük benek
    assert _find_quad(gray) is None
