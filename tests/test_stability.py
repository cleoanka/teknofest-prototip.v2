"""16/8 kararlılık state machine (plan.md §6.3, §15): 7/16→ret, 8/16→kabul, flicker koruması."""

from __future__ import annotations

from aura.stability import StabilityTracker


def test_8_consecutive_accepted(cfg):
    st = StabilityTracker(cfg)  # window=16, min_consistent=8
    last = None
    for _ in range(8):
        last = st.update("k", True)
    assert last is True
    assert st.committed("k") is True


def test_7of16_true_rejected(cfg):
    st = StabilityTracker(cfg)
    for v in [True] * 7 + [False] * 9:
        st.update("k", True if v else False)
    assert st.support("k", True) == 7
    assert st.committed("k") is not True  # 7/16 True reddedildi
    assert st.committed("k") is False  # 9/16 False kabul edildi


def test_16of16_accepted(cfg):
    st = StabilityTracker(cfg)
    last = None
    for _ in range(16):
        last = st.update("k", True)
    assert last is True


def test_flicker_preserves_previous(cfg):
    st = StabilityTracker(cfg)
    for _ in range(10):
        st.update("k", True)  # committed True
    last = None
    for _ in range(3):  # kısa flicker (3/16 False) → override yok
        last = st.update("k", False)
    assert last is True


def test_independent_keys(cfg):
    st = StabilityTracker(cfg)
    for _ in range(8):
        st.update("a", True)
    assert st.committed("a") is True
    assert st.update("b", True) is False  # tek gözlem → henüz konsensüs yok


def test_initial_default_false(cfg):
    st = StabilityTracker(cfg)
    assert st.update("k", True) is False  # ilk kare: kanıt yok → False
