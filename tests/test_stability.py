"""16/8 kararlılık state machine (plan.md §6.3, §15): 7/16→ret, 8/16→kabul, flicker koruması."""

from __future__ import annotations

from roadguard.stability import StabilityTracker


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


# --- non-bool değer: _default passthrough (değer döner, False değil) -----------
def test_non_bool_value_passthrough_before_commit():
    # bool olmayan değer için _default değeri AYNEN döndürür (False değil).
    class _Cfg:
        def get(self, k, d=None):
            return {"stability.window": 16, "stability.min_consistent": 8}.get(k, d)

    st = StabilityTracker(_Cfg())
    # tek gözlem, henüz commit yok → _default(value) = value (string passthrough)
    assert st.update("speed", "fast") == "fast"


def test_non_bool_value_commits_after_consensus():
    class _Cfg:
        def get(self, k, d=None):
            return {"stability.window": 16, "stability.min_consistent": 8}.get(k, d)

    st = StabilityTracker(_Cfg())
    last = None
    for _ in range(8):
        last = st.update("speed", "fast")
    assert last == "fast" and st.committed("speed") == "fast"


# --- pencere yeniden-boyutlama (maxlen değişimi içeriği koruyarak deque'i kurar) ---
def test_window_resize_rebuilds_deque_preserving_content():
    class _Cfg:
        def __init__(self, w, m):
            self._d = {"stability.window": w, "stability.min_consistent": m}

        def get(self, k, d=None):
            return self._d.get(k, d)

    st = StabilityTracker(_Cfg(16, 8))
    for _ in range(4):
        st.update("k", True)
    assert st.support("k", True) == 4
    # pencereyi küçült: mevcut içerik korunarak yeni maxlen'li deque kurulmalı
    st.window = 2
    st.update("k", True)  # resize tetiklenir; eski içerikten son 2 öğe + yeni = maxlen 2
    w = st._windows["k"]
    assert w.maxlen == 2 and len(w) == 2


# --- reset(key) vs reset(None) ------------------------------------------------
def test_reset_single_key(cfg):
    st = StabilityTracker(cfg)
    for _ in range(8):
        st.update("a", True)
        st.update("b", True)
    st.reset("a")
    assert st.committed("a") is None and st.support("a", True) == 0
    assert st.committed("b") is True  # b etkilenmedi


def test_reset_all(cfg):
    st = StabilityTracker(cfg)
    for _ in range(8):
        st.update("a", True)
        st.update("b", True)
    st.reset()
    assert st.committed("a") is None and st.committed("b") is None
    assert st._windows == {}


def test_support_unknown_key_zero(cfg):
    st = StabilityTracker(cfg)
    assert st.support("nope", True) == 0
