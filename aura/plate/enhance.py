"""Plaka kontrast/keskinlik iyileştirme (karanlık otopark) — saf OpenCV/numpy.

WP-A1 dersi: karanlık/açılı otoparkta plaka düşük kontrastlı + hafif bulanık
gelir → EasyOCR karakterleri (özellikle il kodu) tutarlı yanlış okur. dewarp
SONRASI tek noktada uygulanan bu adım: CLAHE (LAB L-kanalı) ile yerel kontrastı
açar, gamma ile karanlık tonları aydınlatır, hafif unsharp ile karakter
kenarlarını netleştirir.

Reader içindeki MEVCUT CLAHE+2x varyantı (`_enhance`, ikinci-şans okuma) ile
ÇAKIŞMAZ: bu fonksiyon dewarp'tan hemen sonra, OCR'a girmeden ÖNCE bir kez
çalışır; ölçek büyütme yapmaz (yalnız ton/kontrast). Parametreler config
`plate.enhance.*` (clahe_clip, gamma). Deterministik, model gerektirmez.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@functools.lru_cache(maxsize=16)
def _gamma_lut(gamma: float):
    """gamma → 256-elemanlı LUT (önbellekli).

    PERF: gamma config'te SABİT olduğundan LUT her enhance çağrısında 256-elemanlı
    listcomp ile yeniden üretilmesin diye gamma anahtarıyla cache'lenir. Davranış
    aynı (aynı gamma → bitsel aynı LUT). lru_cache sınırlı (maxsize) → bellek sabit.
    """
    import numpy as np

    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype="float32")
    return np.clip(lut, 0, 255).astype("uint8")


def enhance_plate(img: np.ndarray | None, cfg) -> np.ndarray | None:
    """Plaka kırpığını CLAHE + gamma + hafif unsharp ile iyileştir.

    Şekil ve dtype (uint8) KORUNUR — yalnız ton/kontrast değişir; ölçek
    büyütme yok (o iş reader'ın 2x varyantına ait). Giriş None/boş ya da
    işlenemezse AYNEN döner (kimlik). Gri (tek kanal) giriş de desteklenir.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return img
    import cv2

    clip = float(cfg.get("plate.enhance.clahe_clip", 2.5))
    gamma = float(cfg.get("plate.enhance.gamma", 1.2))

    src = img
    is_gray = src.ndim == 2
    if is_gray:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    # 1) CLAHE: LAB L-kanalında yerel kontrast (renkleri bozmadan aydınlatır).
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    lab_l, lab_a, lab_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, clip), tileGridSize=(8, 8))
    lab_l = clahe.apply(lab_l)
    out = cv2.cvtColor(cv2.merge((lab_l, lab_a, lab_b)), cv2.COLOR_LAB2BGR)

    # 2) Gamma düzeltmesi: gamma>1 karanlık tonları açar (LUT ile hızlı/deterministik).
    if abs(gamma - 1.0) > 1e-3 and gamma > 0:
        out = cv2.LUT(out, _gamma_lut(gamma))

    # 3) Hafif unsharp mask: orijinal + (orijinal − blur) ile kenarları netleştir.
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
    out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)

    if is_gray:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return out
