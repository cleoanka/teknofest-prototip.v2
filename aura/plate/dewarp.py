"""Plaka perspektif düzeltme (fronto-paralel dewarp) — saf OpenCV/numpy.

Karanlık/açılı otopark dersi (WP-A1): plaka kameraya AÇIYLA görünür (yandan
yaklaşan/park eden araç) → karakterler trapez biçiminde sıkışır ve EasyOCR il
kodunu tutarlı yanlış okur (ör. 3→0/2). Çözüm OCR ÖNCESİ plaka kırpığını
fronto-paralel düzleştirmektir: en büyük dörtgen konturu bulup TR plaka en-boy
oranına (520/110 ≈ 4.73) warp ederiz.

Tasarım ilkeleri (K-004):
- Deterministik, model GEREKTİRMEZ (yalnız cv2 + numpy).
- Dörtgen kontur GÜVENLE bulunamazsa görüntüyü AYNEN döndür (kimlik) —
  belirsizlikte asla "uydurma" bir warp uygulanmaz; bozuk warp OCR'ı doğru
  kırpıktan daha kötü yapardı.
- Video-özel sabit yok; eşikler giriş boyutundan türetilir (oransal).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# TR Tip-1 plaka standardı: 520 mm × 110 mm → hedef en-boy oranı.
_TARGET_W = 520
_TARGET_H = 110
_TARGET_RATIO = _TARGET_W / _TARGET_H  # ≈ 4.727


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Dört köşeyi (TL, TR, BR, BL) saat yönünde sırala.

    Köşe sıralaması warp için zorunludur: toplam (x+y) en küçük = sol-üst, en
    büyük = sağ-alt; fark (x−y) en küçük = sağ-üst, en büyük = sol-alt.
    """
    import numpy as np

    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]  # TR
    rect[3] = pts[np.argmax(d)]  # BL
    return rect


def _find_quad(gray: np.ndarray) -> np.ndarray | None:
    """Gri görüntüde plaka çerçevesi olabilecek en büyük 4-köşe dörtgeni bul.

    Sıra: blur → adaptif eşik + Canny birleşimi → kontur → approxPolyDP. 4 köşe
    bulunan en büyük (yeterince geniş alanlı) kontur döner; yoksa minAreaRect'in
    4 köşesine düşülür. Hiçbir aday görüntü alanının makul payını kaplamıyorsa
    None (kimlik kalır).
    """
    import cv2
    import numpy as np

    h, w = gray.shape[:2]
    area = float(h * w)
    if area <= 0:
        return None
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptif eşik + Canny: karanlık/düşük-kontrast plakada tek yöntem kenarı
    # kaçırabilir; ikisinin birleşimi çerçeveyi daha kararlı yakalar.
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    edges = cv2.Canny(blur, 50, 150)
    mask = cv2.bitwise_or(thr, edges)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Çok küçük konturları (gürültü/karakter) ele: aday en az alanın %20'si olmalı.
    min_area = 0.20 * area
    best_quad: np.ndarray | None = None
    best_area = 0.0
    for cnt in contours:
        c_area = cv2.contourArea(cnt)
        if c_area < min_area or c_area <= best_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best_quad = approx
            best_area = c_area
    if best_quad is not None:
        return _order_corners(best_quad)
    # 4-köşe konturu yoksa: en büyük konturun minAreaRect'i (eğik dikdörtgen).
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    return _order_corners(np.asarray(box, dtype="float32"))


def dewarp_plate(img: np.ndarray | None) -> np.ndarray | None:
    """Plaka ROI'sini fronto-paralel düzleştir; başarısızsa AYNEN döndür.

    Adımlar: gri → blur → (adaptif eşik | Canny) → en büyük 4-köşe dörtgen
    (yoksa minAreaRect) → getPerspectiveTransform → warpPerspective. Hedef tuval
    TR plaka en-boy oranını (520/110) korur; çıktı yüksekliği girişin en büyük
    kenarından oransal türetilir (sabit boyut dayatmaz, video-bağımsız).

    Köşe bulunamaz / görüntü çok küçük / bozuksa giriş AYNEN döner (kimlik):
    belirsizlikte yanlış warp uygulamaktansa orijinal kırpık korunur (K-004).
    """
    if img is None or getattr(img, "size", 0) == 0:
        return img
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return img  # çok küçük: warp anlamsız
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    quad = _find_quad(gray)
    if quad is None:
        return img  # güvenli dörtgen yok → kimlik

    # Hedef tuval boyutu: ROI ölçeğini koru ama TR en-boy oranına oturt. Çıktı
    # yüksekliğini kaynak köşelerin kapladığı dikey aralıktan türet (oransal).
    tl, tr, br, bl = quad
    side_top = float(np.linalg.norm(tr - tl))
    side_bottom = float(np.linalg.norm(br - bl))
    side_left = float(np.linalg.norm(bl - tl))
    side_right = float(np.linalg.norm(br - tr))
    dst_w = max(side_top, side_bottom)
    dst_h = max(side_left, side_right)
    if dst_w < 8 or dst_h < 4:
        return img
    # TR plaka oranına oturt: genişliği yükseklik × hedef-oran'dan yeniden hesapla
    # (foreshortening'i düzleştirir; il-kodu karakterlerini yatayda açar).
    out_h = int(round(dst_h))
    out_w = int(round(out_h * _TARGET_RATIO))
    if out_w < 8 or out_h < 4:
        return img
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC)
    if warped is None or warped.size == 0:
        return img
    return warped
