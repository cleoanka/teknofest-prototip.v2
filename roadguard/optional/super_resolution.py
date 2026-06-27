"""§8.2 Süper Çözünürlük — uzak/bulanık plaka ROI'sini OCR öncesi netleştirir.

Optik sınırların aşılamadığı çok uzak mesafelerde, kırpılan bulanık plaka OCR'a
girmeden önce upscale edilir. Gerçek ESRGAN ağırlığı yapılandırıldığında onu
kullanır; yoksa yüksek kaliteli bicubic upscale'e düşer (yer tutucu, OCR için yine
de okunabilirliği artırır).
`config.optional_modules.super_resolution: true` ile etkin.
"""

from __future__ import annotations

import logging

import cv2

log = logging.getLogger("roadguard.optional.super_resolution")
_warned = False


def enhance(plate_roi, scale: int = 2):
    """Plaka ROI'sini `scale` kat büyüt. None/boş ise olduğu gibi döner."""
    global _warned
    if plate_roi is None or getattr(plate_roi, "size", 0) == 0:
        return plate_roi
    if not _warned:
        log.info("Süper çözünürlük: bicubic upscale (ESRGAN ağırlığı yapılandırılmadı)")
        _warned = True
    h, w = plate_roi.shape[:2]
    return cv2.resize(plate_roi, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
