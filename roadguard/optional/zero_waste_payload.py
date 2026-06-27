"""§8.1 Sıfır-Atık Veri Aktarımı (Zero-Waste Payload).

Downstream'e tam çözünürlüklü kare gönderilmez; yalnızca küçük ROI görüntüsü +
ID'ye bağlı yapısal metin iletilir. 5G bant genişliğini gereksiz tüketmez.
`config.optional_modules.zero_waste_payload: true` ile etkin.
"""

from __future__ import annotations

import base64

import cv2

_STRUCT_KEYS = (
    "cls",
    "plate",
    "plate_status",
    "driver",
    "speed_kmh",
    "relative_velocity_flag",
    "risk_flags",
    "qod_active",
)


def build_payload(track_dict: dict, plate_roi=None, jpeg_quality: int = 70) -> dict:
    """Track için kompakt payload: yapısal metin + (varsa) küçük plaka ROI JPEG (base64)."""
    payload = {
        "track_id": track_dict.get("track_id"),
        "structured": {k: track_dict.get(k) for k in _STRUCT_KEYS},
    }
    if plate_roi is not None and getattr(plate_roi, "size", 0) > 0:
        ok, buf = cv2.imencode(".jpg", plate_roi, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            payload["plate_roi_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
            payload["roi_bytes"] = int(buf.size)
    return payload
