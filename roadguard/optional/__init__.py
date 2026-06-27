"""optional — §8 opsiyonel modüller (lazy, default kapalı).

Toggle: config.optional_modules.*. Kapalıyken modüller import bile edilmez.
Detay: docs/mimari_ek_moduller.md.
"""

from roadguard.optional.loader import get_optional, is_enabled

__all__ = ["get_optional", "is_enabled"]
