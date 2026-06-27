"""scene — sahne-seviyesi (ID-merkezli olmayan) bağlam katmanı.

Şu an yalnızca trafik tabelası takibini (SignTracker) içerir; aktif hız limitini
çıkarıp accumulator'ın risk değerlendirmesine besler.
"""

from roadguard.scene.sign_tracker import SCENE_TRACK_ID, SignTracker

__all__ = ["SCENE_TRACK_ID", "SignTracker"]
