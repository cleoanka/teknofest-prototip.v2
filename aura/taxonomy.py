"""Sınıf adı taksonomisi — model-uzayı ↔ RoadGuard kanonik uzayı.

Farklı ağırlıklar aynı kavrama farklı ad verir: stok COCO ``cell phone`` der,
fine-tune modelimiz ``phone``; bir sigara modeli ``cigarette`` der, RoadGuard şeması
``smoking`` bekler. Bu eşleme TEK noktada yapılır ki model değişince pipeline,
şema ve config sözleşmesi değişmesin (hidden_prototip "iki-uzaylı taksonomi"
dersi). Yeni bir ağırlığın sınıf adları buradaki kanonik adlara çevrilemiyorsa
tek yapılacak iş bu sözlüğe satır eklemektir.
"""

from __future__ import annotations

# model çıktısı (küçük harf) → kanonik RoadGuard adı
CLASS_ALIASES: dict[str, str] = {
    # telefon
    "cell phone": "phone",
    "mobile phone": "phone",
    "cellphone": "phone",
    "telefon": "phone",
    # sigara → şemadaki davranış bayrağı 'smoking'
    "cigarette": "smoking",
    "cigar": "smoking",
    "smoke": "smoking",
    "sigara": "smoking",
    # emniyet kemeri
    "seatbelt": "no_seatbelt_evidence",  # kemer NESNESİ ihlal değildir; ayrı işaretlenir
    # araç eş anlamlıları (bazı açık veri setleri)
    "van": "minibus",
    "lorry": "truck",
    "motorbike": "motorcycle",
}


def canonical(name: str) -> str:
    """Model sınıf adını kanonik RoadGuard adına çevir (bilinmeyen ad olduğu gibi döner)."""
    return CLASS_ALIASES.get(str(name).strip().lower(), str(name))
