"""TR plaka normalizasyonu + format-öncelikli ağırlıklı oylama.

Gerçek video dersleri (v1 baseline ölçümü, 12 Haz 2026):
    EasyOCR aynı plakayı kareler arasında farklı okur: ``34TC8532`` →
    ``041C8532`` (3→0, T→1), ``8532`` (sol blok kadraj/parlaklık yüzünden
    okunamadı)... Ham metinle çoğunluk oylaması bu varyantlara bölünür ve
    konsensüs HİÇ oluşmaz (baseline'da 11×PLATE_REJECTED, 0×CONFIRMED).

Çözüm iki katman (v1 ``plate_ocr.py`` + ``PlateTracker`` fikirlerinin portu):
1.  **Pozisyon-farkında normalizasyon**: TR formatı (2 rakam il + 1-3 harf +
    2-4 rakam) blok blok parse edilir; rakam bloklarında harf-görünümlü
    karakterler (O→0, I→1, B→8...), harf bloğunda rakam-görünümlüler (1→I,
    0→O, 8→B...) düzeltilir. Kaç ikame gerektiği sayılır.
2.  **Format-öncelikli ağırlıklı oylama**: ikamesiz format-geçerli okuma tam
    oy (1.0), 1-ikameli düzeltme kısmi oy (0.45), 2-ikameli 0.20 alır;
    geçersiz okumalar aday OLUŞTURMAZ ama bir adayın ALT-DİZİSİyse (ör.
    ``8532`` ⊂ ``34TC8532``) ona küçük destek (0.25) verir. Böylece yanlış
    ama "düzeltilebilir" varyantlar doğru ham okumayı asla ezemez —
    K-004: hiçbir kural tek videoya özgü değildir.
"""

from __future__ import annotations

import re
from collections import Counter

TR_PLATE_RE = re.compile(r"^\d{2}[A-Z]{1,3}\d{2,4}$")

# Rakam beklenen pozisyonda harf görüldüyse → en olası rakam (muhafazakâr küme)
_LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}
# Harf beklenen pozisyonda rakam görüldüyse → en olası harf
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}


def normalize_tr(raw: str) -> tuple[str | None, int]:
    """Ham OCR metnini TR plaka formatına oturtmaya çalış.

    Dönüş: ``(normalize_plaka | None, ikame_sayısı)``. Ham metin zaten geçerliyse
    ``(raw, 0)``. Birden çok blok-bölmesi mümkünse EN AZ ikame gerektiren seçilir.
    Güvenle normalize edilemiyorsa ``(None, 0)``.
    """
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if TR_PLATE_RE.match(s):
        prov = int(s[:2])
        return (s, 0) if 1 <= prov <= 81 else (None, 0)
    if not 6 <= len(s) <= 9:
        return None, 0

    best: tuple[str, int] | None = None
    for letters_len in (1, 2, 3):
        prov, mid, suf = s[:2], s[2 : 2 + letters_len], s[2 + letters_len :]
        if not (len(mid) == letters_len and 2 <= len(suf) <= 4):
            continue
        fixes = 0
        ok = True
        out = []
        for ch in prov:
            if ch.isdigit():
                out.append(ch)
            elif ch in _LETTER_TO_DIGIT:
                out.append(_LETTER_TO_DIGIT[ch])
                fixes += 1
            else:
                ok = False
                break
        if not ok:
            continue
        for ch in mid:
            if ch.isalpha():
                out.append(ch)
            elif ch in _DIGIT_TO_LETTER:
                out.append(_DIGIT_TO_LETTER[ch])
                fixes += 1
            else:
                ok = False
                break
        if not ok:
            continue
        for ch in suf:
            if ch.isdigit():
                out.append(ch)
            elif ch in _LETTER_TO_DIGIT:
                out.append(_LETTER_TO_DIGIT[ch])
                fixes += 1
            else:
                ok = False
                break
        if not ok:
            continue
        cand = "".join(out)
        if not TR_PLATE_RE.match(cand) or not 1 <= int(cand[:2]) <= 81:
            continue
        if best is None or fixes < best[1]:
            best = (cand, fixes)
    return best if best else (None, 0)


class PlateVotePool:
    """Track ömrü boyunca biriken, format-öncelikli ağırlıklı oy havuzu.

    v2'nin eski 7'lik ``VotingBuffer``'ından farkları:
    - Oylar redde SIFIRLANMAZ; track yaşadıkça birikir (v1 PlateTracker dersi:
      "97 oy vs 1 oy" kararlılığı ancak birikimle oluşur).
    - Geçerli-format okuma > düzeltilmiş okuma > alt-dizi desteği hiyerarşisi.
    - Kazanan, ikinciye `margin` farkla VE `min_weight` mutlak ağırlıkla önde
      olmalı (erken yanlış kilitlenme koruması).
    """

    def __init__(
        self,
        min_weight: float = 2.0,
        margin_weight: float = 1.5,
        ratio: float = 0.6,
        fix1_weight: float = 0.45,
        fix2_weight: float = 0.20,
        substring_weight: float = 0.25,
        char_consensus: bool = True,
        max_reads: int = 400,
    ):
        self.min_weight = float(min_weight)
        self.margin_weight = float(margin_weight)
        self.ratio = float(ratio)
        self.fix_w = {0: 1.0, 1: float(fix1_weight), 2: float(fix2_weight)}
        self.substring_w = float(substring_weight)
        # Pozisyon-hizalı karakter füzyonu: ayrı-aday kararı başarısız olursa
        # (iki format-geçerli okuma yarışıyor, ör. T↔I misread'i 34TC8532 vs
        # 34IC8532) aynı YAPIDAKİ okumalar pozisyon pozisyon birleştirilir.
        # Pozisyon-hizalı karakter füzyonu YALNIZ best_partial (kanıt izi) içindir;
        # CONFIRMED kararına KATILMAZ. Gerçek video dersi: uzak/bulanık karelerde OCR
        # sistematik yanlış okuyor (T→I, 3→2) ve doğru okuma hiç gelmiyor; füzyonla
        # 'onaylamak' yanlış plakayı kesinleştirir — 'okuyamadım' (pending + partial)
        # daha dürüst. Onay yalnız katı ayrı-aday konsensüsüyle verilir.
        self.char_consensus = bool(char_consensus)
        self.max_reads = int(max_reads)
        self.raw_reads: list[tuple[str, float]] = []  # (metin, etkin kanıt ağırlığı)

    def add(self, text: str | None, conf: float = 1.0, weight: float = 1.0) -> None:
        """Okuma ekle. ``weight``: kaynak-kalitesi çarpanı (0..1).

        Gerçek video dersi (12 Haz akşam ölçümü): UZAK kareden gelen sistematik
        misread'ler ("041C8532", "34IC8532"≡T→I formatça GEÇERLİ!) sayıca üstünlük
        kurup konsensüsü kilitliyordu. Okumanın kanıt değeri OCR güveni × kaynak
        kalitesidir (plaka kırpık yüksekliğinden türetilir, reader hesaplar);
        yakın/net okuma uzak/bulanık okumayı hem güvenle hem ağırlıkla ezer.
        """
        if text and len(self.raw_reads) < self.max_reads:
            eff = max(0.0, min(1.0, float(conf))) * max(0.0, min(1.0, float(weight)))
            self.raw_reads.append((text, eff))

    # --- iç hesap ----------------------------------------------------------- #
    def _weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        invalid: list[str] = []
        for raw, conf in self.raw_reads:
            cand, fixes = normalize_tr(raw)
            if cand is not None and fixes in self.fix_w:
                weights[cand] = weights.get(cand, 0.0) + self.fix_w[fixes] * conf
            else:
                invalid.append(re.sub(r"[^A-Z0-9]", "", raw.upper()))
        # Geçersiz ama bir adayın alt-dizisi olan okumalar o adaya küçük destek verir
        # ("8532" gibi kesik okumalar kanıtı güçlendirir, yeni aday üretmez).
        for frag in invalid:
            if len(frag) < 3:
                continue
            for cand in weights:
                if frag in cand:
                    weights[cand] += self.substring_w
        return weights

    def counts(self) -> dict[str, int]:
        """Ham okuma sayımı (telemetri/PLATE_REJECTED payload'ı için)."""
        return dict(Counter(t for t, _ in self.raw_reads))

    def best_partial(self) -> str | None:
        """Konsensüs yokken raporlanacak en güçlü aday (yalnızca KANIT İZİ).

        ÖNEMLİ: Bu yalnız 'en olası tahmin'dir — KESİN DEĞİLDİR (PlateState.partial,
        status hâlâ 'pending'). char_consensus açıksa pozisyon-hizalı karakter
        füzyonu (eşiksiz) en olası birleşik plakayı verir; yoksa ağırlık-sıralı en
        güçlü adaya düşer. Karara (CONFIRMED) ASLA katılmaz; o yalnız katı ayrı-aday
        konsensüsüyle verilir (yanlış onay üretmemek için — gerçek video dersi:
        uzak/bulanık karelerde OCR sistematik yanlış okuyor, bunu CONFIRMED yapmak
        'okuyamadım' demekten kötü).
        """
        raw_valid: dict[str, float] = {}
        for raw, conf in self.raw_reads:
            cand, fixes = normalize_tr(raw)
            if cand is not None and fixes == 0:
                raw_valid[cand] = raw_valid.get(cand, 0.0) + conf
        if self.char_consensus and len(raw_valid) > 1:
            fused = self._char_fuse_best(raw_valid)
            if fused is not None:
                return fused
        w = self._weights()
        if w:
            return max(w, key=lambda k: w[k])
        c = Counter(t for t, _ in self.raw_reads)
        return c.most_common(1)[0][0] if c else None

    def _char_fuse_best(self, raw_valid: dict[str, float]) -> str | None:
        """Eşiksiz pozisyonel füzyon (best_partial için): en ağır yapı grubunda
        pozisyon başına en baskın karakter — onay eşiği aramaz, kanıt izi üretir."""
        groups: dict[tuple, list[tuple[str, float]]] = {}
        for text, w in raw_valid.items():
            pattern = tuple("D" if c.isdigit() else "L" for c in text)
            groups.setdefault(pattern, []).append((text, w))
        if not groups:
            return None
        best_pattern = max(groups, key=lambda p: sum(w for _, w in groups[p]))
        members = groups[best_pattern]
        out = []
        for i in range(len(best_pattern)):
            char_w: dict[str, float] = {}
            for text, w in members:
                char_w[text[i]] = char_w.get(text[i], 0.0) + w
            out.append(max(char_w.items(), key=lambda kv: (kv[1], kv[0]))[0])
        cand = "".join(out)
        return cand if TR_PLATE_RE.match(cand) else None

    def consensus(self) -> tuple[str | None, float]:
        """(kazanan|None, güven 0..1).

        KARAR yalnızca İKAMESİZ format-geçerli ham okumalara dayanır (en güçlü
        kanıt sınıfı) ve her okuma OCR GÜVENİYLE ağırlıklanır: yakın/net plakadan
        gelen okuma, uzak/bulanık okumadan daha değerlidir (sistematik tek-karakter
        hatası — ör. uzaktan 3→0 — yüksek güvenli yakın okumalarca ezilir).
        Kazanan ``min_weight`` toplam ağırlığa, ikinciye ``margin_weight`` farka ve
        ham-geçerli ağırlıklar içinde ``ratio`` paya sahip olmalı. Düzeltilmiş
        (ikameli) ve kesik okumalar karara KATILMAZ — yalnız ``best_partial`` ve
        güven görüntülemesine katkı verir (erken-yanlış-kilit koruması).
        """
        raw_valid: dict[str, float] = {}
        for raw, conf in self.raw_reads:
            cand, fixes = normalize_tr(raw)
            if cand is not None and fixes == 0:
                raw_valid[cand] = raw_valid.get(cand, 0.0) + conf
        if not raw_valid:
            return None, 0.0
        ranked = sorted(raw_valid.items(), key=lambda kv: kv[1], reverse=True)
        top, w_top = ranked[0]
        w_second = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(raw_valid.values())
        if (
            w_top >= self.min_weight
            and (w_top - w_second) >= self.margin_weight
            and w_top / total >= self.ratio
        ):
            return top, round(min(1.0, w_top / total), 2)
        return None, round(w_top / max(total, 1e-9), 2)
