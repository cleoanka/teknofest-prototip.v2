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
        char_margin: float = 1.5,
        confirm_min_char_margin: float | None = None,
        confirm_peak_weight: float = 0.30,
        max_reads: int = 400,
    ):
        self.min_weight = float(min_weight)
        self.margin_weight = float(margin_weight)
        self.ratio = float(ratio)
        self.fix_w = {0: 1.0, 1: float(fix1_weight), 2: float(fix2_weight)}
        self.substring_w = float(substring_weight)
        # CONFIRM zemin koşulu (gerçek video_3 dersi: UZAK plaka tutarlı şekilde
        # YANLIŞ okunabilir — '24IC8532' gibi — ve yalnız sayıca birikimle min_weight'i
        # aşıp YANLIŞ onaya gidebilir). Kazanan plaka, EN AZ BİR kez bu etkin-ağırlıkla
        # (= OCR güveni × kırpık-yüksekliği kalitesi) okunmuş OLMALI: "plakayı en az bir
        # kez NET/YAKIN gördük" güvencesi. Hep-uzak okuma → dürüst pending. Ayarlanabilir
        # (0 = kapalı). K-004: videoya değil okuma-kalitesine bağlı.
        self.confirm_peak_weight = float(confirm_peak_weight)
        # Pozisyon-hizalı karakter füzyonu: ayrı-aday kararı başarısız olursa
        # (iki format-geçerli okuma yarışıyor, ör. T↔I misread'i 34TC8532 vs
        # 34IC8532) aynı YAPIDAKİ okumalar pozisyon pozisyon birleştirilir.
        # Pozisyon-hizalı karakter füzyonu. OCR aynı plakayı varyantlara böler
        # (34TC8532/04TC8532/34IC8532 — 3↔0, T↔I); ayrı-aday kararı bunlar arasında
        # bölünür ve hangi varyant baskınsa onu (bazen yanlış 04) seçer. Füzyon
        # pozisyon pozisyon en güçlü karakteri alır; ONAY için her pozisyonda kazanan
        # ikinciyi 'confirm_char_margin' (ONAY-sıkı eşik, aşağıda) MUTLAK ağırlıkla geçmeli
        # — bir pozisyon belirsizse dürüst 'pending' (ASLA yanlış plaka onaylanmaz).
        # char_consensus=False ise füzyon yalnız best_partial kanıt izinde kalır (orası
        # EŞİKSİZ — pozisyon başına en baskın karakter; char_margin/confirm_char_margin
        # UYGULANMAZ, bkz. _char_fuse_best), onaya girmez.
        self.char_consensus = bool(char_consensus)
        self.char_margin = float(char_margin)
        # ONAY için pozisyon-kesinliği eşiği (char_margin'den SIKI olabilir). Onayda HER
        # pozisyonda kazanan, ikinciyi bu MUTLAK ağırlıkla geçmeli. Gerçek video ölçümü
        # (17 Haz, stok yolo26l): yanlış ilk-harf '0' pos0-margin'i ~1.55, doğru '3'
        # margin'i ~1.52 — ikisi de char_margin=1.5'i geçip YANLIŞ onaya gidiyordu. Bu
        # ayrı/sıkı eşik (vars. 2.0) belirsiz ilk-karakteri dürüst PENDING yapar; net
        # plaka (video_2, margin yüksek) onaylanmaya devam eder. best_partial füzyonu ve
        # diğer kanıt-izi yolları char_margin'i kullanmaya devam eder (kanıt izi gevşek,
        # ONAY sıkı). None → char_margin'e düşer (geriye dönük uyum). K-004: oran/ağırlık
        # temelli, videoya-özel sabit DEĞİL — okuma-belirsizliğine bağlı.
        if confirm_min_char_margin is None:
            self.confirm_char_margin = self.char_margin
        else:
            self.confirm_char_margin = max(self.char_margin, float(confirm_min_char_margin))
        self.max_reads = int(max_reads)
        self.raw_reads: list[tuple[str, float]] = []  # (metin, etkin kanıt ağırlığı)
        # PERF: normalize_tr(raw) sonucu (cand, fixes) okuma başına SABİTtir ama
        # consensus/best_partial/_weights HER karede tüm raw_reads üzerinde yeniden
        # çalıştırıyordu (consensus her update'te çağrıldığından birikimli O(N²)).
        # add()'te bir kez normalize edip burada cache'liyoruz → tüm okuyucular O(N).
        # raw_reads ile İNDEKS-HİZALI: _norm[i] == normalize_tr(raw_reads[i][0]).
        self._norm: list[tuple[str | None, int]] = []

    def add(self, text: str | None, conf: float = 1.0, weight: float = 1.0) -> None:
        """Okuma ekle. ``weight``: kaynak-kalitesi çarpanı (0..1).

        Gerçek video dersi (12 Haz akşam ölçümü): UZAK kareden gelen sistematik
        misread'ler ("041C8532", "34IC8532"≡T→I formatça GEÇERLİ!) sayıca üstünlük
        kurup konsensüsü kilitliyordu. Okumanın kanıt değeri OCR güveni × kaynak
        kalitesidir (plaka kırpık yüksekliğinden türetilir, reader hesaplar);
        yakın/net okuma uzak/bulanık okumayı hem güvenle hem ağırlıkla ezer.
        """
        if not text:
            return
        eff = max(0.0, min(1.0, float(conf))) * max(0.0, min(1.0, float(weight)))
        if len(self.raw_reads) < self.max_reads:
            self.raw_reads.append((text, eff))
            self._norm.append(normalize_tr(text))  # PERF: bir kez normalize, cache'le
            return
        # Havuz dolu: yeni okuma en düşük-ağırlıklı mevcut okumadan DAHA kaliteliyse onu
        # değiştir. Aksi halde yavaş yaklaşan araçta erken biriken uzak/bulanık okumalar
        # havuzu kilitleyip sonradan gelen yakın/net okumanın konsensüsü düzeltmesini
        # engellerdi (Codex bulgusu). Kalite-ağırlıklı replacement bu kilidi açar.
        min_i = min(range(len(self.raw_reads)), key=lambda i: self.raw_reads[i][1])
        if eff > self.raw_reads[min_i][1]:
            self.raw_reads[min_i] = (text, eff)
            self._norm[min_i] = normalize_tr(text)

    # --- iç hesap ----------------------------------------------------------- #
    def _weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        invalid: list[str] = []
        for (raw, conf), (cand, fixes) in zip(self.raw_reads, self._norm, strict=True):
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
        """Konsensüs yokken raporlanacak en güçlü aday (KANIT İZİ — KESİN DEĞİL).

        ``PlateState.partial`` hâlâ 'pending' iken raporlanan 'en olası tahmin'.
        char_consensus açıksa pozisyon-hizalı füzyonun EŞİKSİZ sürümünü (her pozisyonda
        en baskın karakter) kullanır; yoksa ağırlık-sıralı en güçlü adaya düşer.
        (Onay için EŞİKLİ sürüm ``consensus`` içinde — bir pozisyon belirsizse pending.)
        """
        raw_valid: dict[str, float] = {}
        for (_raw, conf), (cand, fixes) in zip(self.raw_reads, self._norm, strict=True):
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
        peak: dict[str, float] = {}  # aday başına EN GÜÇLÜ tek okuma (net/yakın kanıt zemini)
        for (_raw, conf), (cand, fixes) in zip(self.raw_reads, self._norm, strict=True):
            if cand is not None and fixes == 0:
                raw_valid[cand] = raw_valid.get(cand, 0.0) + conf
                peak[cand] = max(peak.get(cand, 0.0), conf)
        if not raw_valid:
            return None, 0.0
        ranked = sorted(raw_valid.items(), key=lambda kv: kv[1], reverse=True)
        top, w_top = ranked[0]
        w_second = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(raw_valid.values())
        # Ayrı-aday onayı: bütün-string marjı + ZEMİN (en az bir net okuma) + her
        # KARAKTER pozisyonunun belirsiz olmaması. Bütün-string baskın olsa bile tek bir
        # pozisyon çekişmeliyse (ör. yolo26l video_1: pos0 0↔3) ya da plaka hiç net
        # görülmemişse (video_3: hep-uzak) onaylamayız → char füzyonuna düşer, o da
        # belirsizse dürüst pending. Hiçbir koşulda YANLIŞ plaka kesinleşmez.
        if (
            w_top >= self.min_weight
            and (w_top - w_second) >= self.margin_weight
            and w_top / total >= self.ratio
            and peak.get(top, 0.0) >= self.confirm_peak_weight
            and self._position_unambiguous(top, raw_valid)
        ):
            return top, round(min(1.0, w_top / total), 2)
        # Tek-varyant kararı yok: OCR aynı plakayı varyantlara bölmüş olabilir
        # (3↔0, T↔I). Pozisyon-hizalı füzyonu dene — her pozisyon NET ise onayla.
        if self.char_consensus:
            cand, conf = self._char_consensus(raw_valid, peak)
            if cand is not None:
                return cand, conf
        return None, round(w_top / max(total, 1e-9), 2)

    def _position_unambiguous(self, winner: str, raw_valid: dict[str, float]) -> bool:
        """Kazanan stringin HER karakteri, aynı-yapıdaki diğer ham-geçerli okumalarda
        kendisine ``confirm_char_margin`` içinde bir rakip karaktere sahip OLMAMALI.

        Tek-yapılı (rakipsiz) okuma → çekişme yok → True. Bir pozisyonda kazanan ile
        ikinci karakter arasındaki ağırlık farkı ``confirm_char_margin``'in (ONAY-sıkı
        eşik) altındaysa o pozisyon belirsizdir → False (ayrı-aday onayını VETO eder;
        dürüstlük). char_consensus kapalıysa veto da kapalıdır (eski davranış)."""
        if not self.char_consensus:
            return True
        pat = tuple("D" if c.isdigit() else "L" for c in winner)
        members = [
            (t, w)
            for t, w in raw_valid.items()
            if tuple("D" if c.isdigit() else "L" for c in t) == pat
        ]
        if len(members) < 2:
            return True
        for i, wc in enumerate(winner):
            char_w: dict[str, float] = {}
            for t, w in members:
                char_w[t[i]] = char_w.get(t[i], 0.0) + w
            win_w = char_w.get(wc, 0.0)
            other = max((w for c, w in char_w.items() if c != wc), default=0.0)
            if (win_w - other) < self.confirm_char_margin:
                return False
        return True

    def _char_consensus(
        self, raw_valid: dict[str, float], peak: dict[str, float] | None = None
    ) -> tuple[str | None, float]:
        """Pozisyon-hizalı karakter füzyonu (CONFIRMED kararı için, güvenli).

        Aynı YAPIDAKİ (uzunluk + rakam/harf deseni) ham-geçerli okumalar pozisyon
        pozisyon birleştirilir. ONAY için HER pozisyonda kazanan karakter, ikinciyi
        ``confirm_char_margin`` (ONAY-sıkı eşik) MUTLAK ağırlıkla geçmeli + grup toplamı
        ``min_weight``'i tutmalı. Bir pozisyon belirsizse (ör. 0↔3 neredeyse eşit, ya da uzaktan I↔T)
        ``None`` döner → dürüst ``pending`` (yanlış plaka ASLA onaylanmaz). Bu,
        OCR'ın doğru plakayı birden çok varyanta böldüğü (34TC8532/04TC8532/34IC8532)
        ama her pozisyonun çoğunluğunun doğru olduğu durumu çözer — K-004: kural
        videoya değil pozisyon-istatistiğine bağlı.
        """
        groups: dict[tuple, list[tuple[str, float]]] = {}
        for text, w in raw_valid.items():
            pattern = tuple("D" if c.isdigit() else "L" for c in text)
            groups.setdefault(pattern, []).append((text, w))
        if not groups:
            return None, 0.0
        best_pattern = max(groups, key=lambda p: sum(w for _, w in groups[p]))
        members = groups[best_pattern]
        # Tek-üye grup ayrı-aday kararının kopyasıdır (füzyon yeni bilgi vermez);
        # ayrıca tek-üyeli rakip plakaların (34ABC123 vs 06XY999) yanlış onayını önler.
        if len(members) < 2:
            return None, 0.0
        group_w = sum(w for _, w in members)
        if group_w < self.min_weight:
            return None, 0.0
        # ZEMİN koşulu: bu yapı grubundaki okumalardan en az biri net/yakın (peak >=
        # confirm_peak_weight) olmalı — hep-uzak (sistematik yanlış) okuma onaylanmaz.
        if peak is not None and self.confirm_peak_weight > 0:
            if max((peak.get(t, 0.0) for t, _ in members), default=0.0) < self.confirm_peak_weight:
                return None, 0.0
        out: list[str] = []
        for i in range(len(best_pattern)):
            char_w: dict[str, float] = {}
            for text, w in members:
                char_w[text[i]] = char_w.get(text[i], 0.0) + w
            ranked = sorted(char_w.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            top_c, top_w = ranked[0]
            second_w = ranked[1][1] if len(ranked) > 1 else 0.0
            if (top_w - second_w) < self.confirm_char_margin:
                return None, 0.0  # bu pozisyon belirsiz → dürüst çekimserlik (ONAY-sıkı eşik)
            out.append(top_c)
        cand = "".join(out)
        if not TR_PLATE_RE.match(cand) or not 1 <= int(cand[:2]) <= 81:
            return None, 0.0
        total = sum(raw_valid.values())
        return cand, round(min(1.0, group_w / max(total, 1e-9)), 2)
