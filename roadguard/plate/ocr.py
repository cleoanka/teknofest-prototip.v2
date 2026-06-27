"""OCR motorları — gerçek (EasyOCR / PaddleOCR) ve deterministik mock.

- `RealOCR`: EasyOCR ile plaka ROI'sinden metin okur (sentetik videodaki çizili
  plakaları gerçekten okuyabilir). Varsayılan motor.
- `PaddleOCRReader`: PaddleOCR'ı EasyOCR'ın `readtext` çıktısıyla UYUMLU bir
  sarmalayıcıyla (bbox, metin, güven) sunar; aynı `_merge_line` satır-birleştirme
  + TR-normalizasyon hattı çalışır. config `plate.ocr_engine: paddleocr` ile
  seçilir; paddleocr kurulu değilse LOGLU olarak EasyOCR'a düşülür.
- `MockOCR`: EasyOCR/torch yokken araç renginden senaryo plakasını üretir
  (track başına kararlı → voting konsensüsü oluşur).
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from roadguard.config import is_synthetic_source

if TYPE_CHECKING:
    pass

log = logging.getLogger("roadguard.plate.ocr")
_NON_ALNUM = re.compile(r"[^A-Z0-9]")


class OCREngine(ABC):
    @abstractmethod
    def read(self, plate_roi, vehicle_crop=None) -> tuple[str | None, float]:
        """Plaka ROI'sinden (metin|None, güven) döndür."""
        raise NotImplementedError


class RealOCR(OCREngine):
    def __init__(self, cfg):
        import easyocr

        from roadguard.device import cuda_is_usable

        langs = list(cfg.get("plate.ocr_lang", ["tr"]))
        # GPU varsa (ve doğrulanmış torch derlemesiyle çalışıyorsa) OCR'ı da
        # hızlandır; aksi halde CPU. cuda_is_usable() önbellekli probe kullanır.
        use_gpu = bool(cfg.get("plate.ocr_gpu", True)) and cuda_is_usable()
        self.reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
        # 4K araç crop'ları OCR'ı gereksiz yavaşlatır: uzun kenar bu değeri aşarsa
        # küçültülür (plaka okunaklılığı korunur, süre kat kat düşer).
        self.max_side = int(cfg.get("plate.ocr_max_side", 1280))
        # Küçük ROI'lerde (yükseklik < enhance_below) CLAHE+2x upscale varyantı denenir.
        self.enhance_below = int(cfg.get("plate.ocr_enhance_below_px", 64))
        log.info("EasyOCR yüklendi (langs=%s, gpu=%s)", langs, use_gpu)

    @staticmethod
    def _merge_line(results) -> tuple[str | None, float]:
        """EasyOCR segmentlerini satır bazında soldan sağa birleştir.

        Plaka çoğu karede '34' + 'TC' + '8532' gibi AYRI kutular halinde döner;
        yalnızca en güvenli tek kutuyu almak kesik okuma ('8532') üretir
        (v1 multi-block concat dersi). En güvenli kutunun satırındaki tüm
        kutular x'e göre sıralanıp birleştirilir.
        """
        if not results:
            return None, 0.0
        best = max(results, key=lambda r: r[2])
        bys = [p[1] for p in best[0]]
        b_cy = sum(bys) / len(bys)
        b_h = max(bys) - min(bys)
        line = []
        for box, txt, conf in results:
            ys = [p[1] for p in box]
            cy = sum(ys) / len(ys)
            if abs(cy - b_cy) <= max(b_h * 0.7, 8.0):
                line.append((min(p[0] for p in box), txt, conf))
        line.sort(key=lambda t: t[0])
        text = _NON_ALNUM.sub("", "".join(t[1] for t in line).upper())
        confs = [t[2] for t in line]
        # NOT: parantezleme ÖNEMLİ — ternary yalnız güven skalerini kapsamalı,
        # yoksa (operatör önceliği) confs boşken fonksiyon TUPLE yerine bare 0.0
        # döndürürdü (sözleşme ihlali). best daima line'da olduğundan confs pratikte
        # boş olmaz; bu parantez gelecekteki line-filtre değişikliğine karşı sözleşmeyi sabitler.
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return (text or None), avg_conf

    def _readtext(self, img) -> tuple[str | None, float]:
        return self._merge_line(self.reader.readtext(img))

    def read(self, plate_roi, vehicle_crop=None) -> tuple[str | None, float]:
        if plate_roi is None or getattr(plate_roi, "size", 0) == 0:
            return None, 0.0
        import cv2

        roi = plate_roi
        # Parlama/far testi (hidden_prototip dersi): aşırı parlak + düşük varyans
        # ROI ışık kaynağıdır, plaka değil → OCR'a hiç girmeden atla (FP + süre kazancı).
        mean = float(roi.mean())
        std = float(roi.std())
        if mean > 215.0 and std < 25.0:
            return None, 0.0
        h, w = roi.shape[:2]
        long_side = max(h, w)
        if long_side > self.max_side:
            scale = self.max_side / long_side
            roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        text, conf = self._readtext(roi)
        if text is None and roi.shape[0] < self.enhance_below:
            # Küçük/karanlık plaka: 2x büyüt + L kanalında CLAHE, bir kez daha dene.
            up = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
            lab_l, lab_a, lab_b = cv2.split(lab)
            lab_l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab_l)
            up = cv2.cvtColor(cv2.merge((lab_l, lab_a, lab_b)), cv2.COLOR_LAB2BGR)
            text, conf = self._readtext(up)
        return text, conf


class _PaddleAdapter:
    """PaddleOCR'ı EasyOCR `readtext` arayüzüne uyarlar.

    EasyOCR `readtext(img)` → `[(box, text, conf), ...]` döner; box = 4 köşe
    [[x,y],...]. PaddleOCR sürümleri çıktıyı farklı sarar:
      • eski: `[[ [box, (text, conf)], ... ]]`  (predict ilk eleman)
      • yeni: `[{"rec_texts": [...], "rec_scores": [...], "dt_polys": [...]}]`
    İkisi de aynı `(box, text, conf)` üçlüsüne indirgenir; `_merge_line` ve TR
    normalizasyonu motor-bağımsız çalışır.
    """

    def __init__(self, engine):
        self._engine = engine

    @staticmethod
    def _norm_box(box) -> list[list[float]]:
        pts = [[float(p[0]), float(p[1])] for p in box]
        return pts if pts else [[0.0, 0.0]]

    def readtext(self, img) -> list:
        try:
            raw = self._engine.ocr(img)
        except TypeError:
            # Bazı sürümler konumsal 'cls' bekler; uyumlu en geniş çağrı.
            raw = self._engine.ocr(img, cls=True)
        out: list = []
        if not raw:
            return out
        first = raw[0]
        # Yeni sözlük formatı (PaddleX / PP-OCRv4+).
        if isinstance(first, dict):
            texts = first.get("rec_texts", []) or []
            scores = first.get("rec_scores", []) or []
            polys = first.get("dt_polys", first.get("rec_polys", [])) or []
            for i, txt in enumerate(texts):
                conf = float(scores[i]) if i < len(scores) else 0.0
                box = self._norm_box(polys[i]) if i < len(polys) else [[0.0, 0.0]]
                out.append((box, str(txt), conf))
            return out
        # Eski liste formatı: first = [[box, (text, conf)], ...]
        for line in first:
            try:
                box, (txt, conf) = line[0], line[1]
            except (TypeError, ValueError, IndexError):
                continue
            out.append((self._norm_box(box), str(txt), float(conf)))
        return out


class PaddleOCRReader(RealOCR):
    """PaddleOCR motoru — `RealOCR`'ın okuma/birleştirme hattını aynen kullanır.

    Yalnız `__init__` motoru PaddleOCR'a çevirir (`self.reader` bir _PaddleAdapter
    olur, EasyOCR'ın `readtext` sözleşmesini taklit eder). `read`, `_readtext`,
    `_merge_line` ve düşük-güven CLAHE+2x varyantı RealOCR'dan miras alınır →
    motor değişse de OCR-sonrası mantık BİREBİR korunur.
    """

    def __init__(self, cfg):
        from paddleocr import PaddleOCR

        from roadguard.device import cuda_is_usable

        use_gpu = bool(cfg.get("plate.ocr_gpu", True)) and cuda_is_usable()
        self._engine = self._build_engine(PaddleOCR, use_gpu)
        self.reader = _PaddleAdapter(self._engine)
        self.max_side = int(cfg.get("plate.ocr_max_side", 1280))
        self.enhance_below = int(cfg.get("plate.ocr_enhance_below_px", 64))

    @staticmethod
    def _build_engine(PaddleOCR, use_gpu: bool):
        """PaddleOCR'ı sürüm-bağımsız kur — açı/oryantasyon sınıflandırması ASLA sessizce kaybolmaz.

        PaddleOCR 2.7+/3.x imzayı kırdı: ``show_log`` kaldırıldı, ``use_angle_cls``
        → ``use_textline_orientation`` oldu, GPU seçimi ``use_gpu=bool`` yerine
        ``device="gpu"/"cpu"`` ile yapılır. Sıra:
          1) MODERN imza: ``use_textline_orientation=True`` + ``device=``
          2) TypeError → ESKİ imza: ``use_angle_cls=True`` + ``use_gpu=``
          3) son çare: yalnız ``lang`` (orientation default'a bırakılır, loglanır)
        Her iki başarılı yolda da textline-orientation / angle-cls AÇIK kalır.
        """
        device = "gpu" if use_gpu else "cpu"
        # 1) Modern imza (PaddleOCR 3.x / PP-OCRv4+).
        try:
            engine = PaddleOCR(use_textline_orientation=True, lang="en", device=device)
            log.info(
                "PaddleOCR yüklendi (modern imza: use_textline_orientation=True, device=%s)",
                device,
            )
            return engine
        except TypeError:
            pass
        # 2) Eski imza (PaddleOCR <=2.6: use_angle_cls + use_gpu).
        try:
            engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=use_gpu)
            log.info("PaddleOCR yüklendi (eski imza: use_angle_cls=True, use_gpu=%s)", use_gpu)
            return engine
        except TypeError:
            pass
        # 3) Ara sürüm: angle bayrağı kabul ama GPU bayrağı reddediyor (ya da tersi).
        #    Orientation sınıflandırmasını yine de AÇIK tutmaya çalış (sessizce kaybetme).
        for kwargs, note in (
            ({"use_textline_orientation": True, "lang": "en"}, "use_textline_orientation=True"),
            ({"use_angle_cls": True, "lang": "en"}, "use_angle_cls=True"),
        ):
            try:
                engine = PaddleOCR(**kwargs)
                log.warning(
                    "PaddleOCR yüklendi (%s; GPU/device bayrağı bu sürümde geçirilemedi → "
                    "varsayılan cihaz kullanılıyor)",
                    note,
                )
                return engine
            except TypeError:
                continue
        # 4) Son çare: yalnız lang. Orientation default'a kalır — bunu AÇIKÇA logla.
        log.warning(
            "PaddleOCR kurulumu yalnız lang ile yapıldı (use_gpu=%s istenmişti); açı/oryantasyon "
            "sınıflandırması ve cihaz seçimi bu sürümde geçirilemedi.",
            use_gpu,
        )
        return PaddleOCR(lang="en")


class _FastPlateAdapter:
    """fast-plate-ocr (ONNX, plakaya-ÖZEL OCR) → EasyOCR `readtext` arayüzü.

    fast-plate-ocr TÜM plakayı TEK bir string olarak okur (karakter-segmentasyonu
    yok); EasyOCR `readtext(img)` → `[(box, text, conf), ...]` çoklu-kutu döndürür.
    Bu yüzden çıktı TEK kutu olarak sarılır: box = ROI'nin dört köşesi, text =
    okunan plaka, conf = pad-DIŞI karakter olasılıklarının ortalaması. `_merge_line`
    bu tek kutuyu olduğu gibi geçirir; TR-normalizasyon hattı motor-bağımsız çalışır.

    Model 1-kanal (gri) giriş bekler; gelen BGR/3-kanal ROI griye çevrilir.
    """

    def __init__(self, recognizer):
        self._rec = recognizer

    def readtext(self, img) -> list:
        if img is None or getattr(img, "size", 0) == 0:
            return []
        import cv2

        gray = img
        if getattr(img, "ndim", 0) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            preds = self._rec.run(gray, return_confidence=True)
        except Exception as exc:  # noqa: BLE001 — motor hatası okuma boşa düşmeli, hattı kırmamalı
            # Gözlemlenebilirlik: KALICI arıza (her karede hata) DEBUG'da görünmez kalıp
            # 'plaka yok' gibi sessizce sürerdi. İlk hatayı bir kez WARNING'e yükselt
            # (kök-neden görünsün), sonrakileri DEBUG'da tut (spam önle).
            if not getattr(self, "_run_err_warned", False):
                log.warning("fast-plate-ocr run hatası (motor arızası olabilir): %s", exc)
                self._run_err_warned = True
            else:
                log.debug("fast-plate-ocr run hatası: %s", exc)
            return []
        if not preds:
            return []
        pred = preds[0]
        text = getattr(pred, "plate", None)
        if not text:
            return []
        probs = getattr(pred, "char_probs", None)
        if probs is not None and len(probs) > 0:
            # plaka uzunluğu kadar (pad-dışı) karakterin olasılık ortalaması
            n = min(len(text), len(probs))
            conf = float(sum(float(p) for p in probs[:n]) / n) if n else 0.0
        else:
            conf = 0.0
        h, w = gray.shape[:2]
        box = [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]
        return [(box, str(text), conf)]


class FastPlateOCRReader(RealOCR):
    """fast-plate-ocr motoru — `RealOCR`'ın okuma/birleştirme hattını aynen kullanır.

    Yalnız `__init__` motoru fast-plate-ocr'a çevirir (`self.reader` bir
    _FastPlateAdapter olur). `read`, `_readtext`, `_merge_line` ve düşük-güven
    CLAHE+2x varyantı RealOCR'dan miras alınır → motor değişse de OCR-sonrası
    mantık (TR-normalizasyon, satır birleştirme, küçük-ROI ikinci şans) BİREBİR korunur.
    """

    def __init__(self, cfg):
        from fast_plate_ocr import LicensePlateRecognizer

        from roadguard.device import cuda_is_usable

        model = str(cfg.get("plate.fastplate_model", "global-plates-mobile-vit-v2-model"))
        # fast-plate-ocr cihazı: cuda | cpu | auto. CUDA doğrulanmazsa cpu
        # (onnxruntime varsayılan derlemesi MPS sağlamaz → cuda olmayan makinede cpu).
        device = "cuda" if (bool(cfg.get("plate.ocr_gpu", True)) and cuda_is_usable()) else "cpu"
        rec = LicensePlateRecognizer(model, device=device)
        self.reader = _FastPlateAdapter(rec)
        self.max_side = int(cfg.get("plate.ocr_max_side", 1280))
        self.enhance_below = int(cfg.get("plate.ocr_enhance_below_px", 64))
        log.info("fast-plate-ocr yüklendi (model=%s, device=%s)", model, device)


class MockOCR(OCREngine):
    """Araç rengi (BGR) → senaryo plakası. Track başına kararlı."""

    _PLATES = [
        ((90, 200, 255), "34ABC123"),
        ((120, 255, 120), "06FY4571"),
        ((200, 150, 255), "35TR07"),
    ]

    def __init__(self, cfg):
        # max_dist config'ten okunabilir (eskiden hardcoded; cfg yok sayılıyordu).
        # cfg None ya da anahtar yoksa MEVCUT varsayılan 180.0 korunur (davranış aynı).
        self.max_dist = float(cfg.get("plate.mock_max_dist", 180.0)) if cfg is not None else 180.0

    def read(self, plate_roi, vehicle_crop=None) -> tuple[str | None, float]:
        if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
            return None, 0.0
        mean = vehicle_crop.reshape(-1, vehicle_crop.shape[-1])[:, :3].mean(axis=0)
        best_plate, best_d = None, 1e9
        for color, plate in self._PLATES:
            d = float(np.linalg.norm(mean - np.array(color, dtype=float)))
            if d < best_d:
                best_d, best_plate = d, plate
        if best_plate is None or best_d > self.max_dist:
            return None, 0.0
        return best_plate, round(max(0.6, 1.0 - best_d / 300.0), 2)


def _easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except ImportError:
        return False  # normal: bağımlılık kurulu değil
    except Exception as exc:  # noqa: BLE001 — KURULU ama bozuk (yarım kurulum/çakışma)
        # ai_mode=real iken sessizce MockOCR'a düşüp uydurma plaka üretmesini önlemek
        # için kök-neden GÖRÜNÜR olsun (yoksa 'EasyOCR yok' diye yanlış atfedilirdi).
        log.warning("EasyOCR kurulu ama import edilemiyor (bozuk kurulum?): %s", exc)
        return False


def _paddleocr_available() -> bool:
    try:
        import paddleocr  # noqa: F401

        return True
    except Exception:
        return False


def _fastplate_available() -> bool:
    try:
        import fast_plate_ocr  # noqa: F401

        return True
    except Exception:
        return False


def _build_real_ocr(cfg) -> OCREngine:
    """Gerçek OCR motorunu config `plate.ocr_engine` ile seç.

    `paddleocr` seçiliyse PaddleOCR'ı, `fastplate` seçiliyse fast-plate-ocr'ı
    (plakaya-özel ONNX) sarmalar; kurulu değilse LOGLU olarak EasyOCR'a düşer.
    Varsayılan `easyocr` (MEVCUT yol birebir korunur).
    """
    engine = str(cfg.get("plate.ocr_engine", "easyocr")).lower()
    if engine == "paddleocr":
        if _paddleocr_available():
            return PaddleOCRReader(cfg)
        log.warning("plate.ocr_engine=paddleocr ama paddleocr kurulu değil → EasyOCR'a düşülüyor")
    elif engine == "fastplate":
        if _fastplate_available():
            return FastPlateOCRReader(cfg)
        log.warning(
            "plate.ocr_engine=fastplate ama fast-plate-ocr kurulu değil → EasyOCR'a düşülüyor"
        )
    return RealOCR(cfg)


def build_agreement_ocr(cfg, primary: OCREngine | None = None) -> OCREngine | None:
    """Gri-bölge erken-okuma MUTABAKATI için İKİNCİ bağımsız OCR motoru kur.

    Gri-bölge (lp_h < lp_vote_min_px) okuması oy havuzuna girmek için fastplate +
    ikinci motorun AYNI format-geçerli plakayı okuması gerekir (reader._early_read).
    İkinci motor `plate.early_read.agreement_engine` ile seçilir; verilmemişse birincil
    motordan FARKLI ilk kullanılabilir gerçek motor seçilir (easyocr↔fastplate). Hiç
    bağımsız ikinci motor kurulamazsa None (reader o zaman yalnız çok-yüksek-conf eşiğine
    düşer — yanlış-onay yine imkânsız). Mock/torch yok ortamında None (gri-bölge erken-
    okuma gerçek-motor özelliği; mock akışı etkilenmez)."""
    mode = str(cfg.get("runtime.ai_mode", "auto")).lower()
    if mode == "mock" or (mode == "auto" and is_synthetic_source(cfg)):
        return None
    if not _easyocr_available():
        return None
    req = cfg.get("plate.early_read.agreement_engine", None)
    primary_name = str(cfg.get("plate.ocr_engine", "easyocr")).lower()
    if req:
        engine = str(req).lower()
    else:
        # birincilden FARKLI ilk kullanılabilir motor (bağımsız kanıt)
        engine = "easyocr" if primary_name != "easyocr" else "fastplate"
    try:
        if engine == "paddleocr" and _paddleocr_available():
            return PaddleOCRReader(cfg)
        if engine == "fastplate" and _fastplate_available():
            return FastPlateOCRReader(cfg)
        if engine == "easyocr":
            return RealOCR(cfg)
    except (
        Exception
    ) as exc:  # noqa: BLE001 — ikinci motor kurulamazsa erken-okuma conf-eşiğine düşer
        log.warning(
            "Mutabakat OCR motoru (%s) kurulamadı: %s — conf-eşiğine düşülüyor", engine, exc
        )
        return None
    log.warning("Mutabakat OCR motoru (%s) kullanılamıyor — conf-eşiğine düşülüyor", engine)
    return None


def _any_real_ocr_available(cfg) -> bool:
    """Yapılandırılmış motor VEYA herhangi bir gerçek OCR motoru kullanılabilir mi.

    Eski sürüm yalnız EasyOCR'a bakıyordu → varsayılan \tcode{fastplate} kurulu ama EasyOCR
    yokken sistem sessizce MockOCR'a (sahte plaka) düşüyordu. Artık config motoru
    (fastplate/paddleocr) ve evrensel fallback (easyocr) ayrı ayrı denetlenir.
    """
    engine = str(cfg.get("plate.ocr_engine", "easyocr")).lower()
    if engine == "fastplate" and _fastplate_available():
        return True
    if engine == "paddleocr" and _paddleocr_available():
        return True
    return _easyocr_available() or _fastplate_available() or _paddleocr_available()


def build_ocr(cfg) -> OCREngine:
    mode = str(cfg.get("runtime.ai_mode", "auto")).lower()
    # auto + gömülü sentetik örnek → mock OCR (renk→plaka, hızlı ve deterministik;
    # detector/driver de bu kaynakta mock'a düştüğü için tüm hat tutarlı kalır)
    if mode == "auto" and is_synthetic_source(cfg):
        return MockOCR(cfg)
    if mode != "mock" and _any_real_ocr_available(cfg):
        return _build_real_ocr(cfg)
    if mode == "real":
        # K-004: real modda gerçek OCR yoksa MockOCR SAHTE plaka üretir → açık hata ver,
        # sessizce uydurma çıktıya düşme.
        raise RuntimeError(
            "ai_mode=real ama hiçbir gerçek OCR motoru (fastplate/easyocr/paddleocr) "
            "kullanılamıyor; mock OCR sahte plaka üretmesin diye durduruldu "
            "(pip install fast-plate-ocr veya easyocr)."
        )
    return MockOCR(cfg)
