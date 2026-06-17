"""Plaka okuma: sweet spot + voting buffer + OCR + Türk plaka regex.

Akış (plan.md §6.5):
  araç sweet-spot'a girene kadar OCR pasif → girince ardışık okuma + voting buffer
  → konsensüs: kalıcı yaz + OCR kapat (erken çıkış) + PLATE_CONFIRMED
  → ret: PLATE_REJECTED + QoD kalite tetiği + yeniden okuma döngüsü
Yetersiz piksel (min_pixel_height altı) → QoD kalite tetiği.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from aura.optional.loader import get_optional
from aura.plate.dewarp import dewarp_plate
from aura.plate.enhance import enhance_plate
from aura.plate.normalize import PlateVotePool
from aura.plate.ocr import build_ocr
from aura.schema import BBox, PlateState

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("aura.plate.reader")


class PlateReader:
    def __init__(self, cfg, qod=None, ocr=None):
        self.cfg = cfg
        ss = cfg.get("plate.sweet_spot", {}) or {}
        self.sweet_spot = {
            "x1": float(ss.get("x1", 0.0)),
            "y1": float(ss.get("y1", 0.0)),
            "x2": float(ss.get("x2", 1.0)),
            "y2": float(ss.get("y2", 1.0)),
        }
        self.buffer_size = int(cfg.get("plate.voting_buffer_size", 7))
        self.consensus_ratio = float(cfg.get("plate.consensus_ratio", 0.6))
        self.regex = re.compile(cfg.get("plate.regex", r"^\d{2}[A-Z]{1,3}\d{2,4}$"))
        self.min_pixel_height = int(cfg.get("plate.min_pixel_height", 16))
        # Format-öncelikli kalıcı oy havuzu parametreleri (plate.voting.*)
        pv = cfg.get("plate.voting", {}) or {}
        self._pool_kwargs = dict(
            min_weight=float(pv.get("min_weight", 2.0)),
            margin_weight=float(pv.get("margin_weight", 1.5)),
            ratio=self.consensus_ratio,
            fix1_weight=float(pv.get("fix1_weight", 0.45)),
            fix2_weight=float(pv.get("fix2_weight", 0.20)),
            substring_weight=float(pv.get("substring_weight", 0.25)),
            char_consensus=bool(pv.get("char_consensus", True)),
            char_margin=float(pv.get("char_margin", 1.5)),
            confirm_min_char_margin=(
                float(pv["confirm_min_char_margin"])
                if pv.get("confirm_min_char_margin") is not None
                else None
            ),
            confirm_peak_weight=float(pv.get("confirm_peak_weight", 0.30)),
        )
        self.ocr = ocr if ocr is not None else build_ocr(cfg)
        # OCR-öncesi görüntü hazırlama (WP-A1): dewarp (fronto-paralel perspektif
        # düzeltme) + enhance (CLAHE+gamma+unsharp). LP kırpığına, OCR'a girmeden
        # HEMEN ÖNCE uygulanır. Karanlık/açılı otoparkta il-kodu misread'ini
        # (3→0/2) azaltır. Flag'ler config'ten. VARSAYILAN KAPALI (güvenli-OFF):
        # commit 8fef7f6 ölçtü ki bu footage'da T→I kayması 34TC8532→34IC8532
        # YANLIŞ onayına yol açıyor — eksik-config kullanıcısı bu regresyona
        # düşmesin diye kod-içi varsayılan da False. (config/default.yaml zaten
        # false; iki kaynak hizalı.) Saf cv2/numpy, model gerektirmez; dışarıdan
        # OCR enjekte edilmiş testlerde de çalışır.
        self._dewarp_enabled = bool(cfg.get("plate.dewarp.enabled", False))
        self._enhance_enabled = bool(cfg.get("plate.enhance.enabled", False))
        self.qod = qod
        self.sr = get_optional(cfg, "super_resolution")  # §8.2 (lazy; kapalıysa None)
        # Sıkı plaka kırpma (opsiyonel LP dedektörü): araç-altı GENİŞ crop yerine
        # plakanın kendisi kırpılıp OCR'a verilir — karakter doğruluğu belirgin
        # artar (v1'in kanıtlanmış yolu). Ağırlık yoksa/ultralytics yoksa sessizce
        # (tek log) eski geniş-crop davranışına düşer. Dışarıdan OCR enjekte
        # edilmişse (testler) bu aşama atlanır.
        lp = cfg.get("plate.lp_detector", {}) or {}
        self._lp_enabled = bool(lp.get("enabled", True)) and ocr is None
        # Düşük güvenli okumada ikinci (CLAHE+2x) varyant denemesi: oy havuzuna ek
        # bağımsız kanıt — tek-atış karakter hataları varyantlar arasında sönümlenir.
        self._second_variant = bool(cfg.get("plate.ocr_second_variant", True)) and ocr is None
        self._second_variant_below = float(cfg.get("plate.ocr_second_variant_below", 0.5))
        self._lp_conf = float(lp.get("conf", 0.30))
        self._lp_imgsz = int(lp.get("imgsz", 640))
        self._lp_pad = float(lp.get("pad_ratio", 0.08))
        self._lp_path = str(lp.get("path", "weights/lp_yolo11n.pt"))
        self._lp_model = None  # lazy yüklenir
        self._lp_failed = False
        # Boyut-farkında kanıt politikası (gerçek video dersi: uzak karelerin
        # sistematik misread'leri sayıca üstünlük kurup konsensüsü kilitliyordu):
        #   lp_h <  vote_min_px  → okuma OYLAMAYA GİRMEZ (+ erken QoD tetiği)
        #   lp_h <  qod_below_px → görüldüğü AN 'plate_too_small' kalite tetiği
        #                          (consensus_fail beklenmez — havuz zehirlenmeden)
        #   ağırlık = clamp(lp_h / size_full_px, size_floor, 1.0)
        self.lp_vote_min_px = int(cfg.get("plate.lp_vote_min_px", 13))
        self.lp_qod_below_px = int(cfg.get("plate.lp_qod_below_px", 26))
        # Keskinlik-farkında kanıt ağırlığı (Aday-1): LP kırpığının Laplacian
        # varyansı size_w'ye çarpan olur (bulanık uzak kareyi kısar). Saf cv2,
        # install gerektirmez; varsayılan KAPALI; enjekte-OCR'da atlanır.
        sw = cfg.get("plate.sharpness_weight", {}) or {}
        self._sharp_enabled = bool(sw.get("enabled", False)) and ocr is None
        self._sharp_var_full = float(sw.get("var_full", 120.0))
        self._sharp_floor = float(sw.get("floor", 0.25))
        # LP-kırpık süper-çözünürlük (TEKNİK-2): OCR-öncesi küçük kırpığı Lanczos
        # ile büyüt + unsharp. Saf cv2 (dnn_superres/contrib gerektirmez). Yalnız
        # kırpık yüksekliği < min_h_px iken uygulanır; varsayılan KAPALI.
        cu = cfg.get("plate.crop_upscale", {}) or {}
        self._cu_enabled = bool(cu.get("enabled", False))
        self._cu_min_h_px = int(cu.get("min_h_px", 40))
        self._cu_scale = float(cu.get("scale", 3.0))
        self._cu_unsharp = float(cu.get("unsharp_amount", 0.6))
        self._size_full_px = float(pv.get("size_full_px", 40))
        self._size_floor = float(pv.get("size_floor", 0.15))
        self._no_lp_weight = float(pv.get("no_lp_weight", 0.5))
        self._state: dict[int, PlateState] = {}
        self._pools: dict[int, PlateVotePool] = {}
        # Son karede LP dedektörünün bulduğu plaka kutusu (FRAME koordinatında) — hız
        # oto-kalibrasyonunun en kesin ppm kaynağı (520 mm referans). Her update()
        # başında sıfırlanır; pipeline bunu hemen ardından speed.update'e geçirir.
        self.last_plate_bbox: BBox | None = None
        self._reads_since_eval: dict[int, int] = {}

    # --- sıkı plaka kırpma --------------------------------------------------- #
    def _lp_crop(
        self, plate_roi: np.ndarray | None
    ) -> tuple[np.ndarray | None, int | None, tuple[int, int, int, int] | None]:
        """(kırpık, plaka_yüksekliği_px | None, plaka_kutusu | None) döndür.

        Yükseklik upscale ÖNCESİ gerçek piksel yüksekliğidir — okumanın kanıt
        ağırlığı ve QoD 'plate_too_small' tetiği bunun üzerinden hesaplanır.
        Plaka_kutusu = sıkı (pad'siz) (x1, y1, x2, y2), plate_roi-yerel koordinatta;
        hız oto-kalibrasyonu (observe_plate) frame'e çevirip kullanır.
        LP dedektörü kapalı/başarısız/tespitsiz ise None (kaynak kalitesi bilinmiyor).
        """
        if plate_roi is None or not self._lp_enabled or self._lp_failed:
            return plate_roi, None, None
        if self._lp_model is None:
            try:
                from ultralytics import YOLO

                from aura.config import resolve_repo_path
                from aura.device import resolve_device

                p = resolve_repo_path(self._lp_path)
                if not p.exists():
                    raise FileNotFoundError(p)
                self._lp_model = YOLO(str(p))
                self._lp_device = resolve_device(self.cfg.get("runtime.device", "auto"))
                log.info("LP dedektörü yüklendi: %s", p)
            except Exception as e:  # noqa: BLE001 - ağırlık/ultralytics yok → geniş crop
                log.warning("LP dedektörü kullanılamıyor (%s) — geniş crop ile devam", e)
                self._lp_failed = True
                return plate_roi, None, None
        try:
            r = self._lp_model.predict(
                plate_roi,
                conf=self._lp_conf,
                imgsz=self._lp_imgsz,
                device=self._lp_device,
                verbose=False,
            )[0]
        except Exception as e:  # noqa: BLE001 - tek kare hatası tüm hattı durdurmasın
            log.debug("LP tahmini başarısız: %s", e)
            return plate_roi, None, None
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return plate_roi, None, None
        best = max(boxes, key=lambda b: float(b.conf.item()))
        x1, y1, x2, y2 = (int(v) for v in best.xyxy[0].tolist())
        lp_h = max(0, y2 - y1)  # gerçek plaka yüksekliği (upscale öncesi)
        lp_box = (x1, y1, x2, y2)  # sıkı kutu (plate_roi-yerel) — oto-kalibrasyon için
        pad = int(self._lp_pad * max(x2 - x1, y2 - y1))
        h, w = plate_roi.shape[:2]
        crop = plate_roi[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)]
        if crop.size == 0:
            return plate_roi, None, None
        if crop.shape[0] < 48:  # küçük plaka: OCR öncesi 2x büyüt
            import cv2

            crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        return crop, lp_h, lp_box

    def _sharpness_factor(self, img: np.ndarray | None) -> float:
        """LP kırpığının keskinliğinden (Laplacian varyansı) [floor..1] çarpan üret.

        Bulanık (uzak/düşük-odak) kırpık küçük çarpan → kanıt değeri kısılır; net
        kırpık 1.0. var >= var_full → 1.0; arası lineer; en altta floor (oy hakkı
        tamamen sıfırlanmaz). Saf cv2/numpy, model gerektirmez. Hesaplanamazsa 1.0
        (güvenli kimlik — keskinlik bilinmiyorsa ağırlık kısılmaz)."""
        if img is None or getattr(img, "size", 0) == 0:
            return 1.0
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        full = max(self._sharp_var_full, 1e-6)
        return max(self._sharp_floor, min(1.0, var / full))

    def _crop_upscale(self, img: np.ndarray | None) -> np.ndarray | None:
        """Küçük LP kırpığını Lanczos ile büyüt + hafif unsharp maske (TEKNİK-2).

        Yalnız kırpık yüksekliği < min_h_px ise uygulanır (büyük/net kırpık
        değişmeden döner — gereksiz yumuşatma/maliyet yok). Lanczos (INTER_LANCZOS4)
        küçük metin kenarlarını CUBIC'ten daha iyi korur; ardından Gaussian-tabanlı
        unsharp maske karakter konturlarını keskinleştirir. Saf cv2; dnn_superres
        ya da opencv-contrib GEREKTİRMEZ. Hata/boş girişte güvenli kimlik (img)."""
        if img is None or getattr(img, "size", 0) == 0:
            return img
        if img.shape[0] >= self._cu_min_h_px:
            return img  # büyük/net kırpık — dokunma
        import cv2

        up = cv2.resize(
            img, None, fx=self._cu_scale, fy=self._cu_scale, interpolation=cv2.INTER_LANCZOS4
        )
        if self._cu_unsharp > 0:
            blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0)
            up = cv2.addWeighted(up, 1.0 + self._cu_unsharp, blur, -self._cu_unsharp, 0)
        return up

    @staticmethod
    def _enhance(img: np.ndarray) -> np.ndarray:
        """CLAHE + 2x büyütme varyantı (karanlık/küçük plaka için ikinci şans)."""
        import cv2

        up = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
        lab_l, lab_a, lab_b = cv2.split(lab)
        lab_l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab_l)
        return cv2.cvtColor(cv2.merge((lab_l, lab_a, lab_b)), cv2.COLOR_LAB2BGR)

    # --- sweet spot -------------------------------------------------------- #
    def in_sweet_spot(self, bbox: BBox, frame_shape: tuple[int, ...]) -> bool:
        h, w = frame_shape[0], frame_shape[1]
        cx, cy = bbox.center
        s = self.sweet_spot
        return (s["x1"] * w <= cx <= s["x2"] * w) and (s["y1"] * h <= cy <= s["y2"] * h)

    # --- ana giriş --------------------------------------------------------- #
    def update(
        self,
        track_id: int,
        plate_roi: np.ndarray | None,
        vehicle_bbox: BBox,
        frame_shape: tuple[int, ...],
        frame: np.ndarray | None = None,
    ) -> PlateState:
        self.last_plate_bbox = None  # bu kare için sıfırla (önceki track'ten sızmasın)
        st = self._state.setdefault(track_id, PlateState())
        if st.status == "confirmed":
            return st  # erken çıkış (OCR kapalı)
        if st.status == "rejected":
            st.status = "pending"  # yeniden okuma döngüsü

        # Sweet spot: araç netlik bölgesine girene kadar OCR pasif
        if not self.in_sweet_spot(vehicle_bbox, frame_shape):
            return st

        # §8.2 süper çözünürlük (etkinse) — küçük plakaları OCR öncesi büyüt
        if self.sr is not None and plate_roi is not None:
            plate_roi = self.sr.enhance(plate_roi)

        # Yetersiz piksel → kalite tetiği (OCR'a girmeden)
        if plate_roi is not None and plate_roi.shape[0] < self.min_pixel_height:
            if self.qod:
                self.qod.request_quality(track_id, reason="low_pixel")
            return st

        vehicle_crop = None
        if frame is not None:
            x1, y1 = max(0, int(vehicle_bbox.x1)), max(0, int(vehicle_bbox.y1))
            x2, y2 = int(vehicle_bbox.x2), int(vehicle_bbox.y2)
            vehicle_crop = frame[y1:y2, x1:x2]

        crop, lp_h, lp_box = self._lp_crop(plate_roi)
        # Hız oto-kalibrasyonu için plaka kutusunu FRAME koordinatına çevir. plate_roi
        # araç bbox'ının alt diliminden (frame[split:y2, x1:x2]) DOĞRUDAN kesildiğinden
        # offset araç bbox'ı + crop yüksekliğinden türetilir (cabin_ratio gerekmez).
        # Yalnız SR kapalıyken: SR plate_roi'yi ölçekler → piksel genişliği/ppm bozulur.
        if lp_box is not None and self.sr is None and plate_roi is not None:
            ox = max(0, int(vehicle_bbox.x1))
            oy = min(int(frame_shape[0]), int(vehicle_bbox.y2)) - plate_roi.shape[0]
            lx1, ly1, lx2, ly2 = lp_box
            self.last_plate_bbox = BBox(
                x1=float(ox + lx1),
                y1=float(oy + ly1),
                x2=float(ox + lx2),
                y2=float(oy + ly2),
                conf=1.0,
                cls="plate",
            )
        size_w = 1.0  # kaynak-kalitesi çarpanı (LP yüksekliğinden)
        if lp_h is not None:
            if self.qod and lp_h < self.lp_qod_below_px:
                # ERKEN kalite tetiği: plaka görüldü ama OCR için küçük — QoD
                # consensus_fail BEKLEMEDEN devreye girer (havuz zehirlenmeden;
                # gerçek video ölçümü: eski akışta ilk tetikten önce 12-14 çöp oy).
                self.qod.request_quality(track_id, reason="plate_too_small")
            if lp_h < self.lp_vote_min_px:
                return st  # kanıt değeri yok: çöp okuma havuza oy yazamaz
            size_w = max(self._size_floor, min(1.0, lp_h / max(self._size_full_px, 1.0)))
        elif self._lp_enabled and not self._lp_failed and self._lp_model is not None:
            # LP dedektörü çalışıyor ama plakayı bulamadı → geniş-crop okuması
            # düşük güvenilirlik sınıfıdır (kanıt tamamen atılmaz, ağırlığı kısılır).
            size_w = self._no_lp_weight
        # KESKİNLİK-FARKINDA kanıt ağırlığı (Aday-1): bulanık/uzak kare (düşük
        # Laplacian varyansı) size_w'yi [floor..1] çarpanla kısar → net/yakın kare
        # konsensüste baskın olur. Yalnız flag açıkken; keskinlik OCR-öncesi
        # dewarp/enhance dönüşümlerinden ETKİLENMESİN diye ham LP kırpığı üzerinden
        # ÖLÇÜLÜR. Hesaplanamazsa 1.0 (kimlik — ağırlık kısılmaz).
        if self._sharp_enabled:
            size_w *= self._sharpness_factor(crop)
        # OCR-öncesi hazırlama: önce fronto-paralel dewarp (açılı plaka → düz),
        # sonra enhance (CLAHE+gamma+unsharp). Köşe bulunamazsa dewarp kimlik
        # döner; ikisi de saf cv2/numpy ve şekil-korur. last_plate_bbox/size_w
        # YUKARIDA lp_box'tan zaten türetildi → buradaki dönüşüm onları ETKİLEMEZ.
        # İkinci-şans (CLAHE+2x) varyantı için dewarp-sonrası ama enhance-ÖNCESİ
        # görüntü ayrıca tutulur: enhance + reader._enhance üst üste binerse
        # (CLAHE-üzerine-CLAHE + çift kontrast) bazı karelerde OCR'ı KÖTÜLEŞTİRİR.
        # İlk geçiş enhance'li okur; ikinci-şans HAM (enhance'siz) crop'tan türer.
        # LP-kırpık süper-çözünürlük (TEKNİK-2): OCR-öncesi, küçük kırpığı Lanczos
        # büyüt + unsharp. lp_h/size_w/last_plate_bbox YUKARIDA upscale-öncesi ham
        # kırpıktan türetildi → bu dönüşüm kanıt ağırlığını/QoD'yi ETKİLEMEZ. Yalnız
        # küçük kırpığa uygulanır (helper içinde min_h_px kapısı). İkinci-şans
        # varyantı da büyütülmüş kırpıktan türesin diye dewarp/enhance'ten ÖNCE.
        if self._cu_enabled and crop is not None and getattr(crop, "size", 0):
            crop = self._crop_upscale(crop)
        raw_crop = crop  # dewarp sonrası, enhance öncesi — ikinci-şans referansı
        if crop is not None and getattr(crop, "size", 0):
            if self._dewarp_enabled:
                crop = dewarp_plate(crop)
                raw_crop = crop  # dewarp ikinci-şansa da uygulanır (yalnız enhance hariç)
            if self._enhance_enabled:
                crop = enhance_plate(crop, self.cfg)
        text, conf = self.ocr.read(crop, vehicle_crop)
        pool = self._pools.setdefault(track_id, PlateVotePool(**self._pool_kwargs))
        pool.add(text, conf, weight=size_w)
        if (
            self._second_variant
            and raw_crop is not None
            and getattr(raw_crop, "size", 0)
            and (text is None or conf < self._second_variant_below)
        ):
            # ÇİFT-ENHANCE önleme: kendi CLAHE+2x varyantını enhance'li crop yerine
            # HAM (dewarp sonrası, enhance öncesi) crop'a uygula → bağımsız ikinci kanıt.
            t2, c2 = self.ocr.read(self._enhance(raw_crop), vehicle_crop)
            pool.add(t2, c2, weight=size_w)
            if t2 and not text:
                text = t2  # okuma sayacı için (kanıt geldi)
        st.votes = pool.counts()
        # Konsensüs olmasa bile en güçlü aday kanıt izi olarak raporlanır
        # (şartname 4.5: kanıtlanamayan hedef puanlanmaz — kısmi okuma da kanıttır).
        st.partial = pool.best_partial()
        if text:
            self._reads_since_eval[track_id] = self._reads_since_eval.get(track_id, 0) + 1

        value, frac = pool.consensus()
        if value is not None:
            # Oy havuzu kazananı zaten TR formatına normalize edilmiş üretir;
            # regex yine de son kapı olarak uygulanır (config'ten daraltılabilir).
            if self.regex.match(value):
                st.value = value
                st.confidence = frac
                st.status = "confirmed"
                st.ocr_disabled = True
                if self.qod:
                    # Plaka çözüldü → kalite oturumu amacına ulaştı, HEMEN bırak
                    # (eski akış: onaydan ~31 kare sonra zaman aşımıyla kapanıyordu).
                    self.qod.release_quality(track_id)
            return st
        # Konsensüs yok: her `buffer_size` okumada bir 'rejected' döngüsü işlet —
        # event + QoD kalite tetiği üretir ama OYLAR SIFIRLANMAZ (kalıcı birikim,
        # v1 PlateTracker dersi: '97 oy vs 1' kararlılığı ancak birikimle oluşur).
        if self._reads_since_eval.get(track_id, 0) >= self.buffer_size:
            self._reads_since_eval[track_id] = 0
            st.status = "rejected"
            st.confidence = frac
            if self.qod:
                self.qod.request_quality(track_id, reason="consensus_fail")
        return st

    def get(self, track_id: int) -> PlateState | None:
        return self._state.get(track_id)
