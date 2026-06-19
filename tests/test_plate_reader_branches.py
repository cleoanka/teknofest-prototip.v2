"""Reader dal/kenar-durum kapsamı: LP-kırpık, boyut-farkında ağırlık, FRAME-koord
dönüşümü, min_pixel kapısı, keskinlik/upscale yardımcıları, QoD release/reason'lar.

Model GEREKTİRMEZ: LP dedektörü sahte bir model nesnesiyle enjekte edilir (gerçek
YOLO/ultralytics yüklenmez); OCR sahte; tüm cv2/numpy saf-CPU çalışır. Bu dosya
inceleme bulgularındaki test boşluklarını (reader._lp_crop / size_w / FRAME-koord /
min_pixel / sharpness / crop_upscale / release_quality) kapatır.
"""

from __future__ import annotations

import numpy as np

from aura.plate.reader import PlateReader
from aura.schema import BBox

FRAME_SHAPE = (360, 640, 3)


# --- sahteler ------------------------------------------------------------- #
class FakeOCR:
    """Sabit (metin, güven) döndürür; çağrılan görüntüleri kaydeder."""

    def __init__(self, value=("34TC8532", 0.9)):
        self.value = value
        self.calls = []

    def read(self, plate_roi, vehicle_crop=None):
        self.calls.append(None if plate_roi is None else np.asarray(plate_roi).copy())
        return self.value


class FakeQoD:
    def __init__(self):
        self.calls = []
        self.released = []

    def request_quality(self, track_id, reason):
        self.calls.append((track_id, reason))

    def release_quality(self, track_id):
        self.released.append(track_id)


class _FakeBox:
    def __init__(self, xyxy, conf):
        self._xyxy = xyxy
        self.conf = _Scalar(conf)

    @property
    def xyxy(self):
        # ultralytics .xyxy[0].tolist() sözleşmesini taklit et
        return [_Listable(self._xyxy)]


class _Scalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _Listable:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return list(self._vals)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeLP:
    """ultralytics YOLO.predict sözleşmesini taklit eder: predict(...) -> [result].

    Verilen sıkı kutuyu (plate_roi-yerel) döndürür → _lp_crop'un kırpma/yükseklik/
    kutu mantığı gerçek model olmadan koşar.
    """

    def __init__(self, box_xyxy, conf=0.9):
        self.box_xyxy = box_xyxy
        self.conf = conf
        self.last_kwargs = None

    def predict(self, img, **kw):
        self.last_kwargs = kw
        return [_FakeResult([_FakeBox(self.box_xyxy, self.conf)])]


class FakeLPEmpty:
    """Plaka bulamayan LP dedektörü (boxes boş)."""

    def predict(self, img, **kw):
        return [_FakeResult([])]


def _center_bbox():
    return BBox(x1=300, y1=210, x2=340, y2=260, conf=0.9, cls="car")


def _enable_lp(reader, fake_model):
    """Enjekte-OCR reader'ında LP yolunu sahte modelle elle aç (ocr is None şartını aş)."""
    reader._lp_enabled = True
    reader._lp_failed = False
    reader._lp_model = fake_model
    reader._lp_device = "cpu"


# --- _lp_crop: kırpma + yükseklik + sıkı kutu ---------------------------- #
def test_lp_crop_returns_height_and_box():
    r = PlateReader_with_ocr()
    fake = FakeLP(box_xyxy=(10, 20, 70, 50))  # yükseklik 30 (>=48 değil → 2x büyütme)
    _enable_lp(r, fake)
    roi = np.full((80, 120, 3), 100, np.uint8)
    crop, lp_h, lp_box = r._lp_crop(roi)
    assert lp_h == 30  # gerçek plaka yüksekliği (upscale öncesi)
    assert lp_box == (10, 20, 70, 50)  # sıkı (pad'siz) kutu
    # crop yüksekliği < 48 → 2x büyütülmüş olmalı
    assert crop.shape[0] > 30


def test_lp_crop_disabled_returns_passthrough():
    r = PlateReader_with_ocr()
    # _lp_enabled False (enjekte-OCR) → kırpık aynen, yükseklik/kutu None
    roi = np.full((40, 120, 3), 100, np.uint8)
    crop, lp_h, lp_box = r._lp_crop(roi)
    assert crop is roi and lp_h is None and lp_box is None


def test_lp_crop_no_detection_returns_none_height():
    r = PlateReader_with_ocr()
    _enable_lp(r, FakeLPEmpty())
    roi = np.full((60, 120, 3), 100, np.uint8)
    crop, lp_h, lp_box = r._lp_crop(roi)
    assert crop is roi and lp_h is None and lp_box is None


def test_lp_crop_passes_conf_imgsz_to_model():
    r = PlateReader_with_ocr()
    fake = FakeLP(box_xyxy=(5, 5, 60, 40))
    _enable_lp(r, fake)
    r._lp_conf = 0.42
    r._lp_imgsz = 512
    r._lp_crop(np.full((50, 100, 3), 80, np.uint8))
    assert fake.last_kwargs["conf"] == 0.42
    assert fake.last_kwargs["imgsz"] == 512


# --- size_w (boyut-farkında kanıt ağırlığı) + QoD reason'ları ------------- #
def _vote_weight_for(reader, roi, fake_lp):
    """update() çalıştır, pool'a yazılan tek okumanın etkin ağırlığını oku."""
    _enable_lp(reader, fake_lp)
    reader.update(1, roi, _center_bbox(), FRAME_SHAPE)
    pool = reader._pools.get(1)
    return pool.raw_reads if pool else []


def test_lp_below_vote_min_px_writes_no_vote():
    # lp_h < lp_vote_min_px (13) → çöp okuma havuza yazılmaz (kanıt değeri yok)
    q = FakeQoD()
    r = PlateReader_with_ocr(qod=q)
    fake = FakeLP(box_xyxy=(0, 0, 40, 10))  # yükseklik 10 < 13
    reads = _vote_weight_for(r, np.full((60, 120, 3), 90, np.uint8), fake)
    assert reads == []  # oy yazılmadı


def test_lp_below_qod_below_px_triggers_plate_too_small():
    # lp_h < lp_qod_below_px → 'plate_too_small' ERKEN tetik (consensus beklemeden).
    # Eşikler config'ten BAĞIMSIZ açıkça verilir (test mekanizmayı doğrular, config'in
    # canlı/3-video için seçilmiş gerçek değerini değil): vote_min=13 < lp_h=20 < qod_below=26.
    q = FakeQoD()
    r = PlateReader_with_ocr(qod=q)
    r.lp_vote_min_px = 13
    r.lp_qod_below_px = 26
    fake = FakeLP(
        box_xyxy=(0, 0, 40, 20)
    )  # yükseklik 20: <qod_below(26) ama >=vote_min(13) → oy yazılır
    _enable_lp(r, fake)
    r.update(1, np.full((60, 120, 3), 90, np.uint8), _center_bbox(), FRAME_SHAPE)
    assert any(reason == "plate_too_small" for _, reason in q.calls)
    # 20 >= vote_min(13) → oy yine de yazıldı
    assert r._pools[1].raw_reads


def test_size_w_clamped_between_floor_and_one():
    # size_w = clamp(lp_h / size_full_px, size_floor, 1.0). size_full_px=40, floor=0.15.
    # lp_h=20 → 0.5; conf=1.0 → eff ağırlık 0.5. vote_min config'ten BAĞIMSIZ (mekanizma testi).
    r = PlateReader_with_ocr(ocr_value=("34TC8532", 1.0))
    r.lp_vote_min_px = 13  # lp_h=20 oylanabilsin (size_w hesabı doğrulanacak)
    fake = FakeLP(box_xyxy=(0, 0, 50, 20))
    reads = _vote_weight_for(r, np.full((70, 120, 3), 90, np.uint8), fake)
    assert reads, "oy yazılmalı (20>=vote_min)"
    _, eff = reads[-1]
    assert abs(eff - 0.5) < 1e-6  # 20/40 = 0.5


def test_size_w_floor_clamp_for_small_plate():
    # lp_h küçük (ama >= vote_min): size_w lineer (floor üstü). vote_min config'ten BAĞIMSIZ.
    r = PlateReader_with_ocr(ocr_value=("34TC8532", 1.0))
    r.lp_vote_min_px = 13  # lp_h=14 oylanabilsin
    fake = FakeLP(box_xyxy=(0, 0, 50, 14))  # 14/40=0.35 — floor üstünde, clamp etmez
    reads = _vote_weight_for(r, np.full((70, 120, 3), 90, np.uint8), fake)
    _, eff = reads[-1]
    assert abs(eff - 14 / 40) < 1e-6


def test_no_lp_weight_when_model_runs_but_no_detection():
    # LP modeli çalışıyor ama plaka bulamadı → size_w = no_lp_weight (0.5 vars.)
    r = PlateReader_with_ocr(ocr_value=("34TC8532", 1.0))
    _enable_lp(r, FakeLPEmpty())
    r.update(1, np.full((60, 120, 3), 90, np.uint8), _center_bbox(), FRAME_SHAPE)
    reads = r._pools[1].raw_reads
    assert reads
    _, eff = reads[-1]
    assert abs(eff - r._no_lp_weight) < 1e-6


# --- FRAME-koordinat dönüşümü (hız oto-kalibrasyon ppm kaynağı) ---------- #
def test_last_plate_bbox_frame_coords():
    # last_plate_bbox = vehicle offset + plate_roi-yerel kutu. SR kapalı olmalı.
    r = PlateReader_with_ocr()
    fake = FakeLP(box_xyxy=(10, 5, 60, 25))
    _enable_lp(r, fake)
    bbox = _center_bbox()  # x1=300, y2=260
    roi = np.full((40, 120, 3), 90, np.uint8)  # yükseklik 40
    r.update(1, roi, bbox, FRAME_SHAPE)
    lpb = r.last_plate_bbox
    assert lpb is not None
    ox = 300  # vehicle x1
    oy = 260 - 40  # vehicle y2 - plate_roi yüksekliği = 220
    assert lpb.x1 == ox + 10 and lpb.y1 == oy + 5
    assert lpb.x2 == ox + 60 and lpb.y2 == oy + 25
    assert lpb.cls == "plate"


def test_last_plate_bbox_reset_each_update():
    # Her update başında sıfırlanır (önceki track'ten sızmasın). LP yoksa None kalır.
    r = PlateReader_with_ocr()
    fake = FakeLP(box_xyxy=(10, 5, 60, 25))
    _enable_lp(r, fake)
    r.update(1, np.full((40, 120, 3), 90, np.uint8), _center_bbox(), FRAME_SHAPE)
    assert r.last_plate_bbox is not None
    # LP kapat → bir sonraki update'te bbox None'a sıfırlanmalı
    r._lp_enabled = False
    r.update(1, np.full((40, 120, 3), 90, np.uint8), _center_bbox(), FRAME_SHAPE)
    assert r.last_plate_bbox is None


# --- min_pixel_height gate + low_pixel QoD reason ------------------------- #
def test_min_pixel_height_gate_triggers_low_pixel():
    # plate_roi.shape[0] < min_pixel_height (16) → 'low_pixel', OCR'a girmez.
    q = FakeQoD()
    r = PlateReader_with_ocr(qod=q)
    tiny = np.full((10, 40, 3), 90, np.uint8)  # yükseklik 10 < 16
    st = r.update(1, tiny, _center_bbox(), FRAME_SHAPE)
    assert any(reason == "low_pixel" for _, reason in q.calls)
    assert st.status == "pending"
    assert 1 not in r._pools  # OCR'a hiç girmedi → pool kurulmadı


# --- QoD release_quality (confirmed yolu) -------------------------------- #
def test_qod_release_on_confirm():
    q = FakeQoD()
    r = PlateReader_with_ocr(qod=q, ocr_value=("34TC8532", 0.9))
    bbox = _center_bbox()
    for _ in range(20):
        st = r.update(1, np.full((30, 80, 3), 90, np.uint8), bbox, FRAME_SHAPE)
        if st.status == "confirmed":
            break
    assert st.status == "confirmed"
    assert 1 in q.released  # plaka çözüldü → kalite oturumu HEMEN bırakıldı


# --- _sharpness_factor yardımcısı ---------------------------------------- #
def test_sharpness_factor_identity_on_none_or_empty():
    r = PlateReader_with_ocr()
    assert r._sharpness_factor(None) == 1.0
    assert r._sharpness_factor(np.zeros((0, 0, 3), np.uint8)) == 1.0


def test_sharpness_factor_blurry_below_sharp_edges():
    r = PlateReader_with_ocr()
    r._sharp_var_full = 120.0
    r._sharp_floor = 0.25
    flat = np.full((40, 120, 3), 100, np.uint8)  # düz → Laplacian var ~0 → floor
    assert abs(r._sharpness_factor(flat) - 0.25) < 1e-6
    # keskin kenarlı (yüksek varyans) görüntü → 1.0'a doğru
    sharp = np.zeros((40, 120, 3), np.uint8)
    sharp[:, ::2] = 255  # dikey çizgili yüksek-frekans
    assert r._sharpness_factor(sharp) >= r._sharpness_factor(flat)


def test_sharpness_applied_to_vote_weight():
    # _sharp_enabled açıkken size_w keskinlik çarpanıyla kısılır (düz → floor).
    r = PlateReader_with_ocr(ocr_value=("34TC8532", 1.0))
    r._sharp_enabled = True
    r._sharp_var_full = 120.0
    r._sharp_floor = 0.25
    r.lp_vote_min_px = 13  # lp_h=40 oylanabilsin (vote_min config'ten bağımsız; sharpness testi)
    fake = FakeLP(box_xyxy=(0, 0, 50, 40))  # lp_h=40 → base size_w=1.0
    _enable_lp(r, fake)
    # düz crop → keskinlik floor 0.25 → eff = conf(1.0)*1.0*0.25
    r.update(1, np.full((50, 120, 3), 100, np.uint8), _center_bbox(), FRAME_SHAPE)
    _, eff = r._pools[1].raw_reads[-1]
    assert abs(eff - 0.25) < 1e-6


# --- _crop_upscale yardımcısı -------------------------------------------- #
def test_crop_upscale_identity_when_large():
    r = PlateReader_with_ocr()
    r._cu_min_h_px = 40
    big = np.full((50, 150, 3), 120, np.uint8)  # >= 40 → dokunma
    out = r._crop_upscale(big)
    assert out is big


def test_crop_upscale_enlarges_small():
    r = PlateReader_with_ocr()
    r._cu_min_h_px = 40
    r._cu_scale = 3.0
    r._cu_unsharp = 0.6
    small = np.full((20, 60, 3), 120, np.uint8)  # < 40 → büyüt
    out = r._crop_upscale(small)
    assert out.shape[0] > small.shape[0] and out.shape[1] > small.shape[1]
    assert out.dtype == np.uint8


def test_crop_upscale_identity_on_none_or_empty():
    r = PlateReader_with_ocr()
    assert r._crop_upscale(None) is None
    empty = np.zeros((0, 0, 3), np.uint8)
    assert r._crop_upscale(empty) is empty


# --- ikinci-şans varyantı (CLAHE+2x) düşük güvende ek kanıt -------------- #
class _LowConfThenNoneOCR:
    """İlk okuma None (ikinci-şans tetikler); sonraki çağrı ayrı sonuç döndürür."""

    def __init__(self, seq):
        self.seq = list(seq)
        self.i = 0

    def read(self, plate_roi, vehicle_crop=None):
        v = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return v


def test_second_variant_adds_independent_vote():
    # İlk okuma None (ikinci-şans tetikler); ikinci varyant geçerli okuma döndürür →
    # havuza ek bağımsız oy yazılır.
    from aura.config import load_config

    r = PlateReader(load_config(), ocr=_LowConfThenNoneOCR([(None, 0.0), ("34TC8532", 0.9)]))
    r._second_variant = True  # enjekte-OCR kapatır; davranışı izole etmek için aç
    r.update(1, np.full((48, 120, 3), 90, np.uint8), _center_bbox(), FRAME_SHAPE)
    reads = r._pools[1].raw_reads
    # ilk add(None) yazılmaz (text None) ama ikinci-şans geçerli okumayı yazar
    assert any(t == "34TC8532" for t, _ in reads)


# --- yardımcı kurucu ----------------------------------------------------- #
def PlateReader_with_ocr(cfg=None, qod=None, ocr_value=("34TC8532", 0.9)):
    """Enjekte-OCR'lı reader (model gerektirmez). cfg fixture'ı doğrudan veremediğimiz
    için modül-içi minimal cfg yüklenir."""
    from aura.config import load_config

    c = cfg if cfg is not None else load_config()
    return PlateReader(c, qod=qod, ocr=FakeOCR(ocr_value))
