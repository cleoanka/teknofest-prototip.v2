"""PoseDriverClassifier birim testleri — MODEL GEREKTİRMEZ (saf CPU, sahte YOLO).

pose.py en karmaşık Katman-A dosyasıydı ve hiç doğrudan kapsanmıyordu. Bu testler
ultralytics ağırlığı YÜKLEMEDEN çalışır: classifier `object.__new__` ile kurulur,
yalnız test edilen metodun dokunduğu alanlar elle set edilir, model yerine
ultralytics çıktı ŞEKLİNİ taklit eden sahte nesneler enjekte edilir.

Kapsanan boşluklar (INCELEME test_gaps):
  - _geometry telefon/sigara GÖRELİ ayrımı (d_ear<d_mouth), ağız vekili (nose+0.30fw),
    fw<2.0 çekimserliği, kulak/burun/bilek-yok erken dönüş, tek-kulak fw fallback
  - _driver_crop önbellek (hit yaş artışı, redetect tazeleme, min_gain, kişi-kaybı)
  - telefon→sigara bastırma latch'i (suppress_conf eşiği, suppress_frames geri sayım,
    geo.smoking=False, telefon bayrağının İLERİ TAŞINMAMASI)
  - _object_evidence (canonical eşleme, phone/smoking filtre, conf max-birleştirme)
  - pt() guard kc-kısa IndexError koruması (bug düzeltmesi)
"""

from __future__ import annotations

import numpy as np
import pytest

from aura.driver_state.pose import (
    L_EAR,
    NOSE,
    R_EAR,
    R_WRIST,
    PoseDriverClassifier,
)
from aura.schema import DriverState

# COCO-17 sırasında toplam keypoint sayısı (ultralytics pose çıktısı)
_NKP = 17


# --------------------------------------------------------------------------- #
# Sahte ultralytics çıktı nesneleri (yalnız erişilen alanlar)
# --------------------------------------------------------------------------- #
class _Scalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _Vec:
    def __init__(self, lst):
        self._lst = list(lst)

    def tolist(self):
        return list(self._lst)


class _Box:
    """ultralytics Boxes elemanı: .conf.item(), .cls.item(), .xyxy[0].tolist()."""

    def __init__(self, conf=0.9, cls=0, xyxy=(0.0, 0.0, 10.0, 10.0)):
        self.conf = _Scalar(conf)
        self.cls = _Scalar(cls)
        self.xyxy = [_Vec(xyxy)]


class _Keypoints:
    """ultralytics Keypoints: .xy[i].tolist(), .conf[i].tolist()."""

    def __init__(self, xy_per_person, conf_per_person):
        self.xy = [_Vec(p) for p in xy_per_person]
        self.conf = [_Vec(c) for c in conf_per_person] if conf_per_person is not None else None


class _Result:
    def __init__(self, boxes=None, keypoints=None, names=None):
        self.boxes = boxes
        self.keypoints = keypoints
        if names is not None:
            self.names = names


class _FakeModel:
    """predict() ultralytics gibi RESULTS LİSTESİ döndürür (results[0] = ilk kare).

    `results` bir _Result ise → her çağrı [result] döndürür (sabit).
    `results` bir liste ise (her elemanı bir 'results listesi') → çağrı başına sıralı.
    `results` boş liste [] ise → boş (model çıktısı yok) döndürür.
    """

    def __init__(self, results, names=None):
        self._results = results
        self.names = names or {0: "person"}
        self.calls = 0

    def predict(self, *_a, **_k):
        self.calls += 1
        if isinstance(self._results, list):
            if not self._results:
                return []
            # sıralı senaryo: her eleman bir results-listesi (örn. [[r0],[r1]])
            idx = min(self.calls - 1, len(self._results) - 1)
            return self._results[idx]
        return [self._results]  # tek _Result → ultralytics gibi listeye sar


def _make_kps(points: dict, base_conf: float = 0.9, conf_present: bool = True):
    """points = {keypoint_idx: (x,y)} → tek-kişi _Keypoints. Eksikler (0,0)/conf 0."""
    xy = [(0.0, 0.0)] * _NKP
    conf = [0.0] * _NKP
    for i, (x, y) in points.items():
        xy[i] = (x, y)
        conf[i] = base_conf
    return _Keypoints([xy], [conf] if conf_present else None)


def _pose_clf(model=None, **overrides):
    """PoseDriverClassifier'ı __init__'siz kur; yalnız gerekli alanları set et."""
    clf = object.__new__(PoseDriverClassifier)
    clf.model = model
    clf.obj_model = None
    clf.conf = 0.25
    clf.kp_conf = 0.30
    clf.imgsz = 640
    clf.device = "cpu"
    clf.phone_ear_ratio = 0.40
    clf.smoke_mouth_ratio = 0.60
    # roi_min_side=1 → ölçek 1.0 sabit: sahte kutular ham ROI uzayında kalır
    # (gerçek upscale koordinat geri-eşlemesi _prep_roi_scaled testinde ayrı doğrulanır).
    clf.roi_min_side = 1
    clf.roi_max_upscale = 4.0
    clf.roi_enhance = False  # testlerde parlatma kapalı: koordinatlar bozulmasın
    clf.crop_enabled = False
    clf.crop_pad = 0.10
    clf.crop_redetect = 15
    clf.crop_min_gain = 1.25
    clf.corner_target = (1.0, 1.0)
    clf._crop_cache = {}
    clf.last_crop_box = None
    clf._last_person_seen = False
    clf.obj_enabled = False
    clf.obj_conf = 0.25
    clf.obj_imgsz = 640
    clf.obj_suppress_frames = 25
    clf.obj_suppress_conf = 0.30
    clf._smoke_suppress = {}
    clf._clahe = None
    clf._gamma_lut = None
    # opsiyonel ikinci sigara modeli — VARSAYILAN YOK (graceful-absent / no-op)
    clf.smoking_enabled = False
    clf.smoking_conf = 0.30
    clf.smoking_imgsz = 640
    clf.smoking_model = None
    for k, v in overrides.items():
        setattr(clf, k, v)
    return clf


def _roi():
    return np.zeros((100, 100, 3), np.uint8)


# --------------------------------------------------------------------------- #
# _prep_roi_scaled — ölçek + parlatma (cache'lenmiş CLAHE/gamma perf-opt yolu)
# --------------------------------------------------------------------------- #
def test_prep_roi_scaled_upscales_small_roi():
    """Küçük ROI roi_min_side'a büyütülür; uygulanan ölçek döndürülür."""
    clf = _pose_clf(roi_min_side=80, roi_max_upscale=4.0, roi_enhance=False)
    roi = np.zeros((20, 30, 3), np.uint8)
    out, scale = clf._prep_roi_scaled(roi)
    assert abs(scale - 4.0) < 1e-6  # min(4.0, 80/20)=4.0
    assert out.shape[0] == 80 and out.shape[1] == 120


def test_prep_roi_scaled_zero_size():
    clf = _pose_clf(roi_enhance=False)
    out, scale = clf._prep_roi_scaled(np.zeros((0, 0, 3), np.uint8))
    assert scale == 1.0


def test_prep_roi_enhance_builds_and_reuses_helpers():
    """Parlatma açıkken CLAHE + gamma LUT bir kez kurulur, sonra yeniden kullanılır."""
    clf = _pose_clf(roi_min_side=1, roi_enhance=True)
    roi = np.full((40, 40, 3), 60, np.uint8)
    assert clf._clahe is None and clf._gamma_lut is None
    out1, _ = clf._prep_roi_scaled(roi)
    clahe_ref, lut_ref = clf._clahe, clf._gamma_lut
    assert clahe_ref is not None
    assert lut_ref is not None and lut_ref.shape == (256,)
    out2, _ = clf._prep_roi_scaled(roi)
    assert clf._clahe is clahe_ref  # aynı nesne yeniden kullanıldı (yeniden kurulmadı)
    assert clf._gamma_lut is lut_ref
    # çıktı deterministik (sabit yardımcılar → birebir aynı)
    assert np.array_equal(out1, out2)


# --------------------------------------------------------------------------- #
# _geometry — telefon/sigara göreli ayrımı
# --------------------------------------------------------------------------- #
def test_geometry_phone_when_wrist_near_ear():
    """Bilek kulağa ÇOK yakın VE kulağa ağızdan daha yakın → TELEFON."""
    # iki kulak arası fw=40 (x 30..70, y=50). nose (50,50). bilek kulağın üstünde.
    pts = {
        NOSE: (50, 50),
        L_EAR: (30, 50),
        R_EAR: (70, 50),
        R_WRIST: (72, 50),  # sağ kulağa ~2px, ağıza (50,62) çok uzak
    }
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model)
    ds = clf._geometry(_roi())
    assert ds.phone is True
    assert ds.smoking is False
    assert ds.confidence["phone"] > 0.0


def test_geometry_smoking_when_wrist_near_mouth():
    """Bilek ağza yakın VE ağza kulaktan daha yakın → SİGARA."""
    # mouth vekili = nose + 0.30*fw aşağı. fw=40 → mouth=(50,62). bilek ağzın yanında.
    pts = {
        NOSE: (50, 50),
        L_EAR: (30, 50),
        R_EAR: (70, 50),
        R_WRIST: (51, 62),  # ağıza ~1px; kulağa ~22px (ağız daha yakın)
    }
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model)
    ds = clf._geometry(_roi())
    assert ds.smoking is True
    assert ds.phone is False
    assert ds.confidence["smoking"] > 0.0


def test_geometry_abstains_when_no_ear():
    """Kulak görünmüyorsa İDDİA YOK (göreli kıyas kurulamaz — video_2 FP dersi)."""
    pts = {NOSE: (50, 50), R_WRIST: (50, 55)}  # kulak yok
    model = _FakeModel(_Result(boxes=[_Box()], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model)
    ds = clf._geometry(_roi())
    assert ds.active_flags() == []


def test_geometry_abstains_when_no_nose():
    pts = {L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (50, 55)}  # burun yok
    model = _FakeModel(_Result(boxes=[_Box()], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model)
    assert clf._geometry(_roi()).active_flags() == []


def test_geometry_abstains_when_no_wrist():
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50)}  # bilek yok
    model = _FakeModel(_Result(boxes=[_Box()], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model)
    assert clf._geometry(_roi()).active_flags() == []


def test_geometry_abstains_when_face_width_degenerate():
    """fw < 2.0 (kulaklar üst üste) → geometri kurulamaz, iddia yok."""
    pts = {
        NOSE: (50, 50),
        L_EAR: (50, 50),
        R_EAR: (51, 50),  # fw=1 < 2.0
        R_WRIST: (50, 50),
    }
    model = _FakeModel(_Result(boxes=[_Box()], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model)
    assert clf._geometry(_roi()).active_flags() == []


def test_geometry_single_ear_fallback_face_width():
    """Tek kulak: fw = 2×(burun-kulak); yine de telefon çıkarımı yapılabilir."""
    # tek kulak (sağ) (70,50), nose (50,50) → fw = 2*20 = 40. bilek kulağa yakın.
    pts = {NOSE: (50, 50), R_EAR: (70, 50), R_WRIST: (71, 50)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.7)], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model)
    ds = clf._geometry(_roi())
    assert ds.phone is True


def test_geometry_low_kp_conf_drops_keypoint():
    """kp_conf altındaki keypoint görünmemiş sayılır → kulak yok → çekimser."""
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (71, 50)}
    kps = _make_kps(pts, base_conf=0.10)  # hepsi kp_conf(0.30) altında
    model = _FakeModel(_Result(boxes=[_Box()], keypoints=kps))
    clf = _pose_clf(model=model)
    assert clf._geometry(_roi()).active_flags() == []


def test_geometry_no_results_returns_empty():
    clf = _pose_clf(model=_FakeModel([]))
    assert clf._geometry(_roi()).active_flags() == []
    clf._last_person_seen = True  # önceki sentinel; sıfır boxes onu False yapmalı
    clf2 = _pose_clf(model=_FakeModel(_Result(boxes=[], keypoints=None)))
    assert clf2._geometry(_roi()).active_flags() == []
    assert clf2._last_person_seen is False


def test_geometry_sets_last_person_seen():
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (50, 90)}
    model = _FakeModel(_Result(boxes=[_Box()], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model)
    clf._geometry(_roi())
    assert clf._last_person_seen is True


def test_geometry_picks_highest_conf_person():
    """Çok kişili ROI'de en yüksek conf'lu kişi sürücü adayı (best_i seçimi)."""
    # iki kişi: 0 düşük conf nötr poz, 1 yüksek conf telefon pozu
    xy0 = [(0.0, 0.0)] * _NKP
    c0 = [0.0] * _NKP
    p1 = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (71, 50)}
    xy1 = [(0.0, 0.0)] * _NKP
    c1 = [0.0] * _NKP
    for i, (x, y) in p1.items():
        xy1[i] = (x, y)
        c1[i] = 0.9
    kps = _Keypoints([xy0, xy1], [c0, c1])
    boxes = [_Box(conf=0.3), _Box(conf=0.95)]
    model = _FakeModel(_Result(boxes=boxes, keypoints=kps))
    clf = _pose_clf(model=model)
    ds = clf._geometry(_roi())
    assert ds.phone is True  # yüksek conf'lu kişi (1) seçildi


# --------------------------------------------------------------------------- #
# pt() guard — kc xy'den KISA (bug düzeltmesi: IndexError olmamalı)
# --------------------------------------------------------------------------- #
def test_geometry_tolerates_short_conf_array():
    """kps.conf, xy'den kısa dönerse pt() IndexError ATMAZ (sessiz çekimser)."""
    xy = [(50.0, 50.0)] * _NKP  # tam uzunluk
    kps = _Keypoints([xy], [[0.9, 0.9]])  # conf yalnız 2 eleman (xy'den çok kısa)
    model = _FakeModel(_Result(boxes=[_Box()], keypoints=kps))
    clf = _pose_clf(model=model)
    ds = clf._geometry(_roi())  # patlamamalı
    assert isinstance(ds, DriverState)


def test_geometry_conf_none_treated_as_full_confidence():
    """kps.conf None ise tüm keypoint'ler güvenli sayılır (conf=[1.0]*len)."""
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (71, 50)}
    kps = _make_kps(pts, conf_present=False)  # conf=None
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=kps))
    clf = _pose_clf(model=model)
    assert clf._geometry(_roi()).phone is True


# --------------------------------------------------------------------------- #
# _object_evidence — nesne kanıtı
# --------------------------------------------------------------------------- #
def test_object_evidence_sets_phone_with_canonical_name():
    """'cell phone' → canonical 'phone' bayrağı set edilir, conf taşınır."""
    boxes = [_Box(conf=0.7, cls=0)]
    r = _Result(boxes=boxes, names={0: "cell phone"})
    obj = _FakeModel(r, names={0: "cell phone"})
    clf = _pose_clf(model=None, obj_model=obj)
    ds = DriverState()
    clf._object_evidence(_roi(), ds)
    assert ds.phone is True
    assert ds.confidence["phone"] == 0.7


def test_object_evidence_filters_unrelated_classes():
    """phone/smoking dışı sınıf (ör. 'car') bayrak set etmez."""
    boxes = [_Box(conf=0.9, cls=0)]
    r = _Result(boxes=boxes, names={0: "car"})
    obj = _FakeModel(r, names={0: "car"})
    clf = _pose_clf(model=None, obj_model=obj)
    ds = DriverState()
    clf._object_evidence(_roi(), ds)
    assert ds.active_flags() == []


def test_object_evidence_max_merges_confidence():
    """Aynı sınıf iki kutu → conf max-birleştirilir (mevcut değerin üstündeyse)."""
    boxes = [_Box(conf=0.4, cls=0), _Box(conf=0.85, cls=0)]
    r = _Result(boxes=boxes, names={0: "cigarette"})
    obj = _FakeModel(r, names={0: "cigarette"})
    clf = _pose_clf(model=None, obj_model=obj)
    ds = DriverState()
    ds.confidence["smoking"] = 0.5  # mevcut değer
    clf._object_evidence(_roi(), ds)
    assert ds.smoking is True
    assert ds.confidence["smoking"] == 0.85  # max(0.5, 0.4, 0.85)


def test_object_evidence_no_results():
    clf = _pose_clf(model=None, obj_model=_FakeModel([]))
    ds = DriverState()
    clf._object_evidence(_roi(), ds)
    assert ds.active_flags() == []


# --------------------------------------------------------------------------- #
# _driver_crop — önbellek mantığı
# --------------------------------------------------------------------------- #
def _locator_model(box_xyxy):
    """_locate_driver için: tek kutuyu döndüren sahte pose modeli.

    Kutu PREPPED (ölçeklenmiş) kare koordinatındadır; bu yüzden crop testlerinde
    roi_min_side=1 ile ölçek 1.0 tutulur (kutu ham ROI ile aynı uzayda kalır).
    """
    return _FakeModel(_Result(boxes=[_Box(conf=0.9, xyxy=box_xyxy)], keypoints=None))


def test_driver_crop_caches_and_ages():
    """İlk kare tespit eder (yaş=0); sonraki kareler cache-hit (yaş artar)."""
    clf = _pose_clf(model=_locator_model((20, 20, 60, 90)), crop_enabled=True)
    roi = _roi()
    crop, box = clf._driver_crop(roi, key=1)
    assert box is not None
    assert clf._crop_cache[1][1] == 0  # taze tespit, yaş 0
    # ikinci kare: cache hit, model PREDICT ÇAĞRILMAZ, yaş artar
    calls_before = clf.model.calls
    crop2, box2 = clf._driver_crop(roi, key=1)
    assert clf.model.calls == calls_before  # yeni inference yok
    assert clf._crop_cache[1][1] == 1  # yaş arttı


def test_driver_crop_redetect_after_threshold():
    """redetect_every aşılınca cache düşer ve yeniden tespit edilir."""
    clf = _pose_clf(model=_locator_model((20, 20, 60, 90)), crop_enabled=True, crop_redetect=3)
    roi = _roi()
    clf._driver_crop(roi, key=1)  # tespit, yaş 0
    for _ in range(3):  # yaş 1,2,3 → 3'te redetect_every'e ulaşır, hit kalmaz
        clf._driver_crop(roi, key=1)
    calls_before = clf.model.calls
    clf._driver_crop(roi, key=1)  # yaş 3 >= 3 → cache miss → predict
    assert clf.model.calls == calls_before + 1


def test_driver_crop_min_gain_keeps_full_roi():
    """Kazanç min_gain altındaysa ROI olduğu gibi bırakılır (kırpma anlamsız)."""
    # kutu neredeyse tüm ROI'yi kaplıyor → kazanç ~1.0 < min_gain
    clf = _pose_clf(model=_locator_model((0, 0, 100, 100)), crop_enabled=True, crop_min_gain=1.25)
    roi = _roi()
    crop, box = clf._driver_crop(roi, key=1)
    assert box is None  # kırpılmadı
    assert 1 not in clf._crop_cache  # cache yazılmadı


def test_driver_crop_no_person_returns_full_roi():
    """Kişi bulunamazsa dürüstçe tüm ROI döner (box None)."""
    clf = _pose_clf(model=_FakeModel(_Result(boxes=[], keypoints=None)), crop_enabled=True)
    crop, box = clf._driver_crop(_roi(), key=1)
    assert box is None
    assert crop.shape == (100, 100, 3)


def test_driver_crop_zero_size_roi():
    clf = _pose_clf(model=_locator_model((0, 0, 1, 1)), crop_enabled=True)
    empty = np.zeros((0, 0, 3), np.uint8)
    crop, box = clf._driver_crop(empty, key=1)
    assert box is None


# --------------------------------------------------------------------------- #
# _locate_driver — köşe seçimi
# --------------------------------------------------------------------------- #
def test_locate_driver_picks_corner_not_highest_conf():
    """Köşeye yakın kutu seçilir — en yüksek conf değil (yansıma/yolcu dersi)."""
    # corner_target (1,1) = sağ-alt. yolcu (sol-üst) yüksek conf, sürücü (sağ-alt) düşük.
    passenger = _Box(conf=0.95, xyxy=(0, 0, 20, 20))  # sol-üst, net
    driver = _Box(conf=0.4, xyxy=(80, 80, 100, 100))  # sağ-alt, daha az net
    model = _FakeModel(_Result(boxes=[passenger, driver], keypoints=None))
    clf = _pose_clf(model=model, corner_target=(1.0, 1.0))
    box = clf._locate_driver(_roi())
    assert box is not None
    cx = (box[0] + box[2]) / 2
    assert cx > 50  # sağ taraftaki (sürücü) seçildi


def test_locate_driver_none_when_no_boxes():
    clf = _pose_clf(model=_FakeModel(_Result(boxes=[], keypoints=None)))
    assert clf._locate_driver(_roi()) is None
    clf2 = _pose_clf(model=_FakeModel([]))
    assert clf2._locate_driver(_roi()) is None


# --------------------------------------------------------------------------- #
# infer — uçtan uca + bastırma latch'i
# --------------------------------------------------------------------------- #
def test_infer_none_roi_returns_empty():
    clf = _pose_clf(model=_FakeModel([]))
    assert clf.infer(None).active_flags() == []
    assert clf.infer(np.zeros((0, 0, 3), np.uint8)).active_flags() == []


def test_infer_geometry_phone_flows_through():
    """Geometri telefonu uçtan uca infer çıktısına taşınır (crop kapalı)."""
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (71, 50)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model, crop_enabled=False)
    ds = clf.infer(_roi(), track_id=5)
    assert ds.phone is True


class _PhoneObjModel:
    """Her karede güçlü telefon NESNESİ döndüren sahte v4 dedektörü."""

    def __init__(self, conf):
        self._conf = conf
        self.names = {0: "phone"}

    def predict(self, *_a, **_k):
        return [_Result(boxes=[_Box(conf=self._conf, cls=0)], names={0: "phone"})]


def test_infer_phone_object_suppresses_geometry_smoking():
    """Güçlü telefon nesnesi → geometrik 'sigara' bastırılır; telefon İLERİ TAŞINMAZ."""
    # geometri bu karede sigara üretir (bilek ağza yakın)
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (51, 62)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(
        model=model,
        crop_enabled=False,
        obj_model=_PhoneObjModel(conf=0.5),  # >= suppress_conf 0.30
        obj_suppress_frames=3,
    )
    ds = clf.infer(_roi(), track_id=9)
    assert ds.phone is True  # nesne kanıtı
    assert ds.smoking is False  # geometrik sigara bastırıldı
    assert clf._smoke_suppress[9] == 2  # latch dolduruldu (3) sonra bu karede -1


def test_infer_weak_phone_object_does_not_suppress():
    """Telefon nesnesi suppress_conf altında → bastırma TETİKLENMEZ (sigara kalır)."""
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (51, 62)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(
        model=model,
        crop_enabled=False,
        obj_model=_PhoneObjModel(conf=0.20),  # < suppress_conf 0.30
        obj_suppress_conf=0.30,
    )
    ds = clf.infer(_roi(), track_id=3)
    assert ds.phone is True  # nesne yine bayraklar (obj_conf duyarlı)
    assert ds.smoking is True  # ama bastırma yok → geometrik sigara kalır
    assert 3 not in clf._smoke_suppress


def test_infer_suppress_latch_counts_down_across_frames():
    """Latch dolduktan sonra telefon nesnesi gitse de bir süre sigara bastırılır."""
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (51, 62)}
    # 1. kare: güçlü telefon nesnesi var → latch dolar
    model1 = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(
        model=model1,
        crop_enabled=False,
        obj_model=_PhoneObjModel(conf=0.5),
        obj_suppress_frames=2,
    )
    clf.infer(_roi(), track_id=1)
    assert clf._smoke_suppress[1] == 1  # 2 dolduruldu, bu karede -1
    # 2. kare: telefon nesnesi YOK ama latch aktif → sigara hâlâ bastırılır
    clf.obj_model = None
    clf.model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    ds2 = clf.infer(_roi(), track_id=1)
    assert ds2.smoking is False  # latch hâlâ bastırıyor
    assert clf._smoke_suppress[1] == 0


def test_infer_drops_cache_when_person_lost():
    """Kırpık var ama kişi görünmüyorsa cache düşürülür (sürücü kaydı/araç döndü)."""
    # crop tespit eder ama _geometry kişi görmez → cache pop
    clf = _pose_clf(
        model=_FakeModel(
            [
                [_Result(boxes=[_Box(xyxy=(20, 20, 60, 90))], keypoints=None)],  # locate
                [_Result(boxes=[], keypoints=None)],  # geometry: kişi yok
            ]
        ),
        crop_enabled=True,
    )
    clf.infer(_roi(), track_id=4)
    assert 4 not in clf._crop_cache  # bayat kutu düşürüldü


@pytest.mark.parametrize("track_id", [None, 0, 7])
def test_infer_track_id_keying(track_id):
    """track_id=None → key -1; latch belleği track'e doğru anahtarlanır."""
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (51, 62)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model, crop_enabled=False, obj_model=_PhoneObjModel(conf=0.5))
    clf.infer(_roi(), track_id=track_id)
    expected_key = -1 if track_id is None else track_id
    assert expected_key in clf._smoke_suppress


# --------------------------------------------------------------------------- #
# OPSİYONEL İKİNCİ MODEL: özel eğitimli sigara dedektörü (smoking_model)
# roi_objects'in YANINDA koşar; 'smoking' OR'lanır, phone yolu KORUNUR.
# --------------------------------------------------------------------------- #
class _SmokingObjModel:
    """Her karede 'smoking' NESNESİ döndüren sahte özel-eğitimli dedektör."""

    def __init__(self, conf, cls_name="cigarette"):
        self._conf = conf
        self._name = cls_name
        self.names = {0: cls_name}

    def predict(self, *_a, **_k):
        return [_Result(boxes=[_Box(conf=self._conf, cls=0)], names={0: self._name})]


def test_smoking_model_absent_is_noop():
    """smoking_model None → davranış DEĞİŞMEZ (graceful-absent, takım/CI no-op)."""
    # nötr geometri (bilek uzakta) → tek başına hiçbir bayrak yok
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (50, 95)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model, crop_enabled=False, smoking_model=None)
    ds = clf.infer(_roi(), track_id=1)
    assert ds.active_flags() == []  # ikinci model yok → kanıt eklenmez


def test_smoking_model_present_adds_smoking_evidence():
    """Mock sigara modeli → 'smoking' kanıtı eklenir (canonical eşleme + conf)."""
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (50, 95)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model, crop_enabled=False, smoking_model=_SmokingObjModel(conf=0.7))
    ds = clf.infer(_roi(), track_id=1)
    assert ds.smoking is True
    assert ds.confidence["smoking"] == 0.7


def test_smoking_model_does_not_set_phone():
    """İkinci model yalnız 'smoking' üretir — phone yolu (roi_objects) ona ait kalır."""
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (50, 95)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    # model yanlışlıkla 'phone' sınıfı verse bile İKİNCİ kanal onu YOK SAYAR
    clf = _pose_clf(
        model=model,
        crop_enabled=False,
        smoking_model=_SmokingObjModel(conf=0.9, cls_name="cell phone"),
    )
    ds = clf.infer(_roi(), track_id=1)
    assert ds.phone is False
    assert ds.smoking is False  # 'phone' adı smoking kanalında yok sayıldı


def test_smoking_model_preserves_phone_object_evidence():
    """roi_objects phone NESNESİ + ikinci-model sigara BİRLİKTE çalışır (phone KAÇMAZ).

    A/B regresyonunun çekirdeği: drop-in phone kanıtını siliyordu. Ayrı kanalda
    phone (roi_objects) KORUNUR ve sigara (ikinci model) EKLENİR.
    """
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (50, 95)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(
        model=model,
        crop_enabled=False,
        obj_model=_PhoneObjModel(conf=0.5),  # roi_objects: phone nesnesi
        smoking_model=_SmokingObjModel(conf=0.6),  # ikinci model: sigara
        obj_suppress_frames=3,
    )
    ds = clf.infer(_roi(), track_id=1)
    assert ds.phone is True  # phone nesne-kanıtı KORUNDU (regresyon yok)
    assert ds.smoking is True  # ikinci-model sigara kanıtı eklendi
    assert ds.confidence["phone"] == 0.5
    assert ds.confidence["smoking"] == 0.6


def test_smoking_model_evidence_survives_phone_suppression():
    """Telefon bastırma latch'i GEOMETRİK sigarayı söndürür ama NESNE kanıtını değil.

    Bastırma yalnız geo.smoking'i sıfırlar; ikinci-model nesne kanıtı gerçek
    sigara nesnesidir → ds.smoking'te kalır (phone-confusable geometri değil).
    """
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (51, 62)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.8)], keypoints=_make_kps(pts)))
    clf = _pose_clf(
        model=model,
        crop_enabled=False,
        obj_model=_PhoneObjModel(conf=0.5),  # bastırma tetikler
        smoking_model=_SmokingObjModel(conf=0.6),
        obj_suppress_frames=3,
    )
    ds = clf.infer(_roi(), track_id=1)
    assert ds.phone is True
    assert ds.smoking is True  # nesne-kanıtı bastırmadan ETKİLENMEDİ
    assert ds.confidence["smoking"] >= 0.6  # en az ikinci-model kanıtı korunur


def test_smoking_model_max_merges_with_geometry_smoking():
    """İkinci-model conf, geometrik sigara conf'u ile max-birleştirilir."""
    # geometri sigara üretir (bilek ağza yakın), düşük conf person → düşük geo conf
    pts = {NOSE: (50, 50), L_EAR: (30, 50), R_EAR: (70, 50), R_WRIST: (51, 62)}
    model = _FakeModel(_Result(boxes=[_Box(conf=0.3)], keypoints=_make_kps(pts)))
    clf = _pose_clf(model=model, crop_enabled=False, smoking_model=_SmokingObjModel(conf=0.95))
    ds = clf.infer(_roi(), track_id=1)
    assert ds.smoking is True
    assert ds.confidence["smoking"] == 0.95  # ikinci model daha güçlü
