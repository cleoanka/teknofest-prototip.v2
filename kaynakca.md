# KAYNAKÇA — AURA (TEKNOFEST 2026 "5G & YZ ile Akıllı Yol Güvenliği")

> **Ne bu?** ULTRAPLAN **W2** (23–28 Haz: rapor + cila) için, **FTR §5 Kaynakça (5 puan)**
> kalemini besleyen tam, akademik formatlı kaynak listesi. Buradaki her kaynak,
> AURA kod tabanında **gerçekten kullanılan** bir yöntem/kütüphane/veri setine ya da
> rapor metninde atıf yapılan doğrulanabilir bir belgeye karşılık gelir.
>
> **Onur zırhı (K-004):** Uydurma kaynak yoktur. Her giriş ya repoda kullanılan bir
> bağımlılıktır (sürümler `.venv` / `pyproject.toml` ile doğrulandı) ya da `docs/veri_seti.md`,
> `docs/yol_haritasi.md`, `ftr_rapor_taslak.md` içinde zaten atıfı geçen bir kaynaktır.
> **Kullanılan** ile **ilgili/alternatif** (araştırıldı ama varsayılan hatta değil) kaynaklar
> ayrı ayrı işaretlenmiştir. Tüm bağlantılar **erişim tarihi 23.06.2026** ile verilmiştir ve
> rapora konmadan önce son kez doğrulanmalıdır.
>
> **Kapsam kuralı:** FTR raporu 3–10 sayfa; §5 kısa olmalı. Bu dosya **havuz**dur — §13'teki
> "FTR §5'e hazır kısa liste" rapora doğrudan kopyalanacak çekirdek alt-kümedir; geri kalanı
> gerekçe/izlenebilirlik içindir.
>
> **Atıf biçimi:** Numaralı (IEEE-benzeri). Yazar(lar), "Başlık," _Kaynak/Yer_, yıl. DOI/URL
> (erişim tarihi). Lisans (veri setlerinde).

---

## 1. Nesne Tespiti ve Çok-Nesne Takibi (Stage-1 çekirdek)

**[1]** Ultralytics, "Ultralytics YOLO — Documentation," sürüm **8.4.66** (repoda kurulu;
`ultralytics>=8.4.0`, YOLO26 mimarisi 8.4.x ile gelir), 2025–2026. https://docs.ultralytics.com
(erişim 23.06.2026). Lisans: AGPL-3.0. — *Kullanılan:* birincil dedektör `yolo26l` (COCO ön-eğitimli)
ve özel fine-tune `custom_*` ağırlıkları; `aura/detection/yolo.py`.

**[2]** J. Redmon, S. Divvala, R. Girshick ve A. Farhadi, "You Only Look Once: Unified, Real-Time
Object Detection," _IEEE/CVF CVPR_, 2016, ss. 779–788. DOI: 10.1109/CVPR.2016.91. — *Kavramsal temel:*
YOLO tek-aşamalı tespit paradigması.

**[3]** Y. Zhang, P. Sun, Y. Jiang ve diğ., "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box," _ECCV_, 2022. arXiv:2110.06864. https://arxiv.org/abs/2110.06864 (erişim 23.06.2026).
— *Kullanılan:* varsayılan takip algoritması (Ultralytics `bytetrack.yaml`); `aura/detection/yolo.py`.

**[4]** N. Wojke, A. Bewley ve D. Paulus, "Simple Online and Realtime Tracking with a Deep
Association Metric (DeepSORT)," _IEEE ICIP_, 2017. arXiv:1703.07402. — *İlgili:* takip literatürü temeli
(ByteTrack'in selefi; karşılaştırma için).

**[5]** A. Bewley, Z. Ge, L. Ott, F. Ramos ve B. Upcroft, "Simple Online and Realtime Tracking (SORT),"
_IEEE ICIP_, 2016, ss. 3464–3468. arXiv:1602.00763. — *Kavramsal temel:* Kalman + Macar atama tabanlı takip.

**[6]** H. W. Kuhn, "The Hungarian Method for the Assignment Problem," _Naval Research Logistics
Quarterly_, c. 2, 1955, ss. 83–97. DOI: 10.1002/nav.3800020109. — *Kavramsal temel:* takipte tespit-iz
eşleştirme (atama problemi).

---

## 2. Plaka Tespiti ve Optik Karakter Tanıma (OCR)

**[7]** A. (ankandrew), "fast-plate-ocr: Lightweight & fast OCR models for license plate recognition,"
sürüm **1.1.0** (repoda kurulu; `onnxruntime` ile), 2024–2026.
https://github.com/ankandrew/fast-plate-ocr (erişim 23.06.2026). — *Kullanılan (VARSAYILAN OCR motoru):*
plakaya-özel hafif ONNX OCR; gerçek video_3'te EasyOCR il-kodu misread'ini kurtardı → **3/3 exact,
CER 0**. `aura/plate/ocr.py:build_ocr`.

**[8]** JaidedAI, "EasyOCR: Ready-to-use OCR with 80+ languages," sürüm **1.7.2** (repoda kurulu),
2020–2026. https://github.com/JaidedAI/EasyOCR (erişim 23.06.2026). Lisans: Apache-2.0. — *Kullanılan
(yedek/fallback OCR motoru):* `fast-plate-ocr` yoksa loglu düşüş; `aura/plate/ocr.py`.

**[9]** PaddlePaddle, "PaddleOCR — PP-OCRv4," 2023–2026. https://github.com/PaddlePaddle/PaddleOCR
(erişim 23.06.2026). Lisans: Apache-2.0. — *Opsiyonel (entegre, varsayılan değil):* `plate.ocr_engine=paddleocr`
seçilince devreye girer; kurulu değilse fallback. `pyproject.toml` extra `paddle`.

**[10]** K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization (CLAHE)," _Graphics Gems IV_,
P. Heckbert (Ed.), Academic Press, 1994, ss. 474–485. DOI: 10.1016/B978-0-12-336156-1.50061-6. —
*Kullanılan:* düşük-ışık plaka/ROI iyileştirme; `aura/plate/ocr.py` (CLAHE) ve `pose.roi_enhance`.

**[11]** R. Hartley ve A. Zisserman, _Multiple View Geometry in Computer Vision_, 2. baskı, Cambridge
University Press, 2004. ISBN 978-0521540513. — *Kavramsal temel:* homografi / perspektif düzeltme
(`cv2.getPerspectiveTransform`); `aura/optional/homography_ipm.py`.

**[12]** S. Silva ve C. R. Jung, "A Flexible Approach for Automatic License Plate Recognition in
Unconstrained Scenarios (WPOD-NET / IWPOD-NET)," _IEEE Trans. Intelligent Transportation Systems_, 2021.
DOI: 10.1109/TITS.2021.3055946. — *İlgili/alternatif (araştırıldı, opsiyonel hat):* kısıtsız plakada 4-köşe
tespiti + dewarp; `docs/yol_haritasi.md` §1.

**[13]** S. Zherzdev ve A. Gruzdev, "LPRNet: License Plate Recognition via Deep Neural Networks,"
arXiv:1806.10447, 2018. — *İlgili/alternatif:* plakaya-özel hafif OCR alternatifi.

**[14]** C. Guo, C. Li, J. Guo ve diğ., "Zero-Reference Deep Curve Estimation for Low-Light Image
Enhancement (Zero-DCE)," _IEEE/CVF CVPR_, 2020. arXiv:2001.06826. — *İlgili/alternatif:* gerçek-zamanlı
düşük-ışık iyileştirme; CLAHE yerine değerlendirilen seçenek (`docs/yol_haritasi.md` §1).

---

## 3. Sürücü Durumu / Dikkatsiz Sürüş Tespiti (Stage-2)

**[15]** Ultralytics, "YOLO — Pose / Keypoint estimation," 2025–2026.
https://docs.ultralytics.com/tasks/pose (erişim 23.06.2026). — *Kullanılan:* sürücü hibrit motorunun
poz-geometrisi katmanı (landmark kütüphanesi gerektirmeyen); `aura/driver_state/pose.py`.

**[16]** T. H. N. Le, Y. Zheng, C. Zhu, K. Luu ve M. Savvides, "Multiple Scale Faster-RCNN Approach to
Driver's Cell-phone Usage and Hands on Steering Wheel Detection," _IEEE/CVF CVPR Workshops_, 2016.
DOI: 10.1109/CVPRW.2016.13. — *Kavramsal temel:* sürücü dikkat dağınıklığı (telefon kullanımı) tespiti
literatürü.

**[17]** State Farm, "Distracted Driver Detection," Kaggle yarışması, 2016.
https://www.kaggle.com/c/state-farm-distracted-driver-detection (erişim 23.06.2026). — *Kavramsal temel:*
araç-içi davranış sınıflandırması referans problemi.

---

## 4. Hız Kestirimi ve Sahne Geometrisi

**[18]** R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," _Transactions of
the ASME — Journal of Basic Engineering_, c. 82, sayı 1, 1960, ss. 35–45. DOI: 10.1115/1.3662552. —
*Kullanılan:* hız düzleştirme/öngörü (Kalman filtresi) ve takipte hareket modeli;
`aura/speed/estimator.py`, `aura/speed/calibration.py`.

**[19]** M. Bertozzi ve A. Broggi, "GOLD: A Parallel Real-Time Stereo Vision System for Generic Obstacle
and Lane Detection (Inverse Perspective Mapping)," _IEEE Trans. Image Processing_, c. 7, sayı 1, 1998,
ss. 62–81. DOI: 10.1109/83.650851. — *Kullanılan:* ters perspektif eşleme (IPM) ile metrik
oto-kalibrasyon; `aura/optional/homography_ipm.py`, `aura/speed/estimator.py`.

**[20]** OpenCV, "Camera Calibration and 3D Reconstruction — Geometric Image Transformations,"
sürüm 4.13. https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html (erişim 23.06.2026). —
*Kullanılan:* `getPerspectiveTransform`, `warpPerspective` (dewarp + IPM uygulaması).

---

## 5. Sahne / Trafik Tabelası ve Hız-Limiti Çıkarımı

**[21]** J. Stallkamp, M. Schlipsing, J. Salmen ve C. Igel, "The German Traffic Sign Recognition
Benchmark (GTSRB): A multi-class classification competition," _IJCNN_, 2011, ss. 1453–1460.
DOI: 10.1109/IJCNN.2011.6033395. — *Kavramsal temel:* trafik tabelası tanıma ve hız-limiti sınıfları
(`speed_limit_*`); `aura/scene/sign_tracker.py`, `config/default.yaml` `sign.value_map`.

---

## 6. 5G Telekom Katmanı — CAMARA QoD ve Number Verification

**[22]** CAMARA Project (Linux Foundation), "Quality on Demand (QoD) API," 2023–2026.
https://github.com/camaraproject/QualityOnDemand (erişim 23.06.2026). — *Kullanılan (sözleşme taklidi):*
QoD oturum oluştur/sorgula/sil + QoS profilleri (QOS_E/L); `aura/qod/client.py`, `services/qod_mock/`.

**[23]** CAMARA Project (Linux Foundation), "Number Verification API," 2023–2026.
https://github.com/camaraproject/NumberVerification (erişim 23.06.2026). — *Kullanılan (sözleşme taklidi):*
SIM/şebeke tabanlı sessiz doğrulama (`POST /verify`, SMS/OTP yok); `services/nv_mock/`.

**[24]** GSMA, "Open Gateway — Universal Network APIs," 2023–2026.
https://www.gsma.com/solutions-and-impact/gsma-open-gateway/ (erişim 23.06.2026). — *Bağlamsal:* CAMARA
API'lerinin operatör tarafı çerçevesi; finalde gerçek uç/credential bu kapsamda gelir.

**[25]** 3GPP, "System Architecture for the 5G System (5GS)," TS 23.501, Rel-18. https://www.3gpp.org
(erişim 23.06.2026). — *Bağlamsal:* 5G mimarisi ve QoS modeli (5QI), QoD'nin altyapı dayanağı.

**[26]** 3GPP, "Policy and Charging Control Framework for the 5G System (5GS); Stage 2," TS 23.503,
Rel-18. https://www.3gpp.org (erişim 23.06.2026). — *Bağlamsal:* QoS/PCC çerçevesi; QoD profillerinin
şebeke karşılığı.

---

## 7. Yazılım Çerçeveleri ve Çalışma-Zamanı (doğrulanmış sürümler)

**[27]** A. Paszke, S. Gross, F. Massa ve diğ., "PyTorch: An Imperative Style, High-Performance Deep
Learning Library," _NeurIPS_, 2019. — *Kullanılan:* derin öğrenme çalışma-zamanı, sürüm **torch 2.12.0**
(Apple Silicon MPS arka-ucu); `bootstrap.py` backend'e göre kurar.

**[28]** G. Bradski, "The OpenCV Library," _Dr. Dobb's Journal of Software Tools_, 2000. — *Kullanılan:*
görüntü işleme; sürüm **opencv-python 4.13.0.92**.

**[29]** C. R. Harris, K. J. Millman, S. J. van der Walt ve diğ., "Array Programming with NumPy,"
_Nature_, c. 585, 2020, ss. 357–362. DOI: 10.1038/s41586-020-2649-2. — *Kullanılan:* sayısal çekirdek;
sürüm **numpy 2.4.6**.

**[30]** S. Ramírez, "FastAPI — Modern, fast web framework for building APIs with Python," 2018–2026.
https://fastapi.tiangolo.com (erişim 23.06.2026). Lisans: MIT. — *Kullanılan:* `inference_api` ve mock
servisler; sürüm **fastapi 0.136.3**, **uvicorn 0.49.0**.

**[31]** Shapely / GEOS, "Shapely: Manipulation and analysis of geometric objects," 2007–2026.
https://shapely.readthedocs.io (erişim 23.06.2026). — *Kullanılan:* ROI/poligon geometri; sürüm
**shapely 2.1.2**.

**[32]** ONNX Runtime geliştiricileri, "ONNX Runtime: cross-platform inference accelerator,"
Microsoft, 2018–2026. https://onnxruntime.ai (erişim 23.06.2026). Lisans: MIT. — *Kullanılan:*
`fast-plate-ocr` ONNX yürütme arka-ucu.

**[33]** Meta / Expo, "React Native" ve "Expo SDK," 2015–2026. https://reactnative.dev ·
https://docs.expo.dev (erişim 23.06.2026). — *Kullanılan:* final demo mobil uygulaması; `mobile/`.

---

## 8. Veri Setleri ve Lisansları (FTR §2 ile birebir tutarlı)

> Kaynak, sayı ve lisanslar `docs/veri_seti.md` ve `ftr_rapor_taslak.md` §2 ile **birebir aynıdır**.
> Tümü PIL ile doğrulanmış, 80/10/10 (seed 42) bölünmüştür. **Kullanım öncesi lisans/içerik
> uyumluluğu son kez teyit edilir** (şartname açık-kaynak veri kullanımını serbest bırakır).

**[34]** T.-Y. Lin, M. Maire, S. Belongie ve diğ., "Microsoft COCO: Common Objects in Context,"
_ECCV_, 2014. arXiv:1405.0312. https://cocodataset.org (erişim 23.06.2026). Lisans: CC BY 4.0. —
*Kullanılan:* genel sınıflar (`car`, `bus`, `truck`, `motorcycle`, `person`); stok `yolo26l` ön-eğitimi
ve held-out doğrulama (val2017, 5000 görsel).

**[35]** keremberke, "License Plate Object Detection Dataset" (Roboflow "Vehicle Registration Plates v1"
→ COCO→YOLO), Hugging Face, 2023. https://huggingface.co/datasets/keremberke/license-plate-object-detection
(erişim 23.06.2026). Lisans: CC BY 4.0. — *Kullanılan:* `license_plate`, **8.823 görsel**; YOLO26s
fine-tune → held-out **mAP50 0.983 / mAP50-95 0.707**; `custom_license_plate` = varsayılan LP dedektör.

**[36]** ramankamran (Roboflow `oohmp/seatbelt-detection` v2), "Seatbelt Detection (YOLOv11) Dataset,"
Hugging Face, 2024. https://huggingface.co/datasets/ramankamran/seatbelt-detection-v2i-yolov11-lt
(erişim 23.06.2026). Lisans: CC BY 4.0. — *Kullanılan:* `seatbelt → no_seatbelt_evidence`, **3.104 görsel**
(denge 1,27); held-out **mAP50 0.895 / mAP50-95 0.546**; opsiyonel (dış-kamera görüş açısı).

**[37]** "CigDet — Cigarette Detection Dataset," Mendeley Data, 2021. **DOI: 10.17632/6hyrr8typ7.1**.
https://data.mendeley.com/datasets/6hyrr8typ7/1 (erişim 23.06.2026). Lisans: CC BY 4.0. — *Kullanılan:*
`cigarette → smoking`, **557 görsel** (446 train / 111 test); held-out **mAP50 0.856 / mAP50-95 0.457**;
`pose.py` ikinci-model.

**[38]** anywaylabs, "Synthetic Driver Monitoring Dataset," Hugging Face, 2024.
https://huggingface.co/datasets/anywaylabs/synthetic-driver-monitoring (erişim 23.06.2026).
Lisans: CC BY 4.0. — *Kullanılan:* `phone`, **659 görsel**; **SENTETİK render → domain-uyum riski**
(rapor metninde dürüstçe belirtilir).

### İlgili / büyük setler (araştırıldı, manifestte listeli; varsayılan eğitimde değil)

**[39]** "driver-smoking-detecor," Roboflow Universe (gordon-v6v6v), ~1.066 görsel, CC BY 4.0.
https://universe.roboflow.com (erişim 23.06.2026). — *İlgili:* API/Roboflow erişimi gerektirir.

**[40]** "Smoker YOLO v4," Roboflow Universe (dingguangyu), ~4.221 görsel, CC BY 4.0.
https://universe.roboflow.com (erişim 23.06.2026). — *İlgili:* daha büyük sigara seti (erişim gerektirir).

**[41]** "seat_belt_detection," Roboflow Universe (helmet-seatbelt-detection), ~3.820 görsel, CC BY 4.0.
https://universe.roboflow.com (erişim 23.06.2026). — *İlgili.*

**[42]** lavdeep1234, "Driver Seat Belt Detection," Kaggle, ~30.000 görsel, CC0.
https://www.kaggle.com/datasets (erişim 23.06.2026). — *İlgili:* manuel indirme (kimlik/lisans onayı).

**[43]** "traffic (minibus/kamyon/otobüs)," Roboflow Universe (johnny), ~5.150 görsel, CC BY 4.0; ve
"_images_oturum3 (İstanbul dolmuş)," Roboflow Universe (geod), ~3.950 görsel, CC BY 4.0.
https://universe.roboflow.com (erişim 23.06.2026). — *İlgili:* `minibus` için araştırılan açık setler.
**Not (onur):** `minibus` için no-auth açık bbox seti **teyit edilemedi**; `fatigue` için doğrulanmış
açık set **yoktur** — manifestte boş bırakılır (uydurma kaynak eklenmez); komite verisiyle gelir.

---

## 9. Değerlendirme Metrikleri

**[44]** M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn ve A. Zisserman, "The PASCAL Visual
Object Classes (VOC) Challenge," _International Journal of Computer Vision_, c. 88, 2010, ss. 303–338.
DOI: 10.1007/s11263-009-0275-3. — *Kullanılan:* mAP / Precision-Recall metodolojisi; `aura/eval/`.

**[45]** COCO Consortium, "COCO Detection Evaluation (mAP@[.50:.95])," 2014–2026.
https://cocodataset.org/#detection-eval (erişim 23.06.2026). — *Kullanılan:* mAP50 / mAP50-95 ölçüm
protokolü (Ultralytics `model.val`); `weights/custom_*.metrics.json`.

**[46]** Karakter Hata Oranı (Character Error Rate, CER) — Levenshtein düzenleme mesafesi temelli OCR
metriği; V. I. Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals,"
_Soviet Physics Doklady_, c. 10, 1966, ss. 707–710. — *Kullanılan:* plaka OCR doğruluğu (exact-match +
CER); `aura/eval/report.py`.

---

## 10. Yarışma, Şartname ve Standartlar

**[47]** T.C. Sanayi ve Teknoloji Bakanlığı / TÜBİTAK, "TEKNOFEST 2026 — 5G ve Yapay Zekâ ile Akıllı
Yol Güvenliği Yarışması Şartnamesi," 2026. https://www.teknofest.org (erişim 23.06.2026). — *Birincil:*
problem tanımı, FTR formatı, değerlendirme kriterleri, açık-veri kullanım izni, aşama tarihleri
(FTR son teslim **28.06.2026**).

**[48]** TÜBİTAK BİDEB / KYS, "Final Tasarım Raporu (FTR) Şablonu ve Biçim Kuralları," 2026. — *Birincil:*
3–10 sayfa, Arial 12 / Arial Black 14, 1.15 satır, iki yana yaslı, kenar boşlukları; bkz. `ftr.md` §B5.

---

## 11. Yöntem ↔ Kaynak izlenebilirlik özeti

| AURA bileşeni | Dosya | Kaynak no. |
|---|---|---|
| Dedektör (YOLO26 + COCO) | `aura/detection/yolo.py` | [1][2][34] |
| Takip (ByteTrack) | `aura/detection/yolo.py` | [3][5][6] |
| Plaka OCR (fast-plate / EasyOCR / Paddle) | `aura/plate/ocr.py` | [7][8][9] |
| Plaka iyileştirme (CLAHE) + dewarp | `aura/plate/ocr.py`, `aura/optional/homography_ipm.py` | [10][11][20] |
| Sürücü durumu (poz hibrit) | `aura/driver_state/pose.py` | [15][16] |
| Hız (Kalman/EMA/IPM) | `aura/speed/estimator.py`, `calibration.py` | [18][19][20] |
| Tabela / hız-limiti | `aura/scene/sign_tracker.py` | [21] |
| QoD (CAMARA) | `aura/qod/client.py`, `services/qod_mock/` | [22][24][25][26] |
| Number Verification | `services/nv_mock/` | [23][24] |
| API / servis çerçevesi | `services/inference_api/` | [30] |
| Mobil demo | `mobile/` | [33] |
| Eğitim verisi | `train/datasets.yaml`, `data/processed/` | [34]–[38] |
| Metrikler (mAP/CER) | `aura/eval/` | [44][45][46] |

---

## 12. Lisans envanteri (FTR'de "açık-kaynak kullanım uyumu" notu için)

| Bileşen | Lisans |
|---|---|
| AURA (bu proje) | MIT (`pyproject.toml`) |
| Ultralytics YOLO | AGPL-3.0 |
| EasyOCR, PaddleOCR, FastAPI, ONNX Runtime | Apache-2.0 / MIT |
| Tüm kullanılan veri setleri ([34]–[38]) | CC BY 4.0 |
| Kaggle seatbelt seti [42] | CC0 |

> **Not:** Ultralytics **AGPL-3.0**'dır; ticari/kapalı dağıtımda Enterprise lisans gerekir. FTR/yarışma
> ve akademik kullanım kapsamında uyumludur. Veri setleri **CC BY 4.0** olduğundan rapor/üründe
> **atıf zorunludur** (bu kaynakça bu yükümlülüğü karşılar).

---

## 13. ✂️ FTR §5'e hazır kısa liste (rapora doğrudan kopyalanacak çekirdek)

> Raporun 3–10 sayfa sınırı için yalnızca **kullanılan çekirdek** kaynaklar. Numaraları rapor
> içi atıflarla ([R1] vb.) eşleyin.

```
[R1]  Ultralytics, "Ultralytics YOLO Documentation," v8.4.66, 2026. docs.ultralytics.com.
[R2]  Y. Zhang vd., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box,"
      ECCV, 2022. arXiv:2110.06864.
[R3]  T.-Y. Lin vd., "Microsoft COCO: Common Objects in Context," ECCV, 2014. arXiv:1405.0312.
      (CC BY 4.0)
[R4]  ankandrew, "fast-plate-ocr," 2024–2026. github.com/ankandrew/fast-plate-ocr.
[R5]  JaidedAI, "EasyOCR," 2020–2026. github.com/JaidedAI/EasyOCR. (Apache-2.0)
[R6]  K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," Graphics Gems IV,
      1994, ss. 474–485.
[R7]  R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," ASME J.
      Basic Eng., c. 82, 1960, ss. 35–45.
[R8]  M. Bertozzi, A. Broggi, "GOLD: Inverse Perspective Mapping," IEEE TIP, c. 7(1), 1998,
      ss. 62–81.
[R9]  CAMARA Project (Linux Foundation), "Quality on Demand API," 2023–2026.
      github.com/camaraproject/QualityOnDemand.
[R10] CAMARA Project (Linux Foundation), "Number Verification API," 2023–2026.
      github.com/camaraproject/NumberVerification.
[R11] 3GPP, "5G System Architecture," TS 23.501, Rel-18.
[R12] keremberke, "License Plate Object Detection," Hugging Face, 2023. (CC BY 4.0)
[R13] "CigDet — Cigarette Detection," Mendeley Data, 2021. DOI:10.17632/6hyrr8typ7.1. (CC BY 4.0)
[R14] ramankamran, "Seatbelt Detection (YOLOv11)," Hugging Face, 2024. (CC BY 4.0)
[R15] M. Everingham vd., "The PASCAL VOC Challenge," IJCV, c. 88, 2010, ss. 303–338. (mAP)
[R16] TEKNOFEST 2026, "5G ve YZ ile Akıllı Yol Güvenliği Şartnamesi," teknofest.org, 2026.
```

---

> **Yapay zekâ desteği:** Bu rapor ve depo dokümantasyonunun yazım/düzenlenmesinde büyük dil
> modelinden (Anthropic Claude) destek alınmıştır. Tüm teknik içerik, mimari kararlar, ölçümler ve
> sayılar takım tarafından üretilmiş; depo artefaktlarından (`eval_results/`, `weights/custom_*_s.metrics.json`)
> doğrulanmıştır (K-004). Yapay zekâ yalnızca yazım/derleme/düzen aracı olarak kullanılmıştır.

*Hazırlık: ULTRAPLAN W2 (rapor + cila), erişim tarihi 23.06.2026. Sürümler `.venv`/`pyproject.toml`,
veri setleri `docs/veri_seti.md` ile çapraz doğrulandı. Onur zırhı (K-004): uydurma kaynak yok;
"kullanılan" ve "ilgili/alternatif" ayrımı korunmuştur. Rapora konmadan önce tüm bağlantılar son kez
açılıp doğrulanmalı; CC BY 4.0 veri setleri için atıf zorunludur.*
