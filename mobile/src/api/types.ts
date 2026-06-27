// RoadGuard backend veri sözleşmeleri (aura/schema.py ile birebir) — TypeScript karşılıkları.
// Bu tipler downstream'in (kartlar, rozetler, overlay) güvendiği tek kaynaktır.

// ---- Number Verification (nv_mock :8082) ---- //
export interface VerifyRequest {
  phone_number: string;
  sim_token?: string | null;
}
export interface VerifyResponse {
  verified: boolean;
  latency_ms: number;
  phone_number: string;
}

// ---- Event stream (WS /stream/events) ---- //
export type AuraEventType =
  | "DETECTION_UPDATE"
  | "PLATE_CONFIRMED"
  | "PLATE_REJECTED"
  | "DRIVER_STATE"
  | "DRIVER_LOCKED"
  | "SPEED"
  | "QOD_TRIGGER"
  | "QOD_RELEASE"
  | "RISK_ALERT"
  | "SPEED_LIMIT_DETECTED"
  | "SPEED_LIMIT_VIOLATION";

export interface AuraEvent {
  event_id: string;
  ts: number;
  track_id: number;
  type: AuraEventType | string; // ileri uyumlu: bilinmeyen tipler de taşınır
  payload: Record<string, unknown>;
  source: string;
}

// Bilinen QoD tetikleyici sebepleri (qod/client.py): "vehicle_approach" |
// "speed_anomaly" | ... (string olarak gelir; UI hepsini gösterir).
export type QodReason = string;

// ---- Annotation stream (WS /stream/annotations) ---- //
// track dict alanları: aura/pipeline/pipeline.py::record_to_annotation
export type PlateStatus = "pending" | "confirmed" | "rejected";

export interface AnnotationTrack {
  track_id: number;
  bbox: [number, number, number, number]; // [x1,y1,x2,y2] piksel
  cls: string; // araç sınıfı (car/truck/bus/...)
  conf: number;
  plate: string | null; // okunan plaka metni
  plate_status: PlateStatus;
  plate_partial?: string | null; // tam doğrulanamayan en güçlü aday
  driver: string[]; // aktif sürücü ihlalleri: ["phone","smoking","no_seatbelt","fatigue"]
  driver_id?: number | null;
  driver_locked?: boolean;
  speed_kmh: number | null;
  speed_calibrated?: boolean;
  relative_velocity_flag?: boolean; // ego'ya göre hızlı yaklaşıyor
  swerving?: boolean; // dikkatsiz sürüş (zigzag/ani kayma)
  risk_flags: string[]; // birleşik risk bayrakları
  qod_active: boolean; // bu araç için yüksek-kalite modu açık mı
}

export interface AnnotationPerson {
  bbox: [number, number, number, number];
  role: "driver" | "passenger";
  track_id: number;
  vehicle_id: number;
  locked: boolean;
}

export interface AnnotationSign {
  bbox: [number, number, number, number];
  cls: string;
  conf?: number;
  speed_limit_kmh: number | null;
}

export interface SceneContext {
  active_speed_limit_kmh?: number | null;
  speed_limit_source_cls?: string | null;
  sign_count?: number;
}

export interface AnnotationFrame {
  frame_id: number;
  ts: number;
  tracks: AnnotationTrack[];
  persons: AnnotationPerson[];
  signs: AnnotationSign[];
  scene: SceneContext;
}

// ---- Stream status (GET /stream/status) ---- //
export interface StreamStatus {
  running: boolean;
  source: string | null;
  device: string | null;
  bbox_overlay: boolean;
  frame_count: number;
  fps: number;
  uptime_s: number;
  active_tracks: number;
  qod_active_sessions: number;
}
