// AURA mobil — renkler ve etiket sözlükleri (tek kaynak; kart/rozet/akış paylaşır).

export const COLORS = {
  bg: "#0d1117",
  card: "#161b22",
  cardAlt: "#1c232c",
  border: "#30363d",
  text: "#e6edf3",
  muted: "#8b97a6",
  green: "#00ff88", // confirmed / canlı / OK
  greenDeep: "#06231a",
  yellow: "#ffcc00", // pending / QoD / uyarı
  yellowDeep: "#2a2200",
  red: "#ff4444", // risk / hata
  blue: "#2f81f7", // sürücü durumu / nötr bilgi
};

// Event tipi → sol-kenar rengi (akış satırları).
export const EVENT_COLOR: Record<string, string> = {
  RISK_ALERT: COLORS.red,
  SPEED_LIMIT_VIOLATION: COLORS.red,
  PLATE_CONFIRMED: COLORS.green,
  PLATE_REJECTED: COLORS.yellow,
  QOD_TRIGGER: COLORS.yellow,
  QOD_RELEASE: COLORS.muted,
  DRIVER_STATE: COLORS.blue,
  DRIVER_LOCKED: COLORS.blue,
  SPEED: COLORS.blue,
  SPEED_LIMIT_DETECTED: COLORS.muted,
  DETECTION_UPDATE: COLORS.muted,
};

// Sürücü ihlali → ikon + okunabilir etiket (DriverState.active_flags()).
export const DRIVER_FLAG: Record<string, { icon: string; label: string }> = {
  phone: { icon: "📱", label: "Telefon" },
  smoking: { icon: "🚬", label: "Sigara" },
  no_seatbelt: { icon: "⚠", label: "Kemersiz" },
  fatigue: { icon: "😴", label: "Yorgunluk" },
};

// Risk bayrağı → okunabilir TR etiket (bilinmeyenler ham gösterilir).
export const RISK_LABEL: Record<string, string> = {
  swerving: "Dikkatsiz sürüş",
  speeding: "Aşırı hız",
  speed_limit_violation: "Hız limiti ihlali",
  high_speed: "Yüksek hız",
  rapid_approach: "Hızlı yaklaşma",
  relative_velocity: "Hızlı yaklaşma",
  tailgating: "Yakın takip",
};

export function riskLabel(flag: string): string {
  return RISK_LABEL[flag] ?? flag.replace(/_/g, " ");
}

// QoD tetik sebebi → TR etiket.
export const QOD_REASON: Record<string, string> = {
  vehicle_approach: "Araç yaklaşıyor",
  speed_anomaly: "Hız anomalisi",
  risk_alert: "Risk uyarısı",
};

export function qodReasonLabel(reason: string): string {
  return QOD_REASON[reason] ?? reason.replace(/_/g, " ");
}
