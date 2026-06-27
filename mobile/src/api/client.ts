// RoadGuard mobil API istemcisi — NV doğrulama + event/annotation WS + QoD config + status.
// Sözleşme tipleri ./types.ts'tedir (roadguard/schema.py ile birebir).
import { API_URL, NV_URL, WS_ANNOTATIONS_URL, WS_EVENTS_URL } from "../config";
import type {
  AnnotationFrame,
  RoadGuardEvent,
  StreamStatus,
  VerifyResponse,
} from "./types";

export type {
  AnnotationFrame,
  AnnotationTrack,
  RoadGuardEvent,
  PlateStatus,
  StreamStatus,
} from "./types";

// --------------------------------------------------------------------------- //
// Number Verification — sessiz doğrulama (nv_mock :8082). SMS/OTP YOK.
// SIM/şebeke bağı (sim_token + TR numarası) kontrol edilir; operatör NV API'sinin mock'u.
// --------------------------------------------------------------------------- //
export interface VerifyResult {
  ok: boolean;
  latencyMs?: number;
  error?: string;
}

export async function verifyNumber(
  phoneNumber: string,
  simToken: string,
): Promise<VerifyResult> {
  try {
    const res = await fetch(`${NV_URL}/verify`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ phone_number: phoneNumber, sim_token: simToken }),
    });
    if (!res.ok) return { ok: false, error: `NV HTTP ${res.status}` };
    const data = (await res.json()) as VerifyResponse;
    return { ok: Boolean(data.verified), latencyMs: data.latency_ms };
  } catch (e) {
    return { ok: false, error: (e as Error).message ?? "ağ hatası" };
  }
}

// --------------------------------------------------------------------------- //
// WebSocket yardımcıları — otomatik yeniden bağlanma + temiz kapatma.
// connect* fonksiyonları bir "RoadGuardSocket" döndürür: .close() ile elle kapatılır;
// bağlantı koparsa (manuel kapatma değilse) artan gecikmeyle yeniden dener.
// --------------------------------------------------------------------------- //
export interface RoadGuardSocket {
  close: () => void;
}

function connectJsonWs<T>(
  url: string,
  onMessage: (data: T) => void,
  onState?: (open: boolean) => void,
): RoadGuardSocket {
  let ws: WebSocket | null = null;
  let closedByUser = false;
  let retry = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const open = () => {
    ws = new WebSocket(url);
    ws.onopen = () => {
      retry = 0;
      onState?.(true);
    };
    ws.onmessage = (m) => {
      try {
        onMessage(JSON.parse(m.data as string) as T);
      } catch {
        /* bozuk kare — yok say */
      }
    };
    ws.onerror = () => {
      /* onclose yeniden bağlanmayı yönetir */
    };
    ws.onclose = () => {
      onState?.(false);
      if (closedByUser) return;
      retry += 1;
      const delay = Math.min(1000 * retry, 5000); // 1s,2s,3s… max 5s
      timer = setTimeout(open, delay);
    };
  };

  open();

  return {
    close: () => {
      closedByUser = true;
      if (timer) clearTimeout(timer);
      ws?.close();
    },
  };
}

// Canlı RoadGuardEvent akışı (WS /stream/events).
export function connectEvents(
  onEvent: (e: RoadGuardEvent) => void,
  onState?: (open: boolean) => void,
): RoadGuardSocket {
  return connectJsonWs<RoadGuardEvent>(WS_EVENTS_URL, onEvent, onState);
}

// Kare-başına annotation akışı (WS /stream/annotations) — kartların canlı kaynağı.
export function connectAnnotations(
  onFrame: (f: AnnotationFrame) => void,
  onState?: (open: boolean) => void,
): RoadGuardSocket {
  return connectJsonWs<AnnotationFrame>(WS_ANNOTATIONS_URL, onFrame, onState);
}

// --------------------------------------------------------------------------- //
// REST yardımcıları — status + QoD çözünürlük/kalite talebi.
// --------------------------------------------------------------------------- //
export async function getStatus(): Promise<StreamStatus | null> {
  try {
    const r = await fetch(`${API_URL}/stream/status`);
    if (!r.ok) return null;
    return (await r.json()) as StreamStatus;
  } catch {
    return null;
  }
}

// QoD-tetikli kalite/çözünürlük talebi.
// Şartname §3: kritik event gelince yüksek kaliteye geç. Backend sözleşmesinde
// "çözünürlük" doğrudan parametre değil; en yakın eşdeğer çalışma-zamanı ayarları:
//   • PATCH /config { qod_profile, conf_threshold }  (kalıcı çalışma ayarı)
//   • PATCH /stream/config { conf_threshold, bbox_overlay } (akış ayarı)
// "high" → yüksek-kalite profili + düşük conf eşiği (daha çok delil yakala);
// "baseline" → bırakma (histerezis sonrası). Mock'ta da 200 döner (no-op güvenli).
export type QualityMode = "high" | "baseline";

export async function requestQuality(mode: QualityMode): Promise<boolean> {
  const qod_profile = mode === "high" ? "quality" : "baseline";
  const conf_threshold = mode === "high" ? 0.25 : 0.4;
  try {
    const r = await fetch(`${API_URL}/config`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ qod_profile, conf_threshold }),
    });
    return r.ok;
  } catch {
    return false;
  }
}

// Inference kaynağını başlat/değiştir (opsiyonel — dashboard'la aynı endpoint).
export async function startStream(source?: string): Promise<boolean> {
  try {
    const r = await fetch(`${API_URL}/stream/start`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(source ? { source } : {}),
    });
    return r.ok;
  } catch {
    return false;
  }
}
