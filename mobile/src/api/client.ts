// AURA mobil API istemcisi — NV doğrulama + event WS + kaynak kontrolü.
import { API_URL, NV_URL } from "../config";

export interface AuraEvent {
  event_id: string;
  ts: number;
  track_id: number;
  type: string;
  payload: Record<string, unknown>;
  source: string;
}

// Sessiz numara doğrulama (NV mock). SMS/OTP yok — SIM/şebeke bağı kontrol edilir.
export async function verifyNumber(phoneNumber: string, simToken: string): Promise<boolean> {
  try {
    const res = await fetch(`${NV_URL}/verify`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ phone_number: phoneNumber, sim_token: simToken }),
    });
    const data = await res.json();
    return Boolean(data.verified);
  } catch {
    return false;
  }
}

// Canlı event akışı (WS /stream/events).
export function connectEvents(onEvent: (e: AuraEvent) => void, onState?: (open: boolean) => void): WebSocket {
  const wsUrl = API_URL.replace(/^http/, "ws") + "/stream/events";
  const ws = new WebSocket(wsUrl);
  ws.onopen = () => onState?.(true);
  ws.onclose = () => onState?.(false);
  ws.onmessage = (m) => {
    try {
      onEvent(JSON.parse(m.data as string) as AuraEvent);
    } catch {
      /* yok say */
    }
  };
  return ws;
}

export async function getStatus(): Promise<Record<string, unknown> | null> {
  try {
    return await fetch(`${API_URL}/stream/status`).then((r) => r.json());
  } catch {
    return null;
  }
}

// Inference kaynağını değiştir (opsiyonel — dashboard'la aynı endpoint).
export async function setSource(source: string): Promise<void> {
  try {
    await fetch(`${API_URL}/stream/start`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ source }),
    });
  } catch {
    /* yok say */
  }
}
