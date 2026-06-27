// RoadGuard mobil — canlı event akışı satırı (WS /stream/events). Kompakt aktivite günlüğü.
import React from "react";
import { StyleSheet, Text, View } from "react-native";

import type { AuraEvent } from "../api/types";
import { COLORS, EVENT_COLOR, qodReasonLabel, riskLabel } from "./theme";

// Event tipini okunabilir TR satıra çevir (payload sözleşmesi accumulator/qod'dan).
export function describe(e: AuraEvent): string {
  const p = (e.payload ?? {}) as Record<string, any>;
  switch (e.type) {
    case "PLATE_CONFIRMED":
      return `plaka doğrulandı: ${p.value ?? ""}`;
    case "PLATE_REJECTED":
      return `plaka reddedildi (${p.reason ?? "?"})`;
    case "DRIVER_STATE": {
      const flags: string[] = p.flags ?? [];
      return flags.length ? `ihlal: ${flags.join(", ")}` : "sürücü temiz";
    }
    case "DRIVER_LOCKED":
      return `sürücü kilitlendi (id ${p.driver_id ?? "?"})`;
    case "SPEED":
      return p.value_kmh != null ? `${Math.round(p.value_kmh)} km/h` : "göreli hız";
    case "QOD_TRIGGER":
      return `QoD aç → ${p.profile ?? "?"} · ${qodReasonLabel(String(p.reason ?? ""))}`;
    case "QOD_RELEASE":
      return `QoD bırak (${p.profile ?? "?"})`;
    case "RISK_ALERT":
      return `risk: ${riskLabel(String(p.rule ?? "risk"))}`;
    case "SPEED_LIMIT_VIOLATION":
      return `hız ihlali: ${Math.round(p.speed_kmh ?? 0)}/${p.limit_kmh ?? "?"} km/h (+${p.over_by_kmh ?? "?"})`;
    case "SPEED_LIMIT_DETECTED":
      return `tabela: ${p.limit_kmh ?? p.active_speed_limit_kmh ?? "?"} km/h`;
    default:
      return "";
  }
}

function timeOf(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("tr-TR", { hour12: false });
}

function EventRowBase({ event }: { event: AuraEvent }) {
  const color = EVENT_COLOR[event.type] ?? COLORS.muted;
  return (
    <View style={[styles.row, { borderLeftColor: color }]}>
      <View style={styles.head}>
        <Text style={[styles.type, { color }]}>{event.type}</Text>
        <Text style={styles.time}>{timeOf(event.ts)}</Text>
      </View>
      <Text style={styles.meta}>
        {event.track_id >= 0 ? `#${event.track_id} · ` : "sahne · "}
        {describe(event)}
      </Text>
    </View>
  );
}

export const EventRow = React.memo(
  EventRowBase,
  (a, b) => a.event.event_id === b.event.event_id,
);

const styles = StyleSheet.create({
  row: {
    backgroundColor: COLORS.card,
    borderLeftWidth: 3,
    borderRadius: 6,
    paddingVertical: 7,
    paddingHorizontal: 10,
    marginBottom: 5,
  },
  head: { flexDirection: "row", justifyContent: "space-between" },
  type: { fontFamily: "monospace", fontSize: 11, fontWeight: "700" },
  time: { color: COLORS.muted, fontSize: 10, fontFamily: "monospace" },
  meta: { color: COLORS.muted, fontSize: 12, marginTop: 2 },
});
