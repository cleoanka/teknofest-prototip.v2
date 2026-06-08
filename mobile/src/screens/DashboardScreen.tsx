// Ana ekran — canlı tespit event'leri + QoD rozeti.
import React, { useEffect, useRef, useState } from "react";
import { FlatList, StyleSheet, Text, View } from "react-native";

import { AuraEvent, connectEvents, getStatus } from "../api/client";

interface Props {
  phone: string;
}

const TYPE_COLOR: Record<string, string> = {
  RISK_ALERT: "#ff4444",
  PLATE_CONFIRMED: "#00ff88",
  PLATE_REJECTED: "#ffcc00",
  QOD_TRIGGER: "#ffcc00",
  QOD_RELEASE: "#ffcc00",
  DRIVER_STATE: "#2f81f7",
};

function describe(e: AuraEvent): string {
  const p = e.payload as any;
  switch (e.type) {
    case "PLATE_CONFIRMED": return p.value ?? "";
    case "DRIVER_STATE": return (p.flags ?? []).join(", ") || "temiz";
    case "QOD_TRIGGER": return `${p.profile} (${p.reason})`;
    case "RISK_ALERT": return p.rule ?? "risk";
    case "SPEED": return p.value_kmh != null ? `${p.value_kmh} km/h` : "göreli hız";
    default: return "";
  }
}

export default function DashboardScreen({ phone }: Props) {
  const [events, setEvents] = useState<AuraEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [qodActive, setQodActive] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    wsRef.current = connectEvents(
      (e) => {
        setEvents((prev) => [e, ...prev].slice(0, 100));
        if (e.type === "QOD_TRIGGER") setQodActive(true);
        if (e.type === "QOD_RELEASE") setQodActive(false);
      },
      setConnected,
    );
    const poll = setInterval(async () => {
      const s = await getStatus();
      if (s) setQodActive(Number(s.qod_active_sessions) > 0);
    }, 2000);
    return () => {
      wsRef.current?.close();
      clearInterval(poll);
    };
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>◈ AURA</Text>
          <Text style={styles.muted}>{phone}</Text>
        </View>
        <View style={styles.badges}>
          <Text style={[styles.badge, { color: connected ? "#00ff88" : "#8b97a6" }]}>
            {connected ? "● CANLI" : "○ bağlanıyor"}
          </Text>
          {qodActive && <Text style={[styles.badge, styles.qod]}>QoD AKTİF</Text>}
        </View>
      </View>

      <FlatList
        data={events}
        keyExtractor={(e) => e.event_id}
        ListEmptyComponent={<Text style={styles.empty}>Tespit bekleniyor… (inference :8080 çalışıyor mu?)</Text>}
        renderItem={({ item }) => (
          <View style={[styles.row, { borderLeftColor: TYPE_COLOR[item.type] ?? "#8b97a6" }]}>
            <Text style={styles.type}>{item.type}</Text>
            <Text style={styles.meta}>ID:{item.track_id} · {describe(item)}</Text>
          </View>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0d1117", paddingTop: 50, paddingHorizontal: 14 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 14 },
  title: { color: "#e6edf3", fontSize: 24, fontWeight: "700" },
  muted: { color: "#8b97a6", fontSize: 12 },
  badges: { alignItems: "flex-end", gap: 6 },
  badge: { fontFamily: "monospace", fontSize: 12 },
  qod: { color: "#2a2200", backgroundColor: "#ffcc00", paddingHorizontal: 8, paddingVertical: 2, borderRadius: 10, overflow: "hidden" },
  row: { backgroundColor: "#161b22", borderLeftWidth: 3, borderRadius: 6, padding: 10, marginBottom: 6 },
  type: { color: "#e6edf3", fontWeight: "700", fontSize: 13 },
  meta: { color: "#8b97a6", fontSize: 12, marginTop: 2 },
  empty: { color: "#8b97a6", textAlign: "center", marginTop: 40 },
});
