// AURA mobil — global QoD durum göstergesi (yüksek-kalite modu + histerezis/bırakma).
import React from "react";
import { StyleSheet, Text, View } from "react-native";

import { COLORS, qodReasonLabel } from "./theme";

export type QodPhase = "idle" | "active" | "releasing";

interface Props {
  phase: QodPhase;
  reason?: string | null; // son tetik sebebi (vehicle_approach / speed_anomaly …)
  activeSessions: number; // status.qod_active_sessions
}

// idle  : taban kalite (5G normal)
// active: kritik event → yüksek kalite talep edildi (PATCH /config gönderildi)
// releasing: histerezis penceresi — event durdu ama hemen bırakmıyoruz (titreme önleme)
export default function QodIndicator({ phase, reason, activeSessions }: Props) {
  const cfg = {
    idle: { bg: COLORS.cardAlt, fg: COLORS.muted, dot: COLORS.muted, label: "QoD: taban kalite" },
    active: { bg: COLORS.yellow, fg: COLORS.yellowDeep, dot: COLORS.yellowDeep, label: "QoD AKTİF — yüksek kalite" },
    releasing: { bg: COLORS.cardAlt, fg: COLORS.yellow, dot: COLORS.yellow, label: "QoD bırakılıyor (histerezis)" },
  }[phase];

  const sub =
    phase === "active"
      ? `${reason ? qodReasonLabel(reason) + " · " : ""}${activeSessions} oturum`
      : phase === "releasing"
        ? "kritik event durdu — kademeli düşürme"
        : "tetik bekleniyor";

  return (
    <View style={[styles.box, { backgroundColor: cfg.bg }]}>
      <View style={[styles.dot, { backgroundColor: cfg.dot }]} />
      <View style={styles.txt}>
        <Text style={[styles.label, { color: cfg.fg }]}>{cfg.label}</Text>
        <Text style={[styles.sub, { color: cfg.fg, opacity: 0.8 }]}>{sub}</Text>
      </View>
      {phase === "active" ? <Text style={[styles.spark, { color: cfg.fg }]}>✦</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    flexDirection: "row",
    alignItems: "center",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginBottom: 10,
    gap: 10,
  },
  dot: { width: 10, height: 10, borderRadius: 5 },
  txt: { flex: 1 },
  label: { fontSize: 13, fontWeight: "700" },
  sub: { fontSize: 11, marginTop: 1 },
  spark: { fontSize: 18, fontWeight: "700" },
});
