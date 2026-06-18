// AURA mobil — tek araç tespit kartı. Annotation track'inden tüm rozetleri birleştirir.
import React from "react";
import { StyleSheet, Text, View } from "react-native";

import type { AnnotationTrack } from "../api/types";
import {
  DriverFlags,
  PlateBadge,
  QodBadge,
  RiskFlags,
  SpeedBadge,
} from "./Badges";
import { COLORS } from "./theme";

function VehicleCardBase({ track }: { track: AnnotationTrack }) {
  const hasRisk = track.risk_flags.length > 0 || track.swerving;
  // Risk varsa kart sol-kenarı kırmızı; QoD aktifse sarı; aksi yeşil (sağlıklı tespit).
  const accent = hasRisk ? COLORS.red : track.qod_active ? COLORS.yellow : COLORS.green;

  return (
    <View style={[styles.card, { borderLeftColor: accent }]}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>
          {(track.cls || "araç").toUpperCase()} <Text style={styles.id}>#{track.track_id}</Text>
        </Text>
        {track.qod_active ? <QodBadge /> : null}
      </View>

      <View style={styles.badgeRow}>
        <PlateBadge track={track} />
        <SpeedBadge
          kmh={track.speed_kmh}
          calibrated={track.speed_calibrated}
          approaching={track.relative_velocity_flag}
        />
      </View>

      <View style={styles.badgeRow}>
        <DriverFlags flags={track.driver} />
      </View>

      <RiskFlags flags={track.risk_flags} swerving={track.swerving} />
    </View>
  );
}

// track aynı kaldıkça yeniden render etme (FlatList performansı).
export const VehicleCard = React.memo(
  VehicleCardBase,
  (a, b) =>
    a.track.track_id === b.track.track_id &&
    a.track.plate === b.track.plate &&
    a.track.plate_status === b.track.plate_status &&
    a.track.speed_kmh === b.track.speed_kmh &&
    a.track.qod_active === b.track.qod_active &&
    a.track.swerving === b.track.swerving &&
    a.track.driver.join() === b.track.driver.join() &&
    a.track.risk_flags.join() === b.track.risk_flags.join(),
);

const styles = StyleSheet.create({
  card: {
    backgroundColor: COLORS.card,
    borderLeftWidth: 4,
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    gap: 8,
  },
  headerRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  title: { color: COLORS.text, fontSize: 15, fontWeight: "700" },
  id: { color: COLORS.muted, fontSize: 13, fontWeight: "500" },
  badgeRow: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 6 },
});
