// RoadGuard mobil — tespit rozetleri: plaka durumu, QoD, sürücü ihlali, risk, hız.
import React from "react";
import { StyleSheet, Text, View } from "react-native";

import type { AnnotationTrack } from "../api/types";
import { COLORS, DRIVER_FLAG, riskLabel } from "./theme";

// Genel küçük etiket primitifi.
export function Pill({
  text,
  fg,
  bg,
  border,
}: {
  text: string;
  fg: string;
  bg?: string;
  border?: string;
}) {
  return (
    <View
      style={[
        styles.pill,
        bg ? { backgroundColor: bg } : null,
        border ? { borderColor: border, borderWidth: 1 } : null,
      ]}
    >
      <Text style={[styles.pillText, { color: fg }]} numberOfLines={1}>
        {text}
      </Text>
    </View>
  );
}

// Plaka rozeti — CONFIRMED yeşil / pending sarı / rejected kırmızı.
export function PlateBadge({ track }: { track: AnnotationTrack }) {
  const value = track.plate ?? track.plate_partial ?? null;
  if (track.plate_status === "confirmed" && track.plate) {
    return <Pill text={`✓ ${track.plate}`} fg={COLORS.greenDeep} bg={COLORS.green} />;
  }
  if (track.plate_status === "rejected") {
    return <Pill text="plaka ✗" fg={COLORS.red} border={COLORS.red} />;
  }
  // pending — kısmi aday varsa onu sarı göster
  return (
    <Pill
      text={value ? `${value}…` : "plaka okunuyor…"}
      fg={COLORS.yellowDeep}
      bg={COLORS.yellow}
    />
  );
}

// QoD-aktif rozeti — yüksek kalite modu bu araç için açık.
export function QodBadge() {
  return <Pill text="QoD ✦" fg={COLORS.yellowDeep} bg={COLORS.yellow} />;
}

// Sürücü ihlal ikonları (sigara/telefon/kemersiz/yorgunluk).
export function DriverFlags({ flags }: { flags: string[] }) {
  if (!flags.length) {
    return <Pill text="sürücü temiz" fg={COLORS.muted} border={COLORS.border} />;
  }
  return (
    <View style={styles.row}>
      {flags.map((f) => {
        const meta = DRIVER_FLAG[f] ?? { icon: "•", label: f };
        return (
          <Pill
            key={f}
            text={`${meta.icon} ${meta.label}`}
            fg={COLORS.yellowDeep}
            bg={COLORS.yellow}
          />
        );
      })}
    </View>
  );
}

// Risk uyarısı — kırmızı (swerving dahil). Boşsa hiçbir şey çizmez.
export function RiskFlags({ flags, swerving }: { flags: string[]; swerving?: boolean }) {
  const all = [...flags];
  if (swerving && !all.includes("swerving")) all.unshift("swerving");
  if (!all.length) return null;
  return (
    <View style={styles.row}>
      {all.map((f) => (
        <Pill key={f} text={`⚠ ${riskLabel(f)}`} fg="#fff" bg={COLORS.red} />
      ))}
    </View>
  );
}

// Hız rozeti — kalibre değilse "~" ön eki (göreli/tahmini).
export function SpeedBadge({
  kmh,
  calibrated,
  approaching,
}: {
  kmh: number | null;
  calibrated?: boolean;
  approaching?: boolean;
}) {
  if (kmh == null) {
    return approaching ? (
      <Pill text="↗ yaklaşıyor" fg={COLORS.blue} border={COLORS.blue} />
    ) : null;
  }
  const txt = `${calibrated ? "" : "~"}${Math.round(kmh)} km/h`;
  return <Pill text={txt} fg={COLORS.text} bg={COLORS.cardAlt} border={COLORS.border} />;
}

const styles = StyleSheet.create({
  pill: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 12,
    overflow: "hidden",
  },
  pillText: { fontSize: 12, fontWeight: "600" },
  row: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
});
