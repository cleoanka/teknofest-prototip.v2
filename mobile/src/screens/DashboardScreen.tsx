// RoadGuard mobil — canlı operatör paneli.
//  • WS /stream/annotations → araç KARTLARI (plaka/sürücü/risk/hız/QoD canlı durumu)
//  • WS /stream/events      → aktivite AKIŞI + QoD histerezis tetiği
//  • GET /stream/status     → bağlantı/QoD-oturum senkronu (poll)
//  • PATCH /config          → QoD-tetikli yüksek kalite talebi (useQod içinde)
//  • GET /stream/video      → MJPEG canlı görüntü (LiveVideo)
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FlatList, StyleSheet, Text, TouchableOpacity, View } from "react-native";

import {
  connectAnnotations,
  connectEvents,
  getStatus,
  type AnnotationTrack,
  type AuraEvent,
  type AuraSocket,
} from "../api/client";
import { useQod } from "../hooks/useQod";
import { EventRow } from "../ui/EventRow";
import LiveVideo from "../ui/LiveVideo";
import QodIndicator from "../ui/QodIndicator";
import { COLORS } from "../ui/theme";
import { VehicleCard } from "../ui/VehicleCard";

interface Props {
  phone: string;
}

type Tab = "vehicles" | "feed";

// Bir track son STALE_FRAMES kare boyunca görünmezse listeden düşür (giden araç).
const STALE_FRAMES = 60;

export default function DashboardScreen({ phone }: Props) {
  const [tracks, setTracks] = useState<AnnotationTrack[]>([]);
  const [events, setEvents] = useState<AuraEvent[]>([]);
  const [annOpen, setAnnOpen] = useState(false);
  const [evOpen, setEvOpen] = useState(false);
  const [tab, setTab] = useState<Tab>("vehicles");

  const qod = useQod();
  const annRef = useRef<AuraSocket | null>(null);
  const evRef = useRef<AuraSocket | null>(null);
  // track_id → en son görüldüğü frame_id (eskiyenleri ayıklamak için).
  const lastSeen = useRef<Map<number, number>>(new Map());

  const handleEvent = useCallback(
    (e: AuraEvent) => {
      setEvents((prev) => [e, ...prev].slice(0, 120));
      qod.onEvent(e); // QoD histerezis makinesi
    },
    [qod],
  );

  useEffect(() => {
    annRef.current = connectAnnotations((frame) => {
      const seen = lastSeen.current;
      for (const t of frame.tracks) seen.set(t.track_id, frame.frame_id);
      // Eskiyen track_id'leri düşür, kalanları gelen kareyle güncelle.
      const fresh = frame.tracks.filter(
        (t) => frame.frame_id - (seen.get(t.track_id) ?? frame.frame_id) < STALE_FRAMES,
      );
      // Bu karede gelmeyen ama hâlâ taze track'leri koru (kısa kayıp toleransı).
      setTracks((prev) => {
        const byId = new Map<number, AnnotationTrack>();
        for (const t of prev) {
          if (frame.frame_id - (seen.get(t.track_id) ?? 0) < STALE_FRAMES) byId.set(t.track_id, t);
        }
        for (const t of fresh) byId.set(t.track_id, t);
        return Array.from(byId.values()).sort((a, b) => a.track_id - b.track_id);
      });
    }, setAnnOpen);

    evRef.current = connectEvents(handleEvent, setEvOpen);

    const poll = setInterval(async () => {
      const s = await getStatus();
      if (s) qod.syncSessions(s.qod_active_sessions ?? 0);
    }, 2500);

    return () => {
      annRef.current?.close();
      evRef.current?.close();
      clearInterval(poll);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const connected = annOpen || evOpen;
  const riskCount = useMemo(
    () => tracks.filter((t) => t.risk_flags.length > 0 || t.swerving).length,
    [tracks],
  );

  return (
    <View style={styles.container}>
      {/* Başlık + canlı/QoD rozetleri */}
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>◈ RoadGuard</Text>
          <Text style={styles.muted}>{phone} · NV doğrulandı ✓</Text>
        </View>
        <View style={styles.badges}>
          <Text style={[styles.live, { color: connected ? COLORS.green : COLORS.muted }]}>
            {connected ? "● CANLI" : "○ bağlanıyor"}
          </Text>
          <Text style={styles.muted}>
            {tracks.length} araç · {riskCount} risk
          </Text>
        </View>
      </View>

      {/* Canlı görüntü (MJPEG) */}
      <LiveVideo bbox height={190} />

      {/* QoD durum göstergesi (histerezis) */}
      <QodIndicator phase={qod.phase} reason={qod.reason} activeSessions={tracks.filter((t) => t.qod_active).length} />

      {/* Sekmeler */}
      <View style={styles.tabs}>
        <TabButton label={`Araçlar (${tracks.length})`} active={tab === "vehicles"} onPress={() => setTab("vehicles")} />
        <TabButton label={`Akış (${events.length})`} active={tab === "feed"} onPress={() => setTab("feed")} />
      </View>

      {tab === "vehicles" ? (
        <FlatList
          data={tracks}
          keyExtractor={(t) => String(t.track_id)}
          renderItem={({ item }) => <VehicleCard track={item} />}
          ListEmptyComponent={
            <Text style={styles.empty}>
              Araç tespiti bekleniyor…{"\n"}inference_api (:8080) çalışıyor ve stream başlatıldı mı?
            </Text>
          }
          contentContainerStyle={styles.listPad}
        />
      ) : (
        <FlatList
          data={events}
          keyExtractor={(e) => e.event_id}
          renderItem={({ item }) => <EventRow event={item} />}
          ListEmptyComponent={<Text style={styles.empty}>Event bekleniyor…</Text>}
          contentContainerStyle={styles.listPad}
        />
      )}
    </View>
  );
}

function TabButton({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <TouchableOpacity style={[styles.tab, active && styles.tabActive]} onPress={onPress}>
      <Text style={[styles.tabText, active && styles.tabTextActive]}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg, paddingTop: 50, paddingHorizontal: 14 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 },
  title: { color: COLORS.text, fontSize: 24, fontWeight: "700" },
  muted: { color: COLORS.muted, fontSize: 12 },
  badges: { alignItems: "flex-end", gap: 4 },
  live: { fontFamily: "monospace", fontSize: 13, fontWeight: "700" },
  tabs: { flexDirection: "row", gap: 8, marginBottom: 10 },
  tab: { flex: 1, paddingVertical: 8, borderRadius: 8, backgroundColor: COLORS.card, alignItems: "center" },
  tabActive: { backgroundColor: COLORS.cardAlt, borderColor: COLORS.green, borderWidth: 1 },
  tabText: { color: COLORS.muted, fontSize: 13, fontWeight: "600" },
  tabTextActive: { color: COLORS.green },
  listPad: { paddingBottom: 30 },
  empty: { color: COLORS.muted, textAlign: "center", marginTop: 40, lineHeight: 20 },
});
