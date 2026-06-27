// RoadGuard mobil — sessiz giriş (Number Verification). SMS/OTP YOK.
// Şebeke/SIM bağı (sim_token + TR numarası) operatör NV API'siyle sessizce doğrulanır;
// burada nv_mock (:8082) POST /verify çağrılır. Başarılıysa otomatik Dashboard'a geçilir.
import React, { useEffect, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from "react-native";

import { verifyNumber } from "../api/client";
import { COLORS } from "../ui/theme";

interface Props {
  onLogin: (phone: string) => void;
}

// Demo SIM kimliği — gerçek cihazda SIM token'ı operatör NV SDK'sı/şebeke sağlar.
const DEMO_PHONE = "+905551112233";
const DEMO_SIM_TOKEN = "demo-sim-token";

type Status =
  | { kind: "checking" }
  | { kind: "ok"; latencyMs?: number }
  | { kind: "failed"; error?: string };

export default function LoginScreen({ onLogin }: Props) {
  const [status, setStatus] = useState<Status>({ kind: "checking" });

  async function attempt() {
    setStatus({ kind: "checking" });
    const res = await verifyNumber(DEMO_PHONE, DEMO_SIM_TOKEN);
    if (res.ok) {
      // Kısa "doğrulandı" gösterimi, sonra geçiş (anlatım için).
      setStatus({ kind: "ok", latencyMs: res.latencyMs });
      setTimeout(() => onLogin(DEMO_PHONE), 650);
    } else {
      setStatus({ kind: "failed", error: res.error });
    }
  }

  useEffect(() => {
    attempt();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.logo}>◈ RoadGuard</Text>
      <Text style={styles.subtitle}>5G &amp; YZ · Akıllı Yol Güvenliği</Text>
      <Text style={styles.phone}>{DEMO_PHONE}</Text>

      {status.kind === "checking" && (
        <View style={styles.center}>
          <ActivityIndicator color={COLORS.green} size="large" />
          <Text style={styles.muted}>Sessiz doğrulama (Number Verification)…</Text>
          <Text style={styles.hint}>SMS/OTP yok — şebeke/SIM bağı kontrol ediliyor</Text>
        </View>
      )}

      {status.kind === "ok" && (
        <View style={styles.center}>
          <Text style={styles.ok}>✓ Doğrulandı</Text>
          <Text style={styles.muted}>
            Şebeke doğrulaması {status.latencyMs != null ? `${status.latencyMs} ms` : "tamam"}
          </Text>
        </View>
      )}

      {status.kind === "failed" && (
        <View style={styles.center}>
          <Text style={styles.error}>Doğrulama başarısız.</Text>
          <Text style={styles.hint}>
            {status.error ? `(${status.error}) ` : ""}NV servisi (:8082) çalışıyor ve
            config.ts'teki adres doğru mu?
          </Text>
          <TouchableOpacity style={styles.btn} onPress={attempt}>
            <Text style={styles.btnText}>Tekrar Dene</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.bg, alignItems: "center", justifyContent: "center" },
  logo: { color: COLORS.text, fontSize: 44, fontWeight: "700" },
  subtitle: { color: COLORS.muted, fontSize: 14, marginTop: 6 },
  phone: { color: COLORS.muted, fontSize: 13, fontFamily: "monospace", marginTop: 4, marginBottom: 40 },
  center: { alignItems: "center", gap: 10, minHeight: 110 },
  muted: { color: COLORS.muted, marginTop: 8 },
  hint: { color: COLORS.muted, fontSize: 12, textAlign: "center", paddingHorizontal: 36, opacity: 0.8 },
  ok: { color: COLORS.green, fontSize: 22, fontWeight: "700" },
  error: { color: COLORS.red, textAlign: "center", paddingHorizontal: 30, fontWeight: "600" },
  btn: { backgroundColor: COLORS.green, paddingHorizontal: 24, paddingVertical: 12, borderRadius: 8, marginTop: 6 },
  btnText: { color: COLORS.greenDeep, fontWeight: "700" },
});
