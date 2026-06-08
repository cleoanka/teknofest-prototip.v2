// Sessiz giriş ekranı — NV mock ile otomatik doğrulama (SMS/OTP yok).
import React, { useEffect, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from "react-native";

import { verifyNumber } from "../api/client";

interface Props {
  onLogin: (phone: string) => void;
}

// Demo SIM kimliği — gerçek cihazda operatör NV API'si SIM token'ı sağlar.
const DEMO_PHONE = "+905551112233";
const DEMO_SIM_TOKEN = "demo-sim-token";

export default function LoginScreen({ onLogin }: Props) {
  const [status, setStatus] = useState<"checking" | "failed">("checking");

  async function attempt() {
    setStatus("checking");
    const ok = await verifyNumber(DEMO_PHONE, DEMO_SIM_TOKEN);
    if (ok) onLogin(DEMO_PHONE);
    else setStatus("failed");
  }

  useEffect(() => {
    attempt();
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.logo}>◈ AURA</Text>
      <Text style={styles.subtitle}>5G &amp; YZ · Akıllı Yol Güvenliği</Text>
      {status === "checking" ? (
        <View style={styles.center}>
          <ActivityIndicator color="#00ff88" size="large" />
          <Text style={styles.muted}>Sessiz doğrulama (Number Verification)…</Text>
        </View>
      ) : (
        <View style={styles.center}>
          <Text style={styles.error}>Doğrulama başarısız. NV servisi (:8082) çalışıyor mu?</Text>
          <TouchableOpacity style={styles.btn} onPress={attempt}>
            <Text style={styles.btnText}>Tekrar Dene</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0d1117", alignItems: "center", justifyContent: "center" },
  logo: { color: "#e6edf3", fontSize: 44, fontWeight: "700" },
  subtitle: { color: "#8b97a6", fontSize: 14, marginTop: 6, marginBottom: 40 },
  center: { alignItems: "center", gap: 14 },
  muted: { color: "#8b97a6", marginTop: 12 },
  error: { color: "#ff4444", textAlign: "center", paddingHorizontal: 30 },
  btn: { backgroundColor: "#00ff88", paddingHorizontal: 24, paddingVertical: 12, borderRadius: 8 },
  btnText: { color: "#06231a", fontWeight: "700" },
});
