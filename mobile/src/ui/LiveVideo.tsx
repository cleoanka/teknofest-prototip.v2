// RoadGuard mobil — canlı MJPEG video (GET /stream/video).
//
// React Native <Image> MJPEG multipart akışını sürekli oynatmaz; güvenilir canlı
// oynatım için react-native-webview gerekir (HTML <img src=mjpeg> tarayıcıda akar).
// Bağımlılık ŞİŞİRMEMEK için webview OPSİYONEL: kuruluysa otomatik kullanılır,
// kurulu değilse bilgilendirici yer-tutucu gösterilir (kart akışı ana gösterimdir).
//
// Tam canlı video istenirse:  npx expo install react-native-webview
import React from "react";
import { Image, StyleSheet, Text, View } from "react-native";

import { videoUrl } from "../config";
import { COLORS } from "./theme";

// Opsiyonel webview (kuruluysa). require try/catch ile sarılır → tsc/runtime güvenli.
function loadWebView(): React.ComponentType<{ source: { uri: string }; style?: unknown }> | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require("react-native-webview");
    return (mod.WebView ?? mod.default) as never;
  } catch {
    return null;
  }
}

const WebView = loadWebView();

interface Props {
  bbox?: boolean; // true → sunucu bbox çizer (?bbox=true)
  height?: number;
}

export default function LiveVideo({ bbox = true, height = 200 }: Props) {
  const uri = videoUrl(bbox);

  if (WebView) {
    // MJPEG'i tam-genişlik <img> içinde akıt (object-fit ile sığdır).
    const html = `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body{margin:0;background:#000;height:100%}img{width:100%;height:100%;object-fit:contain}</style></head><body><img src="${uri}"></body></html>`;
    return (
      <View style={[styles.frame, { height }]}>
        <WebView source={{ uri: `data:text/html,${encodeURIComponent(html)}` }} style={styles.fill} />
        <View style={styles.tag}>
          <Text style={styles.tagText}>● MJPEG canlı</Text>
        </View>
      </View>
    );
  }

  // Yer-tutucu: webview yokken ilk kareyi <Image> ile denemek (statik) + bilgi.
  return (
    <View style={[styles.frame, styles.placeholder, { height }]}>
      <Image source={{ uri }} style={styles.fill} resizeMode="contain" />
      <View style={styles.overlay}>
        <Text style={styles.phTitle}>📹 Canlı görüntü</Text>
        <Text style={styles.phSub}>
          Akan video için: npx expo install react-native-webview
        </Text>
        <Text style={styles.phSub}>Tespit kartları aşağıda canlıdır.</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  frame: {
    backgroundColor: "#000",
    borderRadius: 10,
    overflow: "hidden",
    marginBottom: 10,
  },
  fill: { flex: 1, width: "100%", height: "100%" },
  placeholder: { alignItems: "center", justifyContent: "center" },
  overlay: {
    position: "absolute",
    alignItems: "center",
    paddingHorizontal: 16,
    gap: 4,
  },
  phTitle: { color: COLORS.text, fontSize: 15, fontWeight: "700" },
  phSub: { color: COLORS.muted, fontSize: 11, textAlign: "center" },
  tag: { position: "absolute", top: 8, left: 8, backgroundColor: "rgba(0,0,0,0.55)", borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2 },
  tagText: { color: COLORS.green, fontSize: 10, fontWeight: "700" },
});
