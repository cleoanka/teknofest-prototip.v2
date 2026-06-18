// AURA mobil — servis adresleri ve WS/MJPEG URL türetme.
//
// Backend İKİ ayrı servistir (farklı portlar):
//   • inference_api  :8080  → /health /cameras /stream/* (WS+MJPEG) /config (PATCH)
//   • nv_mock        :8082  → /verify  (Number Verification — sessiz doğrulama)
//   (qod_mock :8081 inference_api tarafından dahili kullanılır; mobil dokunmaz.)
//
// CİHAZDAN ERİŞİM: Telefon/emülatör "localhost"u kendi içinde arar; geliştirme
// makinenize ulaşamaz. Bu yüzden gerçek demoda makinenizin LAN IP'sini girin:
//   1) Mac'te IP'yi öğrenin:  ipconfig getifaddr en0   (ör. 192.168.1.20)
//   2) Aşağıdaki DEV_LAN_IP'yi o IP ile değiştirin VEYA env ile override edin:
//        EXPO_PUBLIC_API_URL=http://192.168.1.20:8080
//        EXPO_PUBLIC_NV_URL=http://192.168.1.20:8082
//   Android emülatörü host makineyi 10.0.2.2 olarak görür (iOS sim'de localhost olur).
//
// mock ↔ gerçek geçişi YALNIZCA bu adresleri değiştirmekle yapılır (sözleşme aynı).

// <<< DEMO İÇİN BURAYI MAKİNENİZİN LAN IP'Sİ YAPIN >>> (env verilmezse kullanılır)
const DEV_LAN_IP = "192.168.1.20"; // PLACEHOLDER — kendi IP'nizle değiştirin

export const API_URL =
  process.env.EXPO_PUBLIC_API_URL ?? `http://${DEV_LAN_IP}:8080`;
export const NV_URL =
  process.env.EXPO_PUBLIC_NV_URL ?? `http://${DEV_LAN_IP}:8082`;

export const USE_MOCK = (process.env.EXPO_PUBLIC_USE_MOCK ?? "true") === "true";

// http(s):// → ws(s):// dönüşümü (WS endpoint'leri için).
export function toWsUrl(httpUrl: string): string {
  return httpUrl.replace(/^http(s?):\/\//i, "ws$1://");
}

// inference_api üzerindeki yardımcı URL'ler.
export const WS_EVENTS_URL = `${toWsUrl(API_URL)}/stream/events`;
export const WS_ANNOTATIONS_URL = `${toWsUrl(API_URL)}/stream/annotations`;
// MJPEG canlı görüntü. ?bbox=true → sunucu bbox çizer (mobilde overlay kolaylığı).
export const videoUrl = (bbox = false): string =>
  `${API_URL}/stream/video${bbox ? "?bbox=true" : ""}`;
