// API adresleri — EXPO_PUBLIC_* env ile override edilir.
// Emülatör/cihazdan erişim için localhost yerine makinenizin LAN IP'sini kullanın
// (Android emülatör: 10.0.2.2). Örnek: EXPO_PUBLIC_API_URL=http://192.168.1.20:8080

export const API_URL = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8080";
export const NV_URL = process.env.EXPO_PUBLIC_NV_URL ?? "http://localhost:8082";

// mock ↔ gerçek geçişi yalnızca bu adresleri değiştirmekle yapılır (sözleşme aynı).
export const USE_MOCK = (process.env.EXPO_PUBLIC_USE_MOCK ?? "true") === "true";
