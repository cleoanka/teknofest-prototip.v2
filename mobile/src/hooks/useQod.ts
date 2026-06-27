// RoadGuard mobil — QoD histerezis durum makinesi (şartname §3).
//
// Kritik event (QOD_TRIGGER / RISK_ALERT / SPEED_LIMIT_VIOLATION / yaklaşma) gelince
// "active"e geçer ve BİR KEZ yüksek-kalite talep eder (PATCH /config high). Event akışı
// durunca hemen bırakmaz; HİSTEREZİS penceresi boyunca "releasing" kalır, pencere
// dolunca "idle"a düşüp baseline kalite talep eder. Bu, kalite modunun titremesini
// (her karede aç/kapa) önler — gerçek QoD davranışının mobil yansıması.
import { useCallback, useEffect, useRef, useState } from "react";

import { requestQuality } from "../api/client";
import type { RoadGuardEvent } from "../api/types";
import type { QodPhase } from "../ui/QodIndicator";

// QoD'u tetikleyen kritik event tipleri.
const TRIGGER_TYPES = new Set([
  "QOD_TRIGGER",
  "RISK_ALERT",
  "SPEED_LIMIT_VIOLATION",
]);

const RELEASE_TYPES = new Set(["QOD_RELEASE"]);

const HYSTERESIS_MS = 6000; // kritik event durduktan sonra yüksek kalitede kalma süresi

export interface QodController {
  phase: QodPhase;
  reason: string | null;
  // Dashboard her gelen event'i buraya verir.
  onEvent: (e: RoadGuardEvent) => void;
  // status.qod_active_sessions ile senkron (backend gerçeği): 0'a düşünce bırakmayı hızlandırır.
  syncSessions: (n: number) => void;
}

export function useQod(): QodController {
  const [phase, setPhase] = useState<QodPhase>("idle");
  const [reason, setReason] = useState<string | null>(null);
  const phaseRef = useRef<QodPhase>("idle");
  const releaseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const setPhaseBoth = useCallback((p: QodPhase) => {
    phaseRef.current = p;
    setPhase(p);
  }, []);

  const clearTimer = () => {
    if (releaseTimer.current) {
      clearTimeout(releaseTimer.current);
      releaseTimer.current = null;
    }
  };

  // active → releasing → (pencere dolunca) idle + baseline talebi.
  const beginRelease = useCallback(() => {
    if (phaseRef.current === "idle") return;
    setPhaseBoth("releasing");
    clearTimer();
    releaseTimer.current = setTimeout(() => {
      setPhaseBoth("idle");
      setReason(null);
      void requestQuality("baseline");
    }, HYSTERESIS_MS);
  }, [setPhaseBoth]);

  const goActive = useCallback(
    (why: string | null) => {
      const wasActive = phaseRef.current === "active";
      clearTimer();
      if (why) setReason(why);
      setPhaseBoth("active");
      if (!wasActive) void requestQuality("high"); // yalnız geçişte talep (tekrar etme)
    },
    [setPhaseBoth],
  );

  const onEvent = useCallback(
    (e: RoadGuardEvent) => {
      if (TRIGGER_TYPES.has(e.type)) {
        const why =
          (e.payload?.reason as string) ??
          (e.payload?.rule as string) ??
          (e.type === "SPEED_LIMIT_VIOLATION" ? "speed_limit_violation" : null);
        goActive(why);
      } else if (RELEASE_TYPES.has(e.type)) {
        beginRelease();
      }
    },
    [goActive, beginRelease],
  );

  // Backend "0 aktif oturum" derse ve hâlâ active isek bırakmayı başlat (gerçekle hizala).
  const syncSessions = useCallback(
    (n: number) => {
      if (n <= 0 && phaseRef.current === "active") beginRelease();
    },
    [beginRelease],
  );

  useEffect(() => clearTimer, []);

  return { phase, reason, onEvent, syncSessions };
}
