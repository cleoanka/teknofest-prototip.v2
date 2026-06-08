// AURA mobil giriş noktası. NV doğrulaması geçince Dashboard'a geçer.
import { StatusBar } from "expo-status-bar";
import React, { useState } from "react";

import DashboardScreen from "./src/screens/DashboardScreen";
import LoginScreen from "./src/screens/LoginScreen";

export default function App() {
  const [phone, setPhone] = useState<string | null>(null);
  return (
    <>
      <StatusBar style="light" />
      {phone ? <DashboardScreen phone={phone} /> : <LoginScreen onLogin={setPhone} />}
    </>
  );
}
