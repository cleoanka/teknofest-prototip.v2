// AURA Dashboard orkestratörü. Modülleri kurar, WS'leri bağlar, header durumunu
// yoklar, track kartlarını ve event log'u günceller.
import { CameraSelector } from "/assets/camera-selector.js";
import { VideoRenderer } from "/assets/video-renderer.js";
import { EventStream } from "/assets/event-stream.js";
import { QodPanel } from "/assets/qod-panel.js";

const $ = (s) => document.querySelector(s);
const feed = $("#video-feed");

// --- Tema ---------------------------------------------------------------- //
$("#theme-toggle").onclick = () => {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
};

// --- Video + bbox toggle ------------------------------------------------- //
const renderer = new VideoRenderer(feed, $("#overlay"));
const bboxBtn = $("#bbox-toggle");
bboxBtn.onclick = () => {
  const on = bboxBtn.classList.toggle("active");
  bboxBtn.textContent = "BBox: " + (on ? "ON" : "OFF");
  renderer.setBbox(on); // sadece canvas; MJPEG kesilmez
};

// --- Kalite talebi (conf eşiğini düşür = daha hassas tespit) -------------- //
$("#quality-btn").onclick = async () => {
  await fetch("/config", {
    method: "PATCH", headers: { "content-type": "application/json" },
    body: JSON.stringify({ conf_threshold: 0.25 }),
  }).catch(() => {});
  $("#quality-btn").textContent = "Kalite ✓";
  setTimeout(() => ($("#quality-btn").textContent = "Kalite Talebi"), 1500);
};

// --- Track kartları (son annotation karesinden) -------------------------- //
const DRIVER_ICONS = { phone: "📱", smoking: "🚬", no_seatbelt: "⚠️", fatigue: "😴" };
function renderTracks(tracks) {
  $("#chip-tracks").textContent = "Araç: " + tracks.length;
  const wrap = $("#track-cards");
  wrap.innerHTML = "";
  for (const t of tracks) {
    const risk = (t.risk_flags || []).length > 0;
    const card = document.createElement("div");
    card.className = "tcard" + (risk ? " risk" : "");
    const icons = (t.driver || []).map((d) => DRIVER_ICONS[d] || "").join(" ");
    const plateBadge = t.plate
      ? `<span class="badge plate-confirmed">${t.plate}</span>`
      : `<span class="badge plate-pending">bekliyor</span>`;
    const speed = t.speed_kmh != null ? t.speed_kmh + " km/h"
      : (t.relative_velocity_flag ? "⚡ göreli" : "—");
    card.innerHTML = `
      <div class="tid">ID:${t.track_id} <small>${t.cls || ""}</small></div>
      <div class="row"><span>plaka</span> ${plateBadge}</div>
      <div class="row"><span>hız</span> <b>${speed}</b></div>
      <div class="row"><span>sürücü</span> <b>${icons || "—"}</b></div>
      ${t.qod_active ? `<div class="row"><span>QoD</span> <span class="badge qod">AKTİF</span></div>` : ""}`;
    card.onclick = () => openModal(t.track_id);
    wrap.appendChild(card);
  }
}

// --- Event log ----------------------------------------------------------- //
let eventCount = 0;
function addEvent(ev) {
  eventCount++;
  $("#event-count").textContent = eventCount;
  const ul = $("#event-list");
  const li = document.createElement("li");
  li.className = "ev-" + ev.type;
  const ts = new Date((ev.ts || Date.now() / 1000) * 1000);
  const hh = ts.toLocaleTimeString("tr-TR", { hour12: false }) +
    "." + String(ts.getMilliseconds()).padStart(3, "0");
  const desc = describe(ev);
  li.innerHTML = `<span class="ev-time">${hh}</span> <span class="ev-type">${ev.type}</span> ID:${ev.track_id} → ${desc}`;
  ul.prepend(li);
  while (ul.children.length > 50) ul.removeChild(ul.lastChild);
}
function describe(ev) {
  const p = ev.payload || {};
  switch (ev.type) {
    case "PLATE_CONFIRMED": return p.value || "";
    case "PLATE_REJECTED": return p.reason || "konsensüs yok";
    case "DRIVER_STATE": return (p.flags || []).join(", ") || "temiz";
    case "SPEED": return p.value_kmh != null ? p.value_kmh + " km/h" : (p.relative_velocity_flag ? "göreli hız ⚡" : "—");
    case "QOD_TRIGGER": return `${p.profile} (${p.reason})`;
    case "QOD_RELEASE": return p.profile || "";
    case "RISK_ALERT": return p.rule || "risk";
    case "SPEED_LIMIT_DETECTED": return "limit " + (p.speed_limit_kmh != null ? p.speed_limit_kmh + " km/h" : "?");
    case "SPEED_LIMIT_VIOLATION":
      return `${p.speed_kmh != null ? p.speed_kmh : "?"} km/h / limit ${p.limit_kmh != null ? p.limit_kmh : "?"}` +
        (p.plate ? ` (${p.plate})` : "");
    default: return p.cls || "";
  }
}

// --- Track detay modalı -------------------------------------------------- //
async function openModal(tid) {
  try {
    const rec = await fetch("/tracks/" + tid).then((r) => r.json());
    $("#modal-body").textContent = JSON.stringify(rec, null, 2);
    $("#track-modal").classList.remove("hidden");
  } catch (_) { /**/ }
}
$("#modal-close").onclick = () => $("#track-modal").classList.add("hidden");
$("#track-modal").onclick = (e) => { if (e.target.id === "track-modal") $("#track-modal").classList.add("hidden"); };

// --- WS akışları --------------------------------------------------------- //
const stream = new EventStream({
  onEvent: addEvent,
  onAnnotation: (anno) => { renderer.update(anno); renderTracks(anno.tracks || []); },
});
stream.connect();

// --- Kaynak seçici ------------------------------------------------------- //
async function startSource(source) {
  await fetch("/stream/start", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ source, bbox_overlay: true }),
  }).catch(() => {});
  feed.src = "/stream/video?bbox=false&_=" + Date.now();
}
new CameraSelector($("#source-list"), startSource).init();

// İlk yükleme: server autostart edebilir → ham akışı bağla
feed.src = "/stream/video?bbox=false";

// --- Header durum yoklama ------------------------------------------------ //
setInterval(async () => {
  try {
    const s = await fetch("/stream/status").then((r) => r.json());
    $("#chip-fps").textContent = "FPS: " + (s.fps || 0);
    $("#chip-rec").classList.toggle("live", !!s.running);
    $("#chip-qod").textContent = "QoD: " + (s.qod_active_sessions > 0 ? "ACTIVE" : "idle");
  } catch (_) { /**/ }
}, 1000);

// --- QoD A/B paneli ------------------------------------------------------ //
const qod = new QodPanel($("#qod-chart"), $("#eval-run"), $("#eval-label"), $("#qod-empty"));
qod.refresh();
