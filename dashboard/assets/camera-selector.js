// Kamera / kaynak seçici. GET /cameras ile listeyi doldurur; webcam, iPhone
// Continuity, video dosyası ve RTSP URL seçeneklerini sunar. Seçim değişince
// onSelect(source, device) çağrılır (app.js → POST /stream/start).

const ICON = { webcam: "🎥", iphone: "📱", video: "🎞️", rtsp: "🌐" };

export class CameraSelector {
  constructor(root, onSelect) {
    this.root = root;
    this.onSelect = onSelect;
    this.selected = null;
  }

  async init() {
    let data = { cameras: [], rtsp_supported: true };
    try {
      data = await fetch("/cameras").then((r) => r.json());
    } catch (e) {
      /* başsız/izinsiz → boş liste */
    }
    this.root.innerHTML = "";

    for (const cam of data.cameras) {
      const isPhone = /iphone|continuity/i.test(cam.name);
      this._addItem({
        icon: isPhone ? ICON.iphone : ICON.webcam,
        label: `${cam.name} (${cam.width}×${cam.height})`,
        source: String(cam.index),
      });
    }
    // Varsayılan örnek video
    this._addItem({ icon: ICON.video, label: "Örnek video (sentetik)",
      source: "data/samples/ornek.mp4" });
    // Video dosyası path
    this._addInputItem({ icon: ICON.video, label: "Video dosyası",
      placeholder: "/yol/video.mp4" });
    // RTSP URL
    this._addInputItem({ icon: ICON.rtsp, label: "RTSP / IP kamera",
      placeholder: "rtsp://192.168.1.10:8554/stream" });

    // ilk öğeyi seç
    const first = this.root.querySelector(".src-item");
    if (first) first.classList.add("active");
  }

  _select(el, source) {
    this.root.querySelectorAll(".src-item").forEach((i) => i.classList.remove("active"));
    el.classList.add("active");
    this.selected = source;
    if (source) this.onSelect(source);
  }

  _addItem({ icon, label, source }) {
    const el = document.createElement("div");
    el.className = "src-item";
    el.innerHTML = `<span>${icon}</span><span>${label}</span>`;
    el.onclick = () => this._select(el, source);
    this.root.appendChild(el);
  }

  _addInputItem({ icon, label, placeholder }) {
    const el = document.createElement("div");
    el.className = "src-item";
    el.innerHTML = `<span>${icon}</span><input type="text" placeholder="${placeholder}" />`;
    const input = el.querySelector("input");
    input.onkeydown = (e) => {
      if (e.key === "Enter" && input.value.trim()) this._select(el, input.value.trim());
    };
    el.onclick = (e) => { if (e.target !== input) input.focus(); };
    this.root.appendChild(el);
  }
}
