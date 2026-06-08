// Canvas bbox overlay motoru — iki-kanal mimarinin client tarafı.
// MJPEG <img> ham akışı taşır; canvas WS annotation'larından bbox çizer.
// BBox toggle yalnızca canvas'ı temizler/çizer — MJPEG akışı kesilmez, sunucuya gidiş yok.

const DRIVER_ICONS = { phone: "📱", smoking: "🚬", no_seatbelt: "⚠️", fatigue: "😴" };

export class VideoRenderer {
  constructor(img, canvas) {
    this.img = img;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.bbox = true;
    this.latest = null;
    this._loop();
  }

  setBbox(on) {
    this.bbox = on;
    if (!on) this._clear();
  }

  update(anno) {
    this.latest = anno;
  }

  _clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  _loop() {
    const draw = () => {
      const w = this.img.naturalWidth, h = this.img.naturalHeight;
      if (w && (this.canvas.width !== w || this.canvas.height !== h)) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      if (this.bbox && this.latest) this._drawAnno(this.latest);
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
  }

  _drawAnno(anno) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    for (const t of anno.tracks || []) {
      const [x1, y1, x2, y2] = t.bbox;
      const risk = (t.risk_flags || []).length > 0;
      const color = risk ? "#ff4444" : "#00ff88";
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      ctx.font = "bold 14px ui-monospace, monospace";
      let label = "ID" + t.track_id;
      if (t.plate) label += " " + t.plate;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - 18, tw + 8, 18);
      ctx.fillStyle = "#06231a";
      ctx.fillText(label, x1 + 4, y1 - 4);

      const icons = (t.driver || []).map((d) => DRIVER_ICONS[d] || "").join("");
      if (icons) {
        ctx.font = "16px sans-serif";
        ctx.fillText(icons, x1, y2 + 18);
      }
      if (t.qod_active) {
        ctx.fillStyle = "#ffcc00";
        ctx.font = "bold 12px ui-monospace, monospace";
        ctx.fillText("QoD", x2 - 34, y1 - 4);
      }
      if (t.relative_velocity_flag) {
        ctx.fillStyle = "#ffcc00";
        ctx.font = "bold 12px ui-monospace, monospace";
        ctx.fillText("⚡", x2 - 14, y2 - 4);
      }
    }
  }
}
