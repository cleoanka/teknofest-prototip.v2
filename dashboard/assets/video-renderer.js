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

    // --- Trafik tabelaları (sahne-seviyesi; bir araca bağlı değil) ---------- //
    for (const s of anno.signs || []) {
      const [sx1, sy1, sx2, sy2] = s.bbox;
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#33ccff";
      ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
      const lbl = s.speed_limit_kmh != null ? s.speed_limit_kmh + " km/h" : (s.cls || "tabela");
      ctx.font = "bold 13px ui-monospace, monospace";
      const tw = ctx.measureText(lbl).width;
      ctx.fillStyle = "#33ccff";
      ctx.fillRect(sx1, sy1 - 17, tw + 8, 17);
      ctx.fillStyle = "#04222e";
      ctx.fillText(lbl, sx1 + 4, sy1 - 4);
    }

    // --- Aktif hız limiti banner'ı (sol-üst; araç tabelayı geçse de görünür) - //
    const lim = anno.scene && anno.scene.active_speed_limit_kmh;
    if (lim != null) {
      const txt = "LİMİT " + lim;
      ctx.font = "bold 18px ui-monospace, monospace";
      const w = ctx.measureText(txt).width + 16;
      ctx.fillStyle = "#33ccff";
      ctx.fillRect(8, 8, w, 26);
      ctx.fillStyle = "#04222e";
      ctx.fillText(txt, 16, 27);
    }
  }
}
