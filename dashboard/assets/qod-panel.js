// QoD A/B karşılaştırma paneli — şartnamenin %40 QoD puanı için kanıt aracı.
// GET /eval/results verisiyle Chart.js bar chart (QoD OFF vs ON + delta %).
// [Eval Çalıştır] → POST /eval/run.

export class QodPanel {
  constructor(canvas, runBtn, label, emptyEl) {
    this.canvas = canvas;
    this.label = label;
    this.empty = emptyEl;
    this.chart = null;
    runBtn.onclick = () => this.run();
  }

  async refresh() {
    let res = null;
    try { res = await fetch("/eval/results").then((r) => r.json()); } catch (_) { /**/ }
    if (!res || res.status === "no_results" || res.status === "error" || !res.metrics) {
      this.empty.classList.remove("hidden");
      this.label.textContent = "Son eval: —";
      return;
    }
    this.empty.classList.add("hidden");
    this.label.textContent = "Son eval: " + (res.timestamp || "şimdi");
    this._render(res.metrics);
  }

  async run() {
    this.label.textContent = "Eval çalışıyor…";
    try {
      await fetch("/eval/run", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ qod_comparison: true }),
      });
    } catch (_) { /**/ }
    // harness arka planda; birkaç kez yokla
    let tries = 0;
    const poll = () => {
      this.refresh();
      if (++tries < 8) setTimeout(poll, 2500);
    };
    setTimeout(poll, 2500);
  }

  _render(metrics) {
    const labels = metrics.map((m) => m.name);
    const off = metrics.map((m) => m.qod_off);
    const on = metrics.map((m) => m.qod_on);
    const deltas = metrics.map((m) => m.delta_pct);
    const data = {
      labels,
      datasets: [
        { label: "QoD OFF", data: off, backgroundColor: "#5b6573" },
        { label: "QoD ON", data: on, backgroundColor: "#00ff88" },
      ],
    };
    const opts = {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#8b97a6" } },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const i = items[0].dataIndex;
              const d = deltas[i];
              return "Δ " + (d >= 0 ? "+" : "") + d + "%";
            },
          },
        },
      },
      scales: {
        x: { ticks: { color: "#8b97a6" }, grid: { color: "#2a3340" } },
        y: { ticks: { color: "#8b97a6" }, grid: { color: "#2a3340" } },
      },
    };
    if (this.chart) {
      this.chart.data = data;
      this.chart.update();
    } else {
      this.chart = new Chart(this.canvas, { type: "bar", data, options: opts });
    }
  }
}
