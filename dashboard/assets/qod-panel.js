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
    // Delta vurgusu da bir veri serisi olsun (gri/yeşil yanında net görünür).
    const deltaColors = deltas.map((d) => (d >= 0 ? "#2f81f7" : "#ff4444"));
    const data = {
      labels,
      datasets: [
        { label: "QoD OFF", data: off, backgroundColor: "#5b6573" },
        { label: "QoD ON", data: on, backgroundColor: "#00ff88" },
        { label: "Δ (kazanç)", data: deltas, backgroundColor: deltaColors },
      ],
    };
    // Custom inline plugin (vanilla, build yok): her ÇUBUĞUN üstüne değerini yazar;
    // Δ serisi için işaretli yüzde (+/−) — ON/OFF deltası ekranda okunur olur.
    // Renkleri chart'ın kendi (güncel) verisinden okur → güncellemede bayatlamaz.
    const DELTA_INDEX = 2;
    const valueLabels = {
      id: "roadguardValueLabels",
      afterDatasetsDraw(chart) {
        const { ctx } = chart;
        ctx.save();
        ctx.font = "bold 10px ui-monospace, monospace";
        ctx.textAlign = "center";
        chart.data.datasets.forEach((ds, di) => {
          const meta = chart.getDatasetMeta(di);
          if (meta.hidden) return;
          const isDelta = di === DELTA_INDEX;
          const colors = Array.isArray(ds.backgroundColor) ? ds.backgroundColor : null;
          meta.data.forEach((bar, i) => {
            const v = ds.data[i];
            if (v == null) return;
            const txt = isDelta ? (v >= 0 ? "+" : "") + v + "%" : v;
            ctx.fillStyle = isDelta && colors ? colors[i] : "#8b97a6";
            ctx.fillText(txt, bar.x, bar.y - 4);
          });
        });
        ctx.restore();
      },
    };
    const opts = {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#8b97a6" } },
        tooltip: {
          callbacks: {
            // Δ serisi tooltip'i işaretli yüzde; ON/OFF serileri ham değer.
            // Tümü item.raw'dan okunur → chart.update sonrası bayatlamaz.
            label: (item) => {
              if (item.datasetIndex === DELTA_INDEX) {
                const d = item.raw;
                return "Δ kazanç: " + (d >= 0 ? "+" : "") + d + "%";
              }
              return item.dataset.label + ": " + item.raw;
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
      this.chart = new Chart(this.canvas, {
        type: "bar",
        data,
        options: opts,
        plugins: [valueLabels],
      });
    }
  }
}
