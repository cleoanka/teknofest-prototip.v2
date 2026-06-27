// WS tüketici — /stream/events (RoadGuardEvent) ve /stream/annotations (AnnotationFrame).
// Kopmada otomatik yeniden bağlanır.

export class EventStream {
  constructor({ onEvent, onAnnotation }) {
    this.onEvent = onEvent;
    this.onAnnotation = onAnnotation;
    this._closed = false;
  }

  connect() {
    this._open("/stream/events", (d) => this.onEvent(d));
    this._open("/stream/annotations", (d) => this.onAnnotation(d));
  }

  _open(path, handler) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}${path}`);
    ws.onmessage = (e) => {
      try { handler(JSON.parse(e.data)); } catch (_) { /* yok say */ }
    };
    ws.onclose = () => {
      if (!this._closed) setTimeout(() => this._open(path, handler), 1500);
    };
    ws.onerror = () => ws.close();
  }

  close() {
    this._closed = true;
  }
}
