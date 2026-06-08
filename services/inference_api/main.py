"""AURA Inference API — FastAPI uygulaması (:8080).

Gerçek YZ mikroservisi: pipeline'ı koşturur, MJPEG + WS yayar, dashboard'u statik
serve eder. Mock'lar (qod_mock :8081, nv_mock :8082) ayrı servislerdir.

Çalıştır: uvicorn services.inference_api.main:app --port 8080
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aura import __version__
from services.inference_api.routers import cameras, config, eval, stream, system, tracks
from services.inference_api.state import StreamManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("aura.api")

DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def create_app(cfg=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.stream.attach_loop(asyncio.get_running_loop())
        app.state.eval_results = None
        if os.environ.get("AURA_AUTOSTART", "1") != "0":
            try:
                app.state.stream.start()
            except Exception as e:  # noqa: BLE001
                log.warning("Otomatik stream başlatılamadı: %s", e)
        yield
        app.state.stream.stop()

    app = FastAPI(
        title="AURA Inference API",
        version=__version__,
        description="5G & YZ Akıllı Yol Güvenliği — gerçek YZ mikroservisi "
        "(araç/plaka/sürücü/hız + QoD). MJPEG + WS iki-kanal akış.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.stream = StreamManager(cfg)

    for r in (system, cameras, stream, tracks, eval, config):
        app.include_router(r.router)

    # Dashboard statik serve (M8). Yoksa API yine tam çalışır.
    if (DASHBOARD_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(DASHBOARD_DIR / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        idx = DASHBOARD_DIR / "index.html"
        if idx.exists():
            return FileResponse(str(idx))
        return {
            "service": "AURA Inference API",
            "version": __version__,
            "docs": "/docs",
            "dashboard": "M8'de eklenir",
        }

    return app


app = create_app()
