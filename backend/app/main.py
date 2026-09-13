"""Ulazna točka FastAPI aplikacije.

Pokretanje u razvoju:
    uvicorn app.main:app --reload --port 8000

Dokumentacija API-ja se generira sama: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.admin import router as admin_router
from app.api.routes import router as api_router
from app.config import BASE_DIR, settings
from app.db import init_db
from app.services.scheduler import shutdown_scheduler, start_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

FRONTEND_BUILD = BASE_DIR.parent / "build"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Baza: %s", settings.sqlalchemy_url.split("@")[-1])
    init_db()
    start_scheduler()
    yield
    shutdown_scheduler()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Prediktivni sustav za analizu sportskih događaja. "
        "Podaci se prikupljaju s javno dostupnog API-ja, obrađuju u Pythonu "
        "i poslužuju kroz REST sučelje."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(admin_router)


# ─── Posluživanje React builda (jedan proces, jedan port) ───
if FRONTEND_BUILD.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_BUILD / "static")),
        name="static",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        candidate = FRONTEND_BUILD / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_BUILD / "index.html")

else:

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "app": settings.app_name,
            "docs": "/docs",
            "napomena": "React build nije pronaden. Pokreni 'npm run build' u korijenu projekta.",
        }


def run() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=False)


if __name__ == "__main__":
    run()
