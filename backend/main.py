"""Run with:  uvicorn backend.main:create_app --factory --port 8000

A factory (not a module-level ``app``) so importing this file never creates a database by accident
and tests can build an app with their own settings.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import __version__
from backend.api.routes import api, public
from backend.core.config import Settings
from incidents.store import IncidentStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = IncidentStore(settings.db_url)
    store.create_schema()          # idempotent; done here so a bare TestClient(app) works too

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        store.dispose()

    app = FastAPI(title="VisionGuard API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "PATCH"],
                       allow_headers=["X-API-Key", "Content-Type"])
    app.include_router(public)
    app.include_router(api)
    return app
