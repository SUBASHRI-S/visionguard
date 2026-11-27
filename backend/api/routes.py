from __future__ import annotations

import hmac
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse

from backend import __version__
from backend.api.schemas import (Health, IncidentDetail, IncidentPage, Status, StatusUpdate, detail_from,
                                 summary_from)
from backend.core.config import Settings
from incidents.store import IncidentNotFound, IncidentStore, InvalidStatusChange

EVIDENCE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def get_store(request: Request) -> IncidentStore:
    return request.app.state.store


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def require_api_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = request.app.state.settings.api_key
    if expected is None:
        return
    # constant-time comparison so the key cannot be guessed from response timing
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="missing or wrong API key")


public = APIRouter()
api = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


def _utc(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


@public.get("/health", response_model=Health)
def health(store: Annotated[IncidentStore, Depends(get_store)]) -> Health:
    try:
        store.list_incidents(limit=1)
    except Exception as exc:  # the database is the only dependency; say so plainly
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return Health(status="ok", version=__version__, database="ok")


@api.get("/incidents", response_model=IncidentPage)
def list_incidents(
    store: Annotated[IncidentStore, Depends(get_store)],
    camera_id: str | None = None,
    rule_id: str | None = None,
    status: Status | None = None,
    active: bool | None = None,
    run_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> IncidentPage:
    items, total = store.list_incidents(camera_id=camera_id, rule_id=rule_id, status=status, active=active,
                                        run_id=run_id, since=_utc(since), until=_utc(until), limit=limit, offset=offset)
    return IncidentPage(items=[summary_from(i) for i in items], total=total, limit=limit, offset=offset)


@api.get("/incidents/{incident_id}", response_model=IncidentDetail)
def get_incident(incident_id: int, store: Annotated[IncidentStore, Depends(get_store)]) -> IncidentDetail:
    try:
        return detail_from(store.get(incident_id))
    except IncidentNotFound:
        raise HTTPException(status_code=404, detail="incident not found") from None


@api.patch("/incidents/{incident_id}", response_model=IncidentDetail)
def update_incident(incident_id: int, body: StatusUpdate,
                    store: Annotated[IncidentStore, Depends(get_store)]) -> IncidentDetail:
    try:
        return detail_from(store.set_status(incident_id, body.status, actor=body.actor, note=body.note))
    except IncidentNotFound:
        raise HTTPException(status_code=404, detail="incident not found") from None
    except InvalidStatusChange as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@api.get("/incidents/{incident_id}/evidence/{kind}")
def get_evidence(incident_id: int, kind: str, store: Annotated[IncidentStore, Depends(get_store)],
                 settings: Annotated[Settings, Depends(get_settings)]) -> FileResponse:
    try:
        record = store.get(incident_id)
    except IncidentNotFound:
        raise HTTPException(status_code=404, detail="incident not found") from None
    match = next((e for e in record.evidence if e["kind"] == kind), None)
    not_found = HTTPException(status_code=404, detail="evidence not found")
    if match is None:
        raise not_found
    path = Path(match["path"])
    try:
        resolved = path.resolve(strict=True)
        root = settings.evidence_root.resolve()
    except (OSError, RuntimeError):
        raise not_found from None
    # only serve files that really are inside the evidence folder, and only pictures
    if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.suffix.lower() not in EVIDENCE_TYPES:
        raise not_found
    return FileResponse(resolved, media_type=EVIDENCE_TYPES[resolved.suffix.lower()])


@api.get("/stats")
def stats(store: Annotated[IncidentStore, Depends(get_store)], days: Annotated[int, Query(ge=1, le=366)] = 14) -> dict:
    return store.stats(days=days)
