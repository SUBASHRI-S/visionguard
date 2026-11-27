"""What the API sends and receives. Server file paths are never exposed: evidence is a URL."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from incidents.store import IncidentRecord

Status = Literal["open", "acknowledged", "resolved", "false_positive"]


class IncidentSummary(BaseModel):
    id: int
    run_id: str
    camera_id: str
    rule_id: str
    track_id: int
    episode: int
    severity: str
    status: Status
    active: bool
    started_s: float
    opened_s: float | None
    last_violation_s: float | None
    ended_s: float | None
    end_reason: str | None
    violating_frames: int
    uncertain_frames: int
    created_at: datetime
    updated_at: datetime


class HistoryEntry(BaseModel):
    kind: str
    from_status: str | None
    to_status: str | None
    actor: str
    note: str
    at: datetime


class EvidenceLink(BaseModel):
    kind: str
    url: str


class IncidentDetail(IncidentSummary):
    details: dict[str, Any]
    history: list[HistoryEntry]
    evidence: list[EvidenceLink]


class IncidentPage(BaseModel):
    items: list[IncidentSummary]
    total: int
    limit: int
    offset: int


class StatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Status
    actor: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)] = Field(
        description="who is making the change")
    note: str = Field(default="", max_length=500)


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    database: Literal["ok"]


def summary_from(record: IncidentRecord) -> IncidentSummary:
    return IncidentSummary(**{k: getattr(record, k) for k in IncidentSummary.model_fields})


def detail_from(record: IncidentRecord) -> IncidentDetail:
    base = {k: getattr(record, k) for k in IncidentSummary.model_fields}
    evidence = [EvidenceLink(kind=e["kind"], url=f"/api/v1/incidents/{record.id}/evidence/{e['kind']}")
                for e in record.evidence]
    return IncidentDetail(**base, details=record.details, history=[HistoryEntry(**h) for h in record.history],
                          evidence=evidence)
