"""Incident storage: the only code that writes incident rows (see docs/ARCHITECTURE.md section 4).

Two different questions are kept apart on purpose:

* ``active`` / ``ended_*``: is the violation still happening? Set by the temporal logic
  (``incidents/temporal.py``), never by a person.
* ``status``: what did a human reviewer decide? ``open -> acknowledged -> resolved``, or
  ``false_positive``. Set only through :meth:`IncidentStore.set_status`, and every change is kept
  in ``incident_events`` with who made it and when.

A violation can have ended while nobody has looked at it yet (inactive, status ``open``), and a
reviewer can acknowledge an incident that is still going on (active, status ``acknowledged``).

SQLite by default (a file, no server) through SQLAlchemy, so the same code runs on PostgreSQL by
changing the URL. Times: ``*_s`` columns are seconds of video time; ``created_at``/``updated_at``
are UTC wall-clock times.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint,
                        create_engine, event, func, select)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

from incidents.temporal import Transition

STATUSES = ("open", "acknowledged", "resolved", "false_positive")
ALLOWED_STATUS_CHANGES: dict[str, set[str]] = {
    "open": {"acknowledged", "resolved", "false_positive"},
    "acknowledged": {"resolved", "false_positive"},
    "resolved": {"open"},          # reopen
    "false_positive": {"open"},    # reviewer changed their mind
}
MAX_PAGE_SIZE = 500


class StoreError(RuntimeError):
    """Base class for storage problems callers are expected to handle."""


class IncidentNotFound(StoreError, KeyError):
    pass


class InvalidStatusChange(StoreError, ValueError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    """SQLite gives back naive datetimes; everything we store is UTC, so say so."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class Base(DeclarativeBase):
    pass


class IncidentRow(Base):
    __tablename__ = "incidents"
    __table_args__ = (UniqueConstraint("run_id", "camera_id", "rule_id", "track_id", "episode", name="uq_incident_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(120), index=True)
    camera_id: Mapped[str] = mapped_column(String(120), index=True)
    rule_id: Mapped[str] = mapped_column(String(60), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    episode: Mapped[int] = mapped_column(Integer)
    severity: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="open", index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    started_s: Mapped[float] = mapped_column(Float)
    started_frame: Mapped[int] = mapped_column(Integer)
    opened_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_violation_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    ended_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_reason: Mapped[str | None] = mapped_column(String(30), nullable=True)
    violating_frames: Mapped[int] = mapped_column(Integer, default=0)
    uncertain_frames: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    events: Mapped[list["IncidentEventRow"]] = relationship(back_populates="incident", order_by="IncidentEventRow.id",
                                                            cascade="all, delete-orphan")
    evidence: Mapped[list["EvidenceRow"]] = relationship(back_populates="incident", order_by="EvidenceRow.id",
                                                         cascade="all, delete-orphan")


class IncidentEventRow(Base):
    __tablename__ = "incident_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))            # opened | ended | status_change
    from_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    actor: Mapped[str] = mapped_column(String(80), default="system")
    note: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    incident: Mapped[IncidentRow] = relationship(back_populates="events")


class EvidenceRow(Base):
    __tablename__ = "incident_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))            # opened | closed (the transition it shows)
    path: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    incident: Mapped[IncidentRow] = relationship(back_populates="evidence")


# ------------------------------------------------------------------ plain records returned to callers

@dataclass(frozen=True)
class IncidentRecord:
    id: int
    run_id: str
    camera_id: str
    rule_id: str
    track_id: int
    episode: int
    severity: str
    status: str
    active: bool
    started_s: float
    started_frame: int
    opened_s: float | None
    last_violation_s: float | None
    ended_s: float | None
    end_reason: str | None
    violating_frames: int
    uncertain_frames: int
    details: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    history: tuple[dict[str, Any], ...] = field(default=())
    evidence: tuple[dict[str, Any], ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        out["created_at"], out["updated_at"] = self.created_at.isoformat(), self.updated_at.isoformat()
        out["history"], out["evidence"] = list(self.history), list(self.evidence)
        return out


def _record(row: IncidentRow, *, detail: bool) -> IncidentRecord:
    history = evidence = ()
    if detail:
        history = tuple({"kind": e.kind, "from_status": e.from_status, "to_status": e.to_status, "actor": e.actor,
                         "note": e.note, "at": _aware(e.created_at).isoformat()} for e in row.events)
        evidence = tuple({"kind": e.kind, "path": e.path} for e in row.evidence)
    return IncidentRecord(
        id=row.id, run_id=row.run_id, camera_id=row.camera_id, rule_id=row.rule_id, track_id=row.track_id,
        episode=row.episode, severity=row.severity, status=row.status, active=row.active, started_s=row.started_s,
        started_frame=row.started_frame, opened_s=row.opened_s, last_violation_s=row.last_violation_s,
        ended_s=row.ended_s, end_reason=row.end_reason, violating_frames=row.violating_frames,
        uncertain_frames=row.uncertain_frames, details=dict(row.details or {}),
        created_at=_aware(row.created_at), updated_at=_aware(row.updated_at), history=history, evidence=evidence,
    )


def _jsonable(value: Any) -> Any:
    """Event details can hold tuples or numpy numbers; make them plain JSON."""
    return json.loads(json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ------------------------------------------------------------------ the store

class IncidentStore:
    def __init__(self, url: str = "sqlite:///outputs/visionguard.db") -> None:
        kwargs: dict[str, Any] = {}
        if url.startswith("sqlite"):
            path = url.split("///", 1)[1] if "///" in url else ""
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            if path in ("", ":memory:"):
                kwargs.update(connect_args={"check_same_thread": False}, poolclass=StaticPool)
        self.engine: Engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _fk_on(dbapi_connection, _record):  # SQLite ignores foreign keys unless asked
                dbapi_connection.execute("PRAGMA foreign_keys=ON")
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()

    def _session(self) -> Session:
        return self._sessions()

    # ---- writing what the temporal logic decided

    def record_transition(self, t: Transition, *, run_id: str, evidence_path: str | None = None) -> int:
        """Store an opened/closed transition. Safe to call twice with the same transition."""
        if not run_id:
            raise ValueError("run_id must not be empty")
        with self._session() as session, session.begin():
            row = session.scalar(select(IncidentRow).where(
                IncidentRow.run_id == run_id, IncidentRow.camera_id == t.camera_id, IncidentRow.rule_id == t.rule_id,
                IncidentRow.track_id == t.track_id, IncidentRow.episode == t.episode))
            if row is None:
                row = IncidentRow(
                    run_id=run_id, camera_id=t.camera_id, rule_id=t.rule_id, track_id=t.track_id, episode=t.episode,
                    severity=str(t.severity), status="open", active=True, started_s=t.started_timestamp,
                    started_frame=t.started_frame, violating_frames=t.violating_frames,
                    uncertain_frames=t.uncertain_frames, details=_jsonable(t.event.details))
                session.add(row)
                session.flush()
            if t.kind == "opened" and row.opened_s is None:
                row.opened_s = t.timestamp
                row.violating_frames, row.uncertain_frames = t.violating_frames, t.uncertain_frames
                row.events.append(IncidentEventRow(kind="opened", to_status="open"))
            elif t.kind == "closed" and row.active:
                row.active = False
                row.ended_s, row.end_reason = t.timestamp, t.reason
                row.last_violation_s = t.event.timestamp
                row.violating_frames, row.uncertain_frames = t.violating_frames, t.uncertain_frames
                row.events.append(IncidentEventRow(kind="ended", note=t.reason or ""))
            if evidence_path and not any(e.path == evidence_path for e in row.evidence):
                row.evidence.append(EvidenceRow(kind=t.kind, path=evidence_path))
            row.updated_at = _utcnow()
            return row.id

    # ---- human review

    def set_status(self, incident_id: int, new_status: str, *, actor: str, note: str = "") -> IncidentRecord:
        if new_status not in STATUSES:
            raise InvalidStatusChange(f"unknown status {new_status!r}; use one of {', '.join(STATUSES)}")
        if not actor or not actor.strip():
            raise ValueError("actor must say who made the change")
        if len(note) > 500:
            raise ValueError("note is limited to 500 characters")
        with self._session() as session, session.begin():
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise IncidentNotFound(f"no incident with id {incident_id}")
            if new_status not in ALLOWED_STATUS_CHANGES[row.status]:
                raise InvalidStatusChange(f"cannot change from {row.status!r} to {new_status!r}")
            row.events.append(IncidentEventRow(kind="status_change", from_status=row.status, to_status=new_status,
                                               actor=actor.strip(), note=note))
            row.status = new_status
            row.updated_at = _utcnow()
            session.flush()
            return _record(row, detail=True)

    # ---- reading

    def get(self, incident_id: int) -> IncidentRecord:
        with self._session() as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise IncidentNotFound(f"no incident with id {incident_id}")
            return _record(row, detail=True)

    def list_incidents(self, *, camera_id: str | None = None, rule_id: str | None = None, status: str | None = None,
             active: bool | None = None, run_id: str | None = None, since: datetime | None = None,
             until: datetime | None = None, limit: int = 50, offset: int = 0) -> tuple[list[IncidentRecord], int]:
        """Newest first. Returns (page of incidents, total matching)."""
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        if offset < 0:
            raise ValueError("offset must not be negative")
        if status is not None and status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        conditions = []
        for column, value in ((IncidentRow.camera_id, camera_id), (IncidentRow.rule_id, rule_id),
                              (IncidentRow.status, status), (IncidentRow.active, active), (IncidentRow.run_id, run_id)):
            if value is not None:
                conditions.append(column == value)
        if since is not None:
            conditions.append(IncidentRow.created_at >= since)
        if until is not None:
            conditions.append(IncidentRow.created_at <= until)
        with self._session() as session:
            total = session.scalar(select(func.count()).select_from(IncidentRow).where(*conditions)) or 0
            rows = session.scalars(select(IncidentRow).where(*conditions).order_by(IncidentRow.id.desc())
                                   .limit(limit).offset(offset)).all()
            return [_record(r, detail=False) for r in rows], int(total)

    def stats(self, *, days: int = 14) -> dict[str, Any]:
        if not 1 <= days <= 366:
            raise ValueError("days must be between 1 and 366")
        with self._session() as session:
            def counts(column) -> dict[str, int]:
                return {str(k): int(v) for k, v in session.execute(select(column, func.count()).group_by(column)).all()}

            total = int(session.scalar(select(func.count()).select_from(IncidentRow)) or 0)
            active = int(session.scalar(select(func.count()).select_from(IncidentRow).where(IncidentRow.active)) or 0)
            cutoff = _utcnow() - timedelta(days=days)
            recent = session.scalars(select(IncidentRow.created_at).where(IncidentRow.created_at >= cutoff)).all()
        by_day: dict[str, int] = {}
        for stamp in recent:
            key = _aware(stamp).date().isoformat()
            by_day[key] = by_day.get(key, 0) + 1
        return {"total": total, "active": active, "by_rule": counts(IncidentRow.rule_id),
                "by_status": counts(IncidentRow.status), "by_severity": counts(IncidentRow.severity),
                "by_camera": counts(IncidentRow.camera_id), "by_day": dict(sorted(by_day.items()))}
