"""Settings come from environment variables, are validated once at start-up, and never hardcoded.

VG_DB_URL            database URL (default: sqlite:///outputs/visionguard.db)
VG_EVIDENCE_ROOT     the only folder evidence pictures may be served from (default: outputs)
VG_API_KEY           if set, every /api request must send it in the X-API-Key header (min. 16 characters)
VG_CORS_ORIGINS      comma-separated browser origins allowed to call the API (default: the Vite dev server)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


class SettingsError(ValueError):
    """Raised when a setting is missing or unsafe."""


@dataclass(frozen=True)
class Settings:
    db_url: str = "sqlite:///outputs/visionguard.db"
    evidence_root: Path = Path("outputs")
    api_key: str | None = None
    cors_origins: tuple[str, ...] = field(default=("http://localhost:5173", "http://127.0.0.1:5173"))

    def __post_init__(self) -> None:
        if not self.db_url.strip():
            raise SettingsError("VG_DB_URL must not be empty")
        if self.api_key is not None and len(self.api_key) < 16:
            raise SettingsError("VG_API_KEY must be at least 16 characters (or leave it unset to disable the check)")
        for origin in self.cors_origins:
            if origin == "*" or not origin.startswith(("http://", "https://")):
                raise SettingsError(f"CORS origin {origin!r} must be a full http(s) address; '*' is not allowed")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        kwargs: dict = {}
        if env.get("VG_DB_URL"):
            kwargs["db_url"] = env["VG_DB_URL"]
        if env.get("VG_EVIDENCE_ROOT"):
            kwargs["evidence_root"] = Path(env["VG_EVIDENCE_ROOT"])
        if env.get("VG_API_KEY"):
            kwargs["api_key"] = env["VG_API_KEY"]
        if env.get("VG_CORS_ORIGINS"):
            kwargs["cors_origins"] = tuple(o.strip() for o in env["VG_CORS_ORIGINS"].split(",") if o.strip())
        return cls(**kwargs)
