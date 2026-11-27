"""The dashboard's list of allowed status changes must match the store's rules."""
import json
from pathlib import Path

from incidents.store import ALLOWED_STATUS_CHANGES

TABLE = Path(__file__).resolve().parents[1] / "frontend" / "src" / "statusTransitions.json"


def test_frontend_status_table_matches_store() -> None:
    frontend = {k: set(v) for k, v in json.loads(TABLE.read_text(encoding="utf-8")).items()}
    assert frontend == ALLOWED_STATUS_CHANGES
