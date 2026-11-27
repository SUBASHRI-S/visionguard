import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.core.config import Settings, SettingsError
from backend.main import create_app
from incidents.store import IncidentStore
from incidents.temporal import Transition
from safety.events import Event

KEY = "k" * 20


def event(rule="no_helmet", camera="cam", track=1, ts=1.0):
    return Event(rule_id=rule, camera_id=camera, frame_index=int(ts * 10), timestamp=ts, outcome="violation",
                 severity="high", person_index=0, track_id=track, details={"box": (1.0, 2.0, 3.0, 4.0)})


def transition(kind="opened", rule="no_helmet", camera="cam", track=1, episode=1, ts=4.0, reason=None):
    return Transition(kind=kind, camera_id=camera, rule_id=rule, track_id=track, episode=episode,
                      frame_index=int(ts * 10), timestamp=ts, started_frame=10, started_timestamp=1.0,
                      severity="high", event=event(rule, camera, track, ts), violating_frames=30,
                      uncertain_frames=2, reason=reason)


@pytest.fixture
def evidence_root(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    return root


@pytest.fixture
def settings(tmp_path, evidence_root):
    return Settings(db_url=f"sqlite:///{tmp_path / 'api.db'}", evidence_root=evidence_root)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def seed(client, evidence_root=None):
    """Three incidents; incident 1 has a real picture on disk."""
    store: IncidentStore = client.app.state.store
    picture = None
    if evidence_root is not None:
        picture = evidence_root / "open.jpg"
        picture.write_bytes(b"\xff\xd8\xff fake jpeg")
    store.record_transition(transition(), run_id="rA", evidence_path=str(picture) if picture else None)
    store.record_transition(transition(rule="no_vest", track=2), run_id="rA")
    store.record_transition(transition("closed", rule="no_vest", track=2, ts=8, reason="cleared"), run_id="rA")
    store.record_transition(transition(camera="cam2", track=3), run_id="rB")
    return store


# ---------------------------------------------------------------- settings

def test_settings_defaults_and_env():
    s = Settings.from_env({})
    assert s.db_url.startswith("sqlite") and s.api_key is None
    s = Settings.from_env({"VG_DB_URL": "sqlite:///x.db", "VG_EVIDENCE_ROOT": "ev", "VG_API_KEY": "z" * 16,
                           "VG_CORS_ORIGINS": "http://a.example, https://b.example"})
    assert s.evidence_root == Path("ev") and s.cors_origins == ("http://a.example", "https://b.example")


@pytest.mark.parametrize("kw", [dict(api_key="short"), dict(db_url="  "), dict(cors_origins=("*",)),
                                dict(cors_origins=("localhost:5173",))])
def test_unsafe_settings_rejected(kw):
    with pytest.raises(SettingsError):
        Settings(**kw)


# ---------------------------------------------------------------- health

def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "version": "0.1.0", "database": "ok"}


def test_health_reports_a_broken_database(client, monkeypatch):
    def boom(**_):
        raise RuntimeError("db gone")
    monkeypatch.setattr(client.app.state.store, "list_incidents", boom)
    r = client.get("/health")
    assert r.status_code == 503 and "db gone" not in r.text


# ---------------------------------------------------------------- listing

def test_empty_list(client):
    assert client.get("/api/v1/incidents").json() == {"items": [], "total": 0, "limit": 50, "offset": 0}


def test_list_filters_and_order(client):
    seed(client)
    body = client.get("/api/v1/incidents").json()
    assert body["total"] == 3 and [i["id"] for i in body["items"]] == [3, 2, 1]
    assert [i["id"] for i in client.get("/api/v1/incidents?camera_id=cam2").json()["items"]] == [3]
    assert [i["id"] for i in client.get("/api/v1/incidents?rule_id=no_vest").json()["items"]] == [2]
    assert [i["id"] for i in client.get("/api/v1/incidents?active=false").json()["items"]] == [2]
    assert client.get("/api/v1/incidents?run_id=rB").json()["total"] == 1


def test_pagination(client):
    seed(client)
    body = client.get("/api/v1/incidents?limit=2&offset=2").json()
    assert body["total"] == 3 and [i["id"] for i in body["items"]] == [1] and body["limit"] == 2


def test_time_filters_accept_naive_and_aware_dates(client):
    seed(client)
    assert client.get("/api/v1/incidents?since=2000-01-01T00:00:00").json()["total"] == 3
    assert client.get("/api/v1/incidents?since=2000-01-01T00:00:00%2B00:00").json()["total"] == 3
    assert client.get("/api/v1/incidents?since=2999-01-01T00:00:00").json()["total"] == 0


@pytest.mark.parametrize("query", ["limit=0", "limit=501", "offset=-1", "status=banana", "since=junk", "active=maybe"])
def test_bad_query_is_422(client, query):
    assert client.get(f"/api/v1/incidents?{query}").status_code == 422


# ---------------------------------------------------------------- detail

def test_detail_has_history_and_evidence_links_but_no_server_paths(client, evidence_root):
    seed(client, evidence_root)
    r = client.get("/api/v1/incidents/1")
    body = r.json()
    assert body["history"][0]["kind"] == "opened"
    assert body["evidence"] == [{"kind": "opened", "url": "/api/v1/incidents/1/evidence/opened"}]
    assert body["details"] == {"box": [1.0, 2.0, 3.0, 4.0]}
    assert str(evidence_root) not in r.text and "open.jpg" not in r.text


def test_unknown_incident_is_404(client):
    assert client.get("/api/v1/incidents/99").status_code == 404
    assert client.get("/api/v1/incidents/abc").status_code == 422


# ---------------------------------------------------------------- review

def test_patch_changes_status_and_records_who(client):
    seed(client)
    r = client.patch("/api/v1/incidents/1", json={"status": "acknowledged", "actor": "asha", "note": "on it"})
    assert r.status_code == 200 and r.json()["status"] == "acknowledged"
    last = r.json()["history"][-1]
    assert (last["actor"], last["from_status"], last["to_status"], last["note"]) == ("asha", "open", "acknowledged", "on it")
    assert client.get("/api/v1/incidents?status=acknowledged").json()["total"] == 1


def test_disallowed_change_is_409_and_nothing_changes(client):
    seed(client)
    client.patch("/api/v1/incidents/1", json={"status": "resolved", "actor": "a"})
    r = client.patch("/api/v1/incidents/1", json={"status": "acknowledged", "actor": "a"})
    assert r.status_code == 409
    assert client.get("/api/v1/incidents/1").json()["status"] == "resolved"


@pytest.mark.parametrize("body", [
    {"status": "banana", "actor": "a"}, {"status": "resolved"}, {"actor": "a"}, {"status": "resolved", "actor": ""},
    {"status": "resolved", "actor": "   "}, {"status": "resolved", "actor": "a", "note": "n" * 501},
    {"status": "resolved", "actor": "a", "id": 5},
])
def test_bad_patch_bodies_are_422(client, body):
    seed(client)
    assert client.patch("/api/v1/incidents/1", json=body).status_code == 422
    assert client.get("/api/v1/incidents/1").json()["status"] == "open"


def test_patch_unknown_incident_is_404(client):
    assert client.patch("/api/v1/incidents/9", json={"status": "resolved", "actor": "a"}).status_code == 404


# ---------------------------------------------------------------- evidence

def test_evidence_is_served_as_an_image(client, evidence_root):
    seed(client, evidence_root)
    r = client.get("/api/v1/incidents/1/evidence/opened")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content.startswith(b"\xff\xd8")


def test_evidence_outside_the_root_is_not_served(client, tmp_path, evidence_root):
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"top secret")
    client.app.state.store.record_transition(transition(), run_id="r", evidence_path=str(secret))
    assert client.get("/api/v1/incidents/1/evidence/opened").status_code == 404


def test_path_tricks_do_not_escape_the_root(client, tmp_path, evidence_root):
    (tmp_path / "secret.jpg").write_bytes(b"top secret")
    sneaky = str(evidence_root / ".." / "secret.jpg")
    client.app.state.store.record_transition(transition(), run_id="r", evidence_path=sneaky)
    assert client.get("/api/v1/incidents/1/evidence/opened").status_code == 404


def test_only_pictures_are_served(client, evidence_root):
    notes = evidence_root / "notes.txt"
    notes.write_text("not a picture")
    client.app.state.store.record_transition(transition(), run_id="r", evidence_path=str(notes))
    assert client.get("/api/v1/incidents/1/evidence/opened").status_code == 404


def test_missing_evidence_cases_are_404(client, evidence_root):
    client.app.state.store.record_transition(transition(), run_id="r", evidence_path=str(evidence_root / "gone.jpg"))
    assert client.get("/api/v1/incidents/1/evidence/opened").status_code == 404     # file deleted
    assert client.get("/api/v1/incidents/1/evidence/closed").status_code == 404     # never recorded
    assert client.get("/api/v1/incidents/42/evidence/opened").status_code == 404    # no such incident


# ---------------------------------------------------------------- stats

def test_stats(client):
    seed(client)
    body = client.get("/api/v1/stats").json()
    assert body["total"] == 3 and body["active"] == 2 and body["by_rule"] == {"no_helmet": 2, "no_vest": 1}
    assert client.get("/api/v1/stats?days=0").status_code == 422


# ---------------------------------------------------------------- api key

@pytest.fixture
def secured(tmp_path, evidence_root):
    s = Settings(db_url=f"sqlite:///{tmp_path / 's.db'}", evidence_root=evidence_root, api_key=KEY)
    with TestClient(create_app(s)) as c:
        yield c


def test_api_requires_the_key_when_one_is_set(secured):
    assert secured.get("/api/v1/incidents").status_code == 401
    assert secured.get("/api/v1/incidents", headers={"X-API-Key": "wrong" * 5}).status_code == 401
    assert secured.get("/api/v1/incidents", headers={"X-API-Key": KEY}).status_code == 200
    assert secured.get("/api/v1/stats").status_code == 401
    assert secured.patch("/api/v1/incidents/1", json={"status": "resolved", "actor": "a"}).status_code == 401
    assert secured.get("/api/v1/incidents/1/evidence/opened").status_code == 401


def test_health_stays_open_so_monitors_can_reach_it(secured):
    assert secured.get("/health").status_code == 200


def test_error_for_wrong_key_does_not_echo_the_key(secured):
    r = secured.get("/api/v1/incidents", headers={"X-API-Key": "nope" * 6})
    assert "nope" not in r.text and KEY not in r.text


# ---------------------------------------------------------------- cors

def test_cors_only_for_allowed_origins(client):
    ok = client.options("/api/v1/incidents", headers={"Origin": "http://localhost:5173",
                                                      "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/api/v1/incidents", headers={"Origin": "http://evil.example",
                                                       "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers
