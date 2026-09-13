from __future__ import annotations

from pathlib import Path

from fpl_automate.storage.db import Database
from tests.conftest import make_raw_team


def test_upsert_and_read_back_teams(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    teams = [make_raw_team(i) for i in range(1, 4)]
    db.upsert_teams(teams)
    stored = db.get_all_raw("teams")
    assert {t["id"] for t in stored} == {1, 2, 3}


def test_upsert_is_idempotent_and_updates(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    team = make_raw_team(1)
    db.upsert_teams([team])
    team["name"] = "Renamed"
    db.upsert_teams([team])
    stored = db.get_all_raw("teams")
    assert len(stored) == 1
    assert stored[0]["name"] == "Renamed"


def test_squad_snapshot_round_trip(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.save_squad_snapshot(team_id=42, as_of_event=5, state_json='{"hello": "world"}')
    snapshot = db.latest_squad_snapshot(team_id=42)
    assert snapshot == {"hello": "world"}


def test_squad_snapshot_returns_most_recent(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.save_squad_snapshot(team_id=42, as_of_event=5, state_json='{"event": 5}')
    db.save_squad_snapshot(team_id=42, as_of_event=6, state_json='{"event": 6}')
    snapshot = db.latest_squad_snapshot(team_id=42)
    assert snapshot == {"event": 6}


def test_decision_history_round_trip(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    decision_id = db.save_decision(event=10, strategy="balanced", recommendation_json="{}")
    db.record_outcome(decision_id, approved=True, actual_points=75, notes="good week")
    history = db.history()
    assert len(history) == 1
    assert history[0]["event"] == 10
    assert bool(history[0]["approved"]) is True
    assert history[0]["actual_points"] == 75


def test_fetch_log_tracks_last_success(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    assert db.last_successful_fetch("bootstrap") is None
    db.log_fetch("bootstrap", "error")
    assert db.last_successful_fetch("bootstrap") is None
    db.log_fetch("bootstrap", "ok")
    assert db.last_successful_fetch("bootstrap") is not None
