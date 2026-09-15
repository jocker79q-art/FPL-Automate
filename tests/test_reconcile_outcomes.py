from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fpl_automate.projections.ml import historical
from fpl_automate.storage.db import Database
from fpl_automate.storage.models import Gameweek
from fpl_automate.workflow import reconcile_outcomes


def _gw(id_: int, finished: bool) -> Gameweek:
    return Gameweek(
        id=id_,
        name=f"Gameweek {id_}",
        deadline_time=datetime.now(UTC) + timedelta(days=id_),
        finished=finished,
        is_current=False,
        is_next=False,
    )


class _StubClient:
    def __init__(self, history_current: list[dict]) -> None:
        self._history_current = history_current

    def get_entry_history(self, team_id: int) -> dict:
        return {"current": self._history_current}


def test_reconcile_outcomes_resolves_decisions_for_finished_gameweeks(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    decision_id = db.save_decision(event=3, strategy="balanced", recommendation_json="{}")

    client = _StubClient(history_current=[{"event": 3, "points": 62}])
    gameweeks = [_gw(3, finished=True), _gw(4, finished=False)]

    resolved = reconcile_outcomes(client, db, team_id=1, gameweeks=gameweeks, historical_dir=tmp_path / "hist")

    assert resolved["decisions"] == 1
    row = db.history(1)[0]
    assert row["id"] == decision_id
    assert row["actual_points"] == 62
    assert row["approved"] is None  # reconciliation must never guess/overwrite approval


def test_reconcile_outcomes_leaves_future_gameweek_decisions_unresolved(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.save_decision(event=5, strategy="balanced", recommendation_json="{}")

    client = _StubClient(history_current=[])
    gameweeks = [_gw(3, finished=True), _gw(5, finished=False)]

    resolved = reconcile_outcomes(client, db, team_id=1, gameweeks=gameweeks, historical_dir=tmp_path / "hist")

    assert resolved["decisions"] == 0
    assert db.history(1)[0]["actual_points"] is None


def test_reconcile_outcomes_does_not_fail_when_entry_history_call_errors(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.save_decision(event=3, strategy="balanced", recommendation_json="{}")

    class _BrokenClient:
        def get_entry_history(self, team_id: int) -> dict:
            raise RuntimeError("network down")

    gameweeks = [_gw(3, finished=True)]
    resolved = reconcile_outcomes(
        _BrokenClient(), db, team_id=1, gameweeks=gameweeks, historical_dir=tmp_path / "hist"
    )

    assert resolved["decisions"] == 0  # best-effort: never raises, just leaves it unresolved


def test_reconcile_outcomes_resolves_projections_from_the_current_season_archive(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.log_projections(
        event=2,
        model_name="ml",
        horizon_gameweeks=1,
        rows=[{"player_id": 1, "expected_points": 5.0, "floor_points": 3.0, "ceiling_points": 7.0, "confidence": 0.8}],
    )

    hist_dir = tmp_path / "hist"
    season_dir = hist_dir / historical.CURRENT_SEASON
    season_dir.mkdir(parents=True)
    (season_dir / "merged_gw.csv").write_text(
        "name,element,GW,total_points\nPlayer1,1,2,9\n"
    )

    client = _StubClient(history_current=[])
    gameweeks = [_gw(2, finished=True)]

    resolved = reconcile_outcomes(client, db, team_id=1, gameweeks=gameweeks, historical_dir=hist_dir)

    assert resolved["projections"] == 1
    resolved_rows = db.projection_performance()
    assert resolved_rows[0]["actual_points"] == 9


def test_reconcile_outcomes_leaves_projections_unresolved_when_archive_has_not_caught_up(tmp_path: Path):
    """vaastav's current-season archive can lag the real live gameweek --
    reconciliation must retry later, not guess or fail."""
    db = Database(tmp_path / "test.db")
    db.log_projections(
        event=5,
        model_name="ml",
        horizon_gameweeks=1,
        rows=[{"player_id": 1, "expected_points": 5.0, "floor_points": 3.0, "ceiling_points": 7.0, "confidence": 0.8}],
    )

    hist_dir = tmp_path / "hist"
    season_dir = hist_dir / historical.CURRENT_SEASON
    season_dir.mkdir(parents=True)
    # Archive only has GW1 so far, even though GW5 has "finished" live.
    (season_dir / "merged_gw.csv").write_text("name,element,GW,total_points\nPlayer1,1,1,3\n")

    client = _StubClient(history_current=[])
    gameweeks = [_gw(5, finished=True)]

    resolved = reconcile_outcomes(client, db, team_id=1, gameweeks=gameweeks, historical_dir=hist_dir)

    assert resolved["projections"] == 0
    assert db.projection_performance() == []


def test_reconcile_outcomes_no_finished_gameweeks_is_a_no_op(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    client = _StubClient(history_current=[])
    resolved = reconcile_outcomes(
        client, db, team_id=1, gameweeks=[_gw(1, finished=False)], historical_dir=tmp_path / "hist"
    )
    assert resolved == {"decisions": 0, "projections": 0}
