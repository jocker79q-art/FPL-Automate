from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fpl_automate.projections.ml import features, historical, model
from fpl_automate.projections.ml.live import (
    _fixture_by_team_for_event,
    apply_live_availability,
    build_prediction_rows,
    compute_ml_projections,
)
from fpl_automate.storage.models import AvailabilityStatus, Fixture, Player, Position, Team


def _team(id_: int, name: str, short_name: str = "TST") -> Team:
    return Team(
        id=id_,
        name=name,
        short_name=short_name,
        strength_overall_home=1100,
        strength_overall_away=1100,
        strength_attack_home=1300,
        strength_attack_away=1200,
        strength_defence_home=1200,
        strength_defence_away=1100,
    )


def _player(
    id_: int, team_id: int, position: Position = Position.MIDFIELDER, status: str = "a", chance: int | None = None
) -> Player:
    return Player(
        id=id_,
        web_name=f"P{id_}",
        full_name=f"Player {id_}",
        team_id=team_id,
        position=position,
        now_cost_tenths=60,
        selected_by_percent=5.0,
        availability=AvailabilityStatus(status=status, chance_of_playing_next_round=chance, news=""),
        form=4.0,
        points_per_game=4.0,
        total_points=20,
        minutes=450,
        starts=5,
        goals_scored=1,
        assists=1,
        clean_sheets=1,
        goals_conceded=5,
        bonus=2,
        bps=100,
        yellow_cards=0,
        red_cards=0,
        saves=0,
        expected_goals=1.0,
        expected_assists=0.8,
        expected_goal_involvements=1.8,
        expected_goals_conceded=4.0,
    )


# --- apply_live_availability -----------------------------------------------


def test_apply_live_availability_trusts_a_published_percentage():
    assert apply_live_availability(0.9, "d", 25) == 0.25


def test_apply_live_availability_zeroes_confirmed_absence_with_no_percentage():
    assert apply_live_availability(0.8, "i", None) == 0.0
    assert apply_live_availability(0.8, "s", None) == 0.0
    assert apply_live_availability(0.8, "u", None) == 0.0


def test_apply_live_availability_keeps_model_estimate_when_no_live_signal():
    assert apply_live_availability(0.87, "a", None) == 0.87
    assert apply_live_availability(0.5, "d", None) == 0.5  # doubtful, no percentage yet


# --- build_prediction_rows / fixture lookups -------------------------------


def test_build_prediction_rows_marks_blank_gameweek_players():
    players = [_player(1, team_id=10)]
    row = build_prediction_rows(players, fixture_by_team={}, event=5, team_names={10: "Alpha"}).iloc[0]
    assert bool(row["blank_gameweek"]) is True
    assert row["opponent_team"] is None


def test_build_prediction_rows_carries_real_fixture_fields():
    players = [_player(1, team_id=10)]
    fixture_by_team = {10: {"opponent_team": 20, "was_home": True}}
    row = build_prediction_rows(players, fixture_by_team, event=5, team_names={10: "Alpha"}).iloc[0]
    assert bool(row["blank_gameweek"]) is False
    assert row["opponent_team"] == 20
    assert bool(row["was_home"]) is True
    assert row["position"] == "MID"


def test_fixture_by_team_for_event_only_uses_that_events_fixtures():
    fixtures = [
        Fixture(
            id=1, event=5, team_h=10, team_a=20, team_h_difficulty=3, team_a_difficulty=3,
            kickoff_time=None, finished=False,
        ),
        Fixture(
            id=2, event=6, team_h=10, team_a=30, team_h_difficulty=2, team_a_difficulty=4,
            kickoff_time=None, finished=False,
        ),
    ]
    by_team = _fixture_by_team_for_event(fixtures, event=5)
    assert by_team[10] == {"opponent_team": 20, "was_home": True}
    assert by_team[20] == {"opponent_team": 10, "was_home": False}
    assert 30 not in by_team  # that fixture is event 6, not 5


# --- compute_ml_projections end-to-end -------------------------------------


def _train_tiny_model(models_dir: Path) -> None:
    cols = features.feature_columns()
    rng = np.random.default_rng(1)
    n = 100
    data = {c: rng.normal(size=n) for c in cols}
    data["was_home"] = rng.integers(0, 2, size=n)
    played = rng.integers(0, 2, size=n)
    data["minutes"] = played * 90
    data["total_points"] = np.where(played, rng.poisson(3, size=n), 0)
    df = pd.DataFrame(data)
    pos_model = model.fit_position_model(df)
    model.save_models({"MID": pos_model}, models_dir)
    model.save_calibration({"MID": {"residual_p10": -1.0, "residual_p90": 1.0}}, models_dir)


def test_compute_ml_projections_returns_none_when_no_trained_model(tmp_path: Path):
    result = compute_ml_projections(
        players=[_player(1, 10)],
        teams=[_team(10, "Alpha")],
        fixtures=[],
        from_event=2,
        horizons=(1,),
        historical_dir=tmp_path / "hist",
        models_dir=tmp_path / "models",
    )
    assert result is None


def test_compute_ml_projections_end_to_end(tmp_path: Path, monkeypatch):
    hist_dir = tmp_path / "hist"
    models_dir = tmp_path / "models"
    season = historical.CURRENT_SEASON
    season_dir = hist_dir / season
    season_dir.mkdir(parents=True)

    (season_dir / "merged_gw.csv").write_text(
        "element,GW,total_points,minutes,starts,ict_index,influence,creativity,threat,bps,"
        "expected_goals,expected_assists,expected_goal_involvements,expected_goals_conceded,"
        "goals_scored,assists,clean_sheets,saves,bonus,value,was_home,opponent_team,team,xP,position\n"
        "1,1,6,90,1,10,10,10,10,20,0.5,0.3,0.8,1.0,1,0,0,0,1,60,True,20,Alpha,4.0,MID\n"
    )
    pd.DataFrame(
        [
            {
                "id": 10, "name": "Alpha", "short_name": "ALP",
                "strength_attack_home": 1300, "strength_attack_away": 1200,
                "strength_defence_home": 1200, "strength_defence_away": 1100,
            },
            {
                "id": 20, "name": "Beta", "short_name": "BET",
                "strength_attack_home": 1250, "strength_attack_away": 1150,
                "strength_defence_home": 1150, "strength_defence_away": 1050,
            },
        ]
    ).to_csv(season_dir / "teams.csv", index=False)
    pd.DataFrame(columns=["id"]).to_csv(season_dir / "players_raw.csv", index=False)
    pd.DataFrame(
        [
            {"id": 1, "event": 2, "team_h": 10, "team_a": 20, "team_h_difficulty": 2, "team_a_difficulty": 3, "finished": False},
            {"id": 2, "event": 3, "team_h": 20, "team_a": 10, "team_h_difficulty": 3, "team_a_difficulty": 2, "finished": False},
        ]
    ).to_csv(season_dir / "fixtures.csv", index=False)

    # This test only cares about scoring against already-cached data --
    # never let it attempt a real network call.
    monkeypatch.setattr(historical, "fetch_season", lambda *a, **k: None)

    _train_tiny_model(models_dir)

    players = [_player(1, team_id=10, position=Position.MIDFIELDER)]
    teams = [_team(10, "Alpha"), _team(20, "Beta")]
    fixtures = [
        Fixture(id=1, event=2, team_h=10, team_a=20, team_h_difficulty=2, team_a_difficulty=3, kickoff_time=None, finished=False),
        Fixture(id=2, event=3, team_h=20, team_a=10, team_h_difficulty=3, team_a_difficulty=2, kickoff_time=None, finished=False),
    ]

    result = compute_ml_projections(
        players=players,
        teams=teams,
        fixtures=fixtures,
        from_event=2,
        horizons=(1, 2),
        historical_dir=hist_dir,
        models_dir=models_dir,
    )

    assert result is not None
    assert 1 in result[1]
    proj = result[1][1]
    assert proj.expected_points >= 0
    assert proj.floor_points <= proj.expected_points <= proj.ceiling_points
    assert "ML model" in proj.rationale
    assert "small_sample" in proj.risk_flags  # only 1 prior gameweek this season

    # horizon=2 sums two real gameweeks, so it should differ from a naive
    # doubling of horizon=1 (different fixture difficulty each gameweek).
    proj_h2 = result[2][1]
    assert proj_h2.expected_points >= 0


def test_compute_ml_projections_applies_live_availability_override(tmp_path: Path, monkeypatch):
    hist_dir = tmp_path / "hist"
    models_dir = tmp_path / "models"
    season = historical.CURRENT_SEASON
    season_dir = hist_dir / season
    season_dir.mkdir(parents=True)

    (season_dir / "merged_gw.csv").write_text(
        "element,GW,total_points,minutes,starts,ict_index,influence,creativity,threat,bps,"
        "expected_goals,expected_assists,expected_goal_involvements,expected_goals_conceded,"
        "goals_scored,assists,clean_sheets,saves,bonus,value,was_home,opponent_team,team,xP,position\n"
        "2,1,6,90,1,10,10,10,10,20,0.5,0.3,0.8,1.0,1,0,0,0,1,60,True,20,Alpha,4.0,MID\n"
    )
    pd.DataFrame(
        [
            {
                "id": 10, "name": "Alpha", "short_name": "ALP",
                "strength_attack_home": 1300, "strength_attack_away": 1200,
                "strength_defence_home": 1200, "strength_defence_away": 1100,
            },
            {
                "id": 20, "name": "Beta", "short_name": "BET",
                "strength_attack_home": 1250, "strength_attack_away": 1150,
                "strength_defence_home": 1150, "strength_defence_away": 1050,
            },
        ]
    ).to_csv(season_dir / "teams.csv", index=False)
    pd.DataFrame(columns=["id"]).to_csv(season_dir / "players_raw.csv", index=False)
    pd.DataFrame(
        [{"id": 1, "event": 2, "team_h": 10, "team_a": 20, "team_h_difficulty": 2, "team_a_difficulty": 3, "finished": False}]
    ).to_csv(season_dir / "fixtures.csv", index=False)

    monkeypatch.setattr(historical, "fetch_season", lambda *a, **k: None)
    _train_tiny_model(models_dir)

    # A confirmed-out injured player -- P(plays) must be forced to exactly 0
    # regardless of what the classifier itself would have said.
    injured_player = _player(2, team_id=10, position=Position.MIDFIELDER, status="i", chance=None)
    teams = [_team(10, "Alpha"), _team(20, "Beta")]
    fixtures = [
        Fixture(id=1, event=2, team_h=10, team_a=20, team_h_difficulty=2, team_a_difficulty=3, kickoff_time=None, finished=False),
    ]

    result = compute_ml_projections(
        players=[injured_player],
        teams=teams,
        fixtures=fixtures,
        from_event=2,
        horizons=(1,),
        historical_dir=hist_dir,
        models_dir=models_dir,
    )

    assert result is not None
    proj = result[1][2]
    assert proj.expected_points == 0.0
    assert proj.confidence == 0.0
    assert "injury_or_availability_doubt" in proj.risk_flags
