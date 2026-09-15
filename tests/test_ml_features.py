from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fpl_automate.projections.ml import features


def _row(season, element, gw, total_points, was_home=True, opponent_team=2, team="Alpha", value=50, **overrides):
    base = {
        "season": season,
        "element": element,
        "GW": gw,
        "total_points": total_points,
        "minutes": 90 if total_points else 0,
        "starts": 1,
        "ict_index": 10.0,
        "influence": 10.0,
        "creativity": 10.0,
        "threat": 10.0,
        "bps": 20,
        "expected_goals": 0.5,
        "expected_assists": 0.3,
        "expected_goal_involvements": 0.8,
        "expected_goals_conceded": 1.0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "saves": 0,
        "bonus": 0,
        "value": value,
        "was_home": was_home,
        "opponent_team": opponent_team,
        "team": team,
        "xP": 3.0,
    }
    base.update(overrides)
    return base


def test_add_rolling_features_never_includes_the_current_row_own_result():
    """The one rule everything in this module is built around: a gameweek's
    rolling features must come from strictly prior gameweeks only."""
    rows = [
        _row("2024-25", 1, 1, total_points=2),
        _row("2024-25", 1, 2, total_points=100),  # huge outlier -- must not leak into itself
        _row("2024-25", 1, 3, total_points=4),
    ]
    df = pd.DataFrame(rows)
    out = features.add_rolling_features(df)

    gw2 = out[out["GW"] == 2].iloc[0]
    # roll3 average of GW1 only (the single prior gameweek) is 2, not
    # anything blended with GW2's own 100.
    assert gw2["total_points_roll3"] == 2.0

    gw3 = out[out["GW"] == 3].iloc[0]
    # roll3 average of GW1 (2) and GW2 (100) = 51, proving GW2's real
    # result *is* available for GW3 -- only a row's *own* result is hidden.
    assert gw3["total_points_roll3"] == pytest.approx(51.0)


def test_add_rolling_features_first_appearance_has_zero_filled_rolling_stats():
    rows = [_row("2024-25", 1, 1, total_points=6)]
    df = pd.DataFrame(rows)
    out = features.add_rolling_features(df)

    row = out.iloc[0]
    assert row["total_points_roll3"] == 0.0
    assert row["games_played_so_far"] == 0


def test_add_rolling_features_does_not_leak_across_seasons_even_with_same_element_id():
    """FPL recycles element IDs across seasons -- a season boundary must
    reset rolling history, not carry it over."""
    rows = [
        _row("2023-24", 7, 38, total_points=99),  # last GW of an old season
        _row("2024-25", 7, 1, total_points=3),  # same element id, new season
    ]
    df = pd.DataFrame(rows)
    out = features.add_rolling_features(df)

    new_season_row = out[(out["season"] == "2024-25") & (out["GW"] == 1)].iloc[0]
    assert new_season_row["total_points_roll3"] == 0.0
    assert new_season_row["games_played_so_far"] == 0


def test_add_cumulative_prior_features_excludes_the_current_gameweek():
    rows = [
        _row("2024-25", 1, 1, total_points=2, minutes=90),
        _row("2024-25", 1, 2, total_points=10, minutes=90),
        _row("2024-25", 1, 3, total_points=1, minutes=90),
    ]
    df = pd.DataFrame(rows)
    out = features.add_cumulative_prior_features(df)

    gw1 = out[out["GW"] == 1].iloc[0]
    gw2 = out[out["GW"] == 2].iloc[0]
    gw3 = out[out["GW"] == 3].iloc[0]
    assert gw1["total_points_cum_prior"] == 0
    assert gw2["total_points_cum_prior"] == 2
    assert gw3["total_points_cum_prior"] == 12  # 2 + 10, not + this row's own 1


def test_add_team_strength_features_uses_correct_home_away_split(tmp_path: Path):
    team_strength = pd.DataFrame(
        [
            {
                "season": "2024-25",
                "id": 2,
                "name": "Beta",
                "strength_attack_home": 1300,
                "strength_attack_away": 1200,
                "strength_defence_home": 1100,
                "strength_defence_away": 1000,
            },
            {
                "season": "2024-25",
                "id": 5,
                "name": "Alpha",
                "strength_attack_home": 1400,
                "strength_attack_away": 1350,
                "strength_defence_home": 1250,
                "strength_defence_away": 1150,
            },
        ]
    )
    df = pd.DataFrame(
        [_row("2024-25", 1, 1, total_points=5, was_home=True, opponent_team=2, team="Alpha")]
    )
    # own team id isn't a column on the player row -- add_team_strength_features
    # joins on team *name*, matching the real merged_gw.csv schema.
    out = features.add_team_strength_features(df, team_strength)
    row = out.iloc[0]

    # was_home=True: own side uses its home strength, opponent uses their away strength.
    assert row["own_attack_strength"] == 1400
    assert row["own_defence_strength"] == 1250
    assert row["opponent_attack_strength"] == 1200
    assert row["opponent_defence_strength"] == 1000


def test_feature_columns_matches_rolling_windows_and_stats():
    cols = features.feature_columns()
    assert "total_points_roll3" in cols
    assert "total_points_roll10" in cols
    assert "value" in cols
    assert "was_home" in cols
    assert "opponent_attack_strength" in cols
