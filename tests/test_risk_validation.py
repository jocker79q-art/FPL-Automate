from __future__ import annotations

import pandas as pd

from fpl_automate.optimization.lineup import optimize_lineup
from fpl_automate.projections.ml.backtest import row_to_player
from fpl_automate.risk.validation import (
    StrategyValidation,
    _build_gameweek_pool,
    render_risk_validation_markdown,
    run_risk_validation,
)

_ROW_DEFAULTS = {
    "total_points_roll5": 4.0,
    "total_points_cum_prior": 40.0,
    "games_played_so_far": 10,
    "minutes_cum_prior": 900,
    "starts_cum_prior": 10,
    "goals_scored_cum_prior": 2,
    "assists_cum_prior": 2,
    "clean_sheets_cum_prior": 3,
    "goals_conceded_cum_prior": 8,
    "bonus_cum_prior": 4,
    "bps_cum_prior": 200,
    "saves_cum_prior": 0,
    "expected_goals_cum_prior": 2.0,
    "expected_assists_cum_prior": 2.0,
    "expected_goal_involvements_cum_prior": 4.0,
    "expected_goals_conceded_cum_prior": 8.0,
    "value": 55,
}


def _row(element: int, name: str, position: str, team: str, opponent_team: int, gw: int, ml_pred, ml_floor, ml_ceiling, actual):
    row = {
        "element": element,
        "name": name,
        "position": position,
        "team": team,
        "opponent_team": opponent_team,
        "GW": gw,
        "season": "2099-00",
        "ml_pred": ml_pred,
        "ml_floor": ml_floor,
        "ml_ceiling": ml_ceiling,
        "actual": actual,
    }
    row.update(_ROW_DEFAULTS)
    return row


def _squad_rows(gw: int, volatile_actual: float) -> list[dict]:
    """15 players (2 GK / 5 DEF / 5 MID / 3 FWD) on team Alpha (opponent
    Beta=20), all with a tight/safe projection band except one MID
    ("Volatile") who has the *highest* expected points but a very wide
    floor-to-ceiling band -- so `balanced` should captain them, while
    `risk_adjusted` at a high enough aversion should captain a safer
    alternative instead. `volatile_actual` controls what they actually
    score that gameweek, to test both a "boom" and a "bust" outcome."""
    rows = []
    element = 1
    for _ in range(2):
        rows.append(_row(element, f"GK{element}", "GK", "Alpha", 20, gw, 4.0, 3.5, 4.5, 4.0))
        element += 1
    for _ in range(5):
        rows.append(_row(element, f"DEF{element}", "DEF", "Alpha", 20, gw, 4.0, 3.5, 4.5, 4.0))
        element += 1
    for _ in range(4):
        rows.append(_row(element, f"MID{element}", "MID", "Alpha", 20, gw, 4.0, 3.5, 4.5, 4.0))
        element += 1
    # The volatile captaincy candidate: highest expected points (8.0) but
    # an enormous band (0 to 16), so its implied std dev dwarfs every
    # other player's.
    rows.append(_row(element, "Volatile", "MID", "Alpha", 20, gw, 8.0, 0.0, 16.0, volatile_actual))
    element += 1
    for _ in range(3):
        rows.append(_row(element, f"FWD{element}", "FWD", "Alpha", 20, gw, 4.0, 3.5, 4.5, 4.0))
        element += 1
    return rows


TEAM_NAME_TO_ID = {"Alpha": 10, "Beta": 20}


VOLATILE_ID = 12  # see _squad_rows: 2 GK + 5 DEF + 4 MID (elements 1-11) come before "Volatile"


class TestBuildGameweekPoolAndCaptaincy:
    def test_balanced_captains_the_highest_expected_points_player(self):
        df = pd.DataFrame(_squad_rows(gw=1, volatile_actual=16.0))
        squad, projections, _actuals = _build_gameweek_pool(df, TEAM_NAME_TO_ID, row_to_player)
        lineup = optimize_lineup(squad, projections, strategy="balanced")
        assert lineup.captain_id == VOLATILE_ID

    def test_high_risk_aversion_captains_a_safe_player_instead(self):
        df = pd.DataFrame(_squad_rows(gw=1, volatile_actual=16.0))
        squad, projections, _actuals = _build_gameweek_pool(df, TEAM_NAME_TO_ID, row_to_player)
        lineup = optimize_lineup(squad, projections, strategy="risk_adjusted", risk_aversion=5.0)
        assert lineup.captain_id != VOLATILE_ID


class TestRunRiskValidation:
    def test_high_risk_aversion_beats_balanced_on_a_captain_bust(self):
        # Volatile busts to 0 actual points -- a real "risky captain backfires" case.
        df = pd.DataFrame(_squad_rows(gw=1, volatile_actual=0.0))
        validations = run_risk_validation(df, TEAM_NAME_TO_ID, risk_aversion_levels=(5.0,))
        by_label = {v.label: v for v in validations}

        assert by_label["balanced"].n_gameweeks == 1
        balanced_realized = by_label["balanced"].realized_points_by_gameweek[1]
        risk_adjusted_realized = by_label["risk_adjusted (aversion=5.0)"].realized_points_by_gameweek[1]

        # balanced still captains Volatile (busts to 0, doubled = still 0)
        # while risk_adjusted's high aversion captains a safe player
        # instead -- a real, measurable realized-points advantage from
        # the risk system on this exact bust scenario, through the real
        # optimizer end to end (not just the isolated scoring formula).
        assert risk_adjusted_realized > balanced_realized

    def test_skips_gameweeks_without_enough_players_per_position(self):
        # Only 3 players this gameweek -- nowhere near the 15-man shape.
        df = pd.DataFrame(
            [
                _row(1, "GK1", "GK", "Alpha", 20, 1, 4.0, 3.5, 4.5, 4.0),
                _row(2, "DEF1", "DEF", "Alpha", 20, 1, 4.0, 3.5, 4.5, 4.0),
                _row(3, "MID1", "MID", "Alpha", 20, 1, 4.0, 3.5, 4.5, 4.0),
            ]
        )
        validations = run_risk_validation(df, TEAM_NAME_TO_ID, risk_aversion_levels=(1.0,))
        assert validations == []


class TestRenderRiskValidationMarkdown:
    def test_reports_monotonic_decrease_when_it_holds(self):
        validations = [
            StrategyValidation("balanced", None, 60.0, 10.0, 20, {}),
            StrategyValidation("risk_adjusted (aversion=0.5)", 0.5, 59.0, 8.0, 20, {}),
            StrategyValidation("risk_adjusted (aversion=2.0)", 2.0, 57.0, 5.0, 20, {}),
        ]
        md = render_risk_validation_markdown(validations, "2024-25")
        assert "decreases monotonically" in md
        assert "does **not** decrease" not in md

    def test_reports_honestly_when_monotonic_decrease_does_not_hold(self):
        validations = [
            StrategyValidation("balanced", None, 60.0, 10.0, 20, {}),
            StrategyValidation("risk_adjusted (aversion=0.5)", 0.5, 59.0, 8.0, 20, {}),
            StrategyValidation("risk_adjusted (aversion=2.0)", 2.0, 57.0, 9.0, 20, {}),  # went back up
        ]
        md = render_risk_validation_markdown(validations, "2024-25")
        assert "does **not** decrease" in md

    def test_handles_empty_validations_without_crashing(self):
        md = render_risk_validation_markdown([], "2024-25")
        assert "Risk system validation" in md
        assert "## Verdict" not in md
