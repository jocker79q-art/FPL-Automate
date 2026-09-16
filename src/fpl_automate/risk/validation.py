"""Historical validation of the risk system: does the `risk_adjusted`
strategy actually reduce the *realized* variance of a manager's points
across real historical gameweeks, not just satisfy the mean-variance math
on paper? Every other module in `risk/` documents its own math carefully;
this is the module that checks the math against reality, in the same
spirit as `projections/ml/backtest.py`'s calibration section (a genuine
finding, surfaced honestly, not tuned to look good).

Methodology: for each gameweek of the ML model's held-out backtest season,
build a synthetic 15-man squad (top-2 GK / top-5 DEF / top-5 MID / top-3
FWD by that gameweek's real ML projection) and run the real, unmodified
`optimization.lineup.optimize_lineup` on it under each strategy -- the
*same* squad and the *same* projections every time, so only the selection
objective varies between strategies. Each strategy is then scored by what
those specific players **actually** scored that gameweek (captain
doubled), never by the projection -- the only honest test of whether
lower risk_aversion volatility claims hold up against real outcomes.

Known simplification, documented rather than hidden: this validates
against a synthetic top-N pool re-picked fresh every gameweek by
projected points, not any real manager's continuous squad under
transfer/budget constraints (that's `transfers/engine.py`'s job, tested
separately). This deliberately isolates a narrower, cleaner question:
*given the same fixed player pool and the same projections, does
choosing the starting XI/captain by the risk_adjusted objective produce
lower realized-outcome variance than the balanced objective?*
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpl_automate.optimization.lineup import LineupOptimizationError, optimize_lineup
from fpl_automate.storage.models import PlayerProjection

SQUAD_SHAPE: dict[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
DEFAULT_RISK_AVERSION_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)


@dataclass(frozen=True)
class StrategyValidation:
    label: str
    risk_aversion: float | None
    mean_realized_points: float
    std_realized_points: float
    n_gameweeks: int
    realized_points_by_gameweek: dict[int, float]


def _build_gameweek_pool(gw_df: pd.DataFrame, team_name_to_id: dict[str, int], row_to_player):
    """Top-N-by-projection players per position, shaped like a real
    15-man squad. Returns None (skip this gameweek) if any position
    doesn't have enough distinct, resolvable players -- e.g. very early
    in a season, or a row whose team name doesn't resolve."""
    squad = []
    projections: dict[int, PlayerProjection] = {}
    actuals: dict[int, float] = {}
    for code, n in SQUAD_SHAPE.items():
        pos_rows = gw_df[gw_df["position"] == code].sort_values("ml_pred", ascending=False).head(n)
        if len(pos_rows) < n:
            return None
        for row in pos_rows.itertuples():
            player = row_to_player(row, team_name_to_id)
            if player is None or player.id in projections:
                return None
            squad.append(player)
            projections[player.id] = PlayerProjection(
                player_id=player.id,
                gameweek=int(row.GW),
                expected_points=round(float(row.ml_pred), 2),
                floor_points=round(float(row.ml_floor), 2),
                ceiling_points=round(float(row.ml_ceiling), 2),
                confidence=0.8,
            )
            actuals[player.id] = float(row.actual)
    return squad, projections, actuals


def _realized_points(starting_xi: list[int], captain_id: int, actuals: dict[int, float]) -> float:
    return sum(actuals[pid] for pid in starting_xi) + actuals[captain_id]  # captain doubles


def run_risk_validation(
    combined_test_df: pd.DataFrame,
    team_name_to_id: dict[str, int],
    risk_aversion_levels: tuple[float, ...] = DEFAULT_RISK_AVERSION_LEVELS,
) -> list[StrategyValidation]:
    """`combined_test_df` needs one row per (played) player-gameweek from
    the ML backtest, with every column `projections.ml.backtest.row_to_player`
    needs plus `ml_pred`/`ml_floor`/`ml_ceiling`/`actual`."""
    from fpl_automate.projections.ml.backtest import row_to_player

    labels = ["balanced", *[f"risk_adjusted (aversion={ra})" for ra in risk_aversion_levels]]
    by_gw: dict[str, dict[int, float]] = {label: {} for label in labels}

    for gw, gw_df in combined_test_df.groupby("GW"):
        pool = _build_gameweek_pool(gw_df, team_name_to_id, row_to_player)
        if pool is None:
            continue
        squad, projections, actuals = pool

        try:
            balanced = optimize_lineup(squad, projections, strategy="balanced")
            by_gw["balanced"][int(gw)] = _realized_points(balanced.starting_xi, balanced.captain_id, actuals)
            for ra, label in zip(risk_aversion_levels, labels[1:], strict=True):
                lineup = optimize_lineup(squad, projections, strategy="risk_adjusted", risk_aversion=ra)
                by_gw[label][int(gw)] = _realized_points(lineup.starting_xi, lineup.captain_id, actuals)
        except LineupOptimizationError:
            continue

    validations = []
    for label in labels:
        values = np.array(list(by_gw[label].values()))
        if len(values) == 0:
            continue
        risk_aversion = float(label.split("=")[1].rstrip(")")) if label.startswith("risk_adjusted") else None
        validations.append(
            StrategyValidation(
                label=label,
                risk_aversion=risk_aversion,
                mean_realized_points=round(float(values.mean()), 2),
                std_realized_points=round(float(values.std(ddof=1)), 2) if len(values) > 1 else 0.0,
                n_gameweeks=len(values),
                realized_points_by_gameweek=by_gw[label],
            )
        )
    return validations


def render_risk_validation_markdown(validations: list[StrategyValidation], test_season: str) -> str:
    lines = [
        "# Risk system validation: does risk_adjusted actually reduce realized variance?",
        "",
        f"Backtested on: {test_season} (the same held-out season `ml/backtest.py` uses).",
        "",
        (
            "Methodology: for each held-out gameweek, builds a synthetic 15-man squad "
            "(top-2 GK / top-5 DEF / top-5 MID / top-3 FWD by that gameweek's ML "
            "projection) and runs the real, unmodified `optimization.lineup."
            "optimize_lineup` on it under each strategy -- the *same* squad and "
            "projections every time, so only the selection objective varies. Scored by "
            "what those players **actually** scored that gameweek (captain doubled), "
            "not by the projection -- the only honest test of whether the risk system "
            "does what it claims. See `risk/validation.py`'s module docstring for the "
            "documented simplification (a synthetic top-N pool, not a real continuous "
            "squad under transfer/budget constraints)."
        ),
        "",
        "| Strategy | Mean realized points | Std dev (realized) | Gameweeks |",
        "|---|---|---|---|",
    ]
    for v in validations:
        lines.append(f"| {v.label} | {v.mean_realized_points} | {v.std_realized_points} | {v.n_gameweeks} |")
    lines.append("")

    balanced = next((v for v in validations if v.label == "balanced"), None)
    risk_adjusted = sorted(
        (v for v in validations if v.risk_aversion is not None), key=lambda v: v.risk_aversion  # type: ignore[arg-type,return-value]
    )
    if balanced and risk_adjusted:
        stds = [v.std_realized_points for v in risk_adjusted]
        monotonic_decrease = all(a >= b for a, b in itertools.pairwise(stds))
        lowest_std = min(risk_adjusted, key=lambda v: v.std_realized_points)
        lines += [
            "## Verdict",
            "",
        ]
        if monotonic_decrease:
            lines.append(
                "Realized std dev decreases monotonically as risk_aversion increases across "
                "the tested levels -- matching the theoretical claim on this backtest."
            )
        else:
            lines.append(
                "Realized std dev does **not** decrease monotonically as risk_aversion "
                "increases across every level tested -- a genuine finding, not smoothed "
                "over. Plausible reasons, not mutually exclusive: the independent-variance "
                "simplification (`risk/portfolio.py`, partially addressed by "
                "`risk/covariance.py`) means the *objective* being optimised isn't the "
                "true realized variance to begin with; and a single test season's worth "
                "of gameweeks is a small sample for estimating realized variance "
                "precisely, so noise in this specific measurement is expected."
            )
        lines += [
            "",
            (
                f"- Lowest realized variance: **{lowest_std.label}** (std "
                f"{lowest_std.std_realized_points}, mean {lowest_std.mean_realized_points})."
            ),
            (
                f"- Balanced (no risk penalty): std {balanced.std_realized_points}, mean "
                f"{balanced.mean_realized_points}."
            ),
            (
                "- Mean realized points at the lowest-variance risk_adjusted setting vs. "
                f"balanced: {lowest_std.mean_realized_points - balanced.mean_realized_points:+.2f} "
                "(the real cost, or lack of one, of choosing lower variance on this backtest)."
            ),
        ]
    lines.append("")
    return "\n".join(lines)
