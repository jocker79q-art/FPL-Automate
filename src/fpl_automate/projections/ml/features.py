"""Leak-free feature engineering for the ML projection model.

The one rule everything here is built around: every feature used to
predict gameweek N's points must be computable from information available
*before* gameweek N's deadline. Concretely:

  - Rolling/lag stats (points, minutes, underlying xG/xA, ICT, bps, ...)
    are computed with `.shift(1)` before the rolling window, so gameweek
    N's row never sees gameweek N's own result -- only strictly prior
    gameweeks. Get this wrong and the model looks great offline and is
    useless in production, because it's silently cheating on data that
    wouldn't exist yet at the real prediction time.
  - `value` (price), `was_home`, and `opponent_team` are legitimately
    known ahead of the deadline -- using them isn't leakage.
  - `xP` is FPL's own official pre-match expected-points estimate,
    carried through unchanged -- not used as a model input, but kept as
    the benchmark the trained model needs to beat (see
    `ml/backtest.py`).

Known simplification, shared with `projections/baseline_model.py`'s own
documented assumptions: opponent/team strength ratings are read from the
season's teams.csv as fetched (a point-in-time snapshot), not as they
stood before each individual gameweek.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROLLING_WINDOWS = [3, 5, 10]
ROLLING_STATS = [
    "total_points",
    "minutes",
    "starts",
    "ict_index",
    "influence",
    "creativity",
    "threat",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "goals_scored",
    "assists",
    "clean_sheets",
    "saves",
    "bonus",
]

TARGET_COL = "total_points"
BASELINE_COL = "xP"
PLAYED_COL = "minutes"

TEAM_STRENGTH_COLS = [
    "strength_attack_home",
    "strength_attack_away",
    "strength_defence_home",
    "strength_defence_away",
]

STRENGTH_FEATURE_COLS = [
    "opponent_attack_strength",
    "opponent_defence_strength",
    "own_attack_strength",
    "own_defence_strength",
]


def load_team_strength(historical_dir: Path, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = historical_dir / season / "teams.csv"
        df = pd.read_csv(path)
        df["season"] = season
        frames.append(df[["season", "id", "name", *TEAM_STRENGTH_COLS]])
    return pd.concat(frames, ignore_index=True)


def add_team_strength_features(df: pd.DataFrame, team_strength: pd.DataFrame) -> pd.DataFrame:
    """Attach both sides' attack/defence strength, not just a single
    "opponent difficulty" scalar: a striker's expected return depends on
    their own team's attacking strength as much as the opponent's
    defence, and a defender/keeper's clean-sheet odds depend on the
    reverse pairing.
    """
    opp = team_strength.rename(
        columns={"id": "opponent_team", **{c: f"opp_{c}" for c in TEAM_STRENGTH_COLS}}
    ).drop(columns=["name"])
    merged = df.merge(opp, on=["season", "opponent_team"], how="left")

    own = team_strength.rename(
        columns={"name": "team", **{c: f"own_{c}" for c in TEAM_STRENGTH_COLS}}
    ).drop(columns=["id"])
    merged = merged.merge(own, on=["season", "team"], how="left")

    was_home = merged["was_home"]
    # The opponent plays the opposite venue to this player's team.
    merged["opponent_attack_strength"] = np.where(
        was_home, merged["opp_strength_attack_away"], merged["opp_strength_attack_home"]
    )
    merged["opponent_defence_strength"] = np.where(
        was_home, merged["opp_strength_defence_away"], merged["opp_strength_defence_home"]
    )
    merged["own_attack_strength"] = np.where(
        was_home, merged["own_strength_attack_home"], merged["own_strength_attack_away"]
    )
    merged["own_defence_strength"] = np.where(
        was_home, merged["own_strength_defence_home"], merged["own_strength_defence_away"]
    )

    drop_cols = [f"opp_{c}" for c in TEAM_STRENGTH_COLS] + [f"own_{c}" for c in TEAM_STRENGTH_COLS]
    return merged.drop(columns=drop_cols)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "element", "GW"]).reset_index(drop=True)
    group_keys = ["season", "element"]
    grouped = df.groupby(group_keys, sort=False)

    # Shift every stat at once (one vectorized groupby op) so no rolling
    # window below can ever see a gameweek's own result.
    shifted = grouped[ROLLING_STATS].shift(1)
    shifted[group_keys] = df[group_keys]
    shifted_grouped = shifted.groupby(group_keys, sort=False)

    roll_frames = []
    for window in ROLLING_WINDOWS:
        rolled = shifted_grouped[ROLLING_STATS].rolling(window, min_periods=1).mean()
        rolled = rolled.reset_index(level=group_keys, drop=True)
        rolled.columns = [f"{stat}_roll{window}" for stat in ROLLING_STATS]
        roll_frames.append(rolled)

    df = pd.concat([df, *roll_frames], axis=1)
    df["games_played_so_far"] = grouped.cumcount()

    roll_cols = [f"{stat}_roll{w}" for w in ROLLING_WINDOWS for stat in ROLLING_STATS]
    # A player's first appearance of a season has no prior gameweeks to
    # roll over; games_played_so_far == 0 flags these rows so the model
    # can weigh them differently rather than silently dropping them.
    df[roll_cols] = df[roll_cols].fillna(0)
    return df


# Season-aggregate fields (as opposed to rolling-window form) needed to
# reconstruct a point-in-time `storage.models.Player` for backtesting
# `projections/baseline_model.py` against historical data (see
# `ml/backtest.py`) -- distinct from ROLLING_STATS/feature_columns above,
# which feed the ML model itself, not the baseline.
CUMULATIVE_STATS = [
    "total_points",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "bonus",
    "bps",
    "saves",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]


def add_cumulative_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `{stat}_cum_prior` = the season-to-date total of `stat` using
    only strictly prior gameweeks (shift(1) before cumsum, same leakage
    discipline as `add_rolling_features`) -- i.e. exactly what a live
    bootstrap-static season-aggregate field would have shown immediately
    before this gameweek's deadline.
    """
    df = df.sort_values(["season", "element", "GW"]).reset_index(drop=True)
    group_keys = ["season", "element"]
    grouped = df.groupby(group_keys, sort=False)

    shifted = grouped[CUMULATIVE_STATS].shift(1).fillna(0)
    shifted[group_keys] = df[group_keys]
    cum = shifted.groupby(group_keys, sort=False)[CUMULATIVE_STATS].cumsum()
    cum.columns = [f"{stat}_cum_prior" for stat in CUMULATIVE_STATS]

    return pd.concat([df, cum], axis=1)


def feature_columns() -> list[str]:
    roll_cols = [f"{stat}_roll{w}" for w in ROLLING_WINDOWS for stat in ROLLING_STATS]
    return roll_cols + ["value", "was_home", "games_played_so_far", *STRENGTH_FEATURE_COLS]


def build_training_frame(df: pd.DataFrame, historical_dir: Path, seasons: list[str]) -> pd.DataFrame:
    """Turn raw merged-gameweek rows into a model-ready frame.

    `df` is the concatenated historical data (`ml.historical.load_merged_gw`
    output); `seasons` must match the seasons present in `df` (used to
    load matching teams.csv files for team strength).
    """
    team_strength = load_team_strength(historical_dir, seasons)
    out = add_team_strength_features(df, team_strength)
    out = add_rolling_features(out)
    out["was_home"] = out["was_home"].astype(int)
    return out
