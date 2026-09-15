"""Live inference using the trained ML model (docs/ROADMAP.md Phase 2),
wired into `workflow.py::compute_projections` alongside
`projections/baseline_model.py` -- never a silent replacement: every
ML-sourced `PlayerProjection.rationale` names the model, and any player
the ML model can't cover (see `compute_ml_projections`'s per-player gaps,
or its `None` return when no trained model exists at all) is the
caller's cue to fall back to the baseline for that player, not an error.

The core trick: fetch the *current* season's merged_gw.csv from the same
vaastav archive `ml/historical.py` already uses for training data.
vaastav updates that file gameweek by gameweek as real results land, so
it doubles as free, already-cached per-gameweek rolling history for the
in-progress season -- no need for hundreds of individual FPL
element-summary calls per run (which `features/engineering.py` already
documents as deliberately avoided for exactly this politeness reason). A
synthetic "row to predict" is appended per player for each future
gameweek needed and run through the *exact* same leak-free feature
pipeline used at training time (`ml/features.py`), so rolling features
for an unplayed gameweek are computed purely from that player's real
prior gameweeks this season -- never from itself.

Multi-gameweek horizons are a genuine per-gameweek walk (score each
gameweek's real fixture, then sum), not the baseline's own
average-difficulty-times-fixture-count simplification -- team-strength
and venue are re-derived per gameweek from that gameweek's real fixture,
reusing the frozen rolling-form features (there's nothing new to roll
over for a gameweek that hasn't been played yet).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fpl_automate.projections.ml import historical
from fpl_automate.projections.ml.features import (
    STRENGTH_FEATURE_COLS,
    build_training_frame,
    feature_columns,
)
from fpl_automate.projections.ml.model import POSITION_TO_CODE, load_calibration, load_models
from fpl_automate.storage.models import Fixture, Player, PlayerProjection, Team

logger = logging.getLogger(__name__)

CLEARLY_OUT_STATUSES = {"i", "s", "u"}  # injured, suspended, unavailable
SMALL_SAMPLE_GAMES = 6


def apply_live_availability(p_play: float, status: str, chance_of_playing_next_round: int | None) -> float:
    """Overrides the ML classifier's P(plays) with FPL's own real-time
    news signal when one is actually published -- the ML classifier, like
    the baseline without this override, only ever sees historical
    minutes/starts, never real-time team news.

    - FPL publishes a percentage (0-100) whenever there's genuine doubt:
      trust it outright over the model's own estimate.
    - No percentage published, but flagged injured/suspended/unavailable:
      FPL's shorthand for "definitely not playing" -- treat as certain.
    - Otherwise: no live signal to override with, keep the model's own
      estimate.
    """
    if chance_of_playing_next_round is not None:
        return chance_of_playing_next_round / 100
    if status in CLEARLY_OUT_STATUSES:
        return 0.0
    return p_play


def _team_names_by_id(teams: list[Team]) -> dict[int, str]:
    return {t.id: t.name for t in teams}


def _team_strength_by_id(teams: list[Team]) -> dict[int, dict]:
    return {
        t.id: {
            "attack_home": t.strength_attack_home,
            "attack_away": t.strength_attack_away,
            "defence_home": t.strength_defence_home,
            "defence_away": t.strength_defence_away,
        }
        for t in teams
    }


def _fixture_by_team_for_event(fixtures: list[Fixture], event: int) -> dict[int, dict]:
    """team_id -> {opponent_team, was_home} for one gameweek's fixtures. A
    team missing from this dict has a blank gameweek that gameweek; a
    team appearing in more than one fixture (double gameweek) only keeps
    the first -- a documented simplification, same spirit as
    baseline_model's own `num_fixtures_next_n` handling."""
    by_team: dict[int, dict] = {}
    for fx in fixtures:
        if fx.event != event:
            continue
        if fx.team_h not in by_team:
            by_team[fx.team_h] = {"opponent_team": fx.team_a, "was_home": True}
        if fx.team_a not in by_team:
            by_team[fx.team_a] = {"opponent_team": fx.team_h, "was_home": False}
    return by_team


def _strength_features_for_fixture(
    team_strength_by_id: dict[int, dict], own_team_id: int, opponent_team_id: int | None, was_home: bool
) -> dict[str, float] | None:
    if opponent_team_id is None or own_team_id not in team_strength_by_id or opponent_team_id not in team_strength_by_id:
        return None
    own = team_strength_by_id[own_team_id]
    opp = team_strength_by_id[opponent_team_id]
    if was_home:
        return {
            "own_attack_strength": own["attack_home"],
            "own_defence_strength": own["defence_home"],
            "opponent_attack_strength": opp["attack_away"],
            "opponent_defence_strength": opp["defence_away"],
        }
    return {
        "own_attack_strength": own["attack_away"],
        "own_defence_strength": own["defence_away"],
        "opponent_attack_strength": opp["attack_home"],
        "opponent_defence_strength": opp["defence_home"],
    }


def build_prediction_rows(
    players: list[Player], fixture_by_team: dict[int, dict], event: int, team_names: dict[int, str]
) -> pd.DataFrame:
    """One row per player for the (unplayed) given gameweek, with only
    legitimately-known-ahead fields populated. Everything in
    `ml.features.ROLLING_STATS` is deliberately left absent so
    `build_training_frame`'s rolling/shift logic treats this row as
    having no result of its own."""
    rows = []
    for p in players:
        fx = fixture_by_team.get(p.team_id)
        code = POSITION_TO_CODE.get(p.position)
        if code is None:
            continue
        rows.append(
            {
                "season": historical.CURRENT_SEASON,
                "element": p.id,
                "name": p.web_name,
                "position": code,
                "GW": event,
                "team": team_names.get(p.team_id, "?"),
                "team_id": p.team_id,
                "opponent_team": fx["opponent_team"] if fx else None,
                "was_home": fx["was_home"] if fx else False,
                "value": p.now_cost_tenths,
                "blank_gameweek": fx is None,
                "is_prediction_row": True,
            }
        )
    return pd.DataFrame(rows)


def _strength_overrides_for_gameweek(
    group: pd.DataFrame, fixture_by_team: dict[int, dict], team_strength_by_id: dict[int, dict]
) -> pd.DataFrame:
    """Vectorized: one row per *team* in `group` (not per player) giving
    the 4 strength features + was_home + blank_gameweek for a specific
    future gameweek, ready to left-merge onto `group` on team_id --
    looping over ~20 teams instead of ~700 players keeps this fast."""
    rows = []
    for team_id in group["team_id"].unique():
        team_id = int(team_id)
        fx = fixture_by_team.get(team_id)
        feats = (
            _strength_features_for_fixture(team_strength_by_id, team_id, fx["opponent_team"], fx["was_home"])
            if fx is not None
            else None
        )
        if fx is None or feats is None:
            rows.append(
                {
                    "team_id": team_id,
                    "own_attack_strength": float("nan"),
                    "own_defence_strength": float("nan"),
                    "opponent_attack_strength": float("nan"),
                    "opponent_defence_strength": float("nan"),
                    "was_home": False,
                    "blank_gameweek": True,
                }
            )
        else:
            rows.append({"team_id": team_id, **feats, "was_home": int(fx["was_home"]), "blank_gameweek": False})
    return pd.DataFrame(rows)


def compute_ml_projections(
    players: list[Player],
    teams: list[Team],
    fixtures: list[Fixture],
    from_event: int,
    horizons: tuple[int, ...],
    historical_dir: Path,
    models_dir: Path,
) -> dict[int, dict[int, PlayerProjection]] | None:
    """Returns {horizon: {player_id: PlayerProjection}}. Returns None
    entirely (never partial-None) only when no trained model exists at
    all yet -- a fresh checkout before `train-model`/`backtest` has ever
    run is a normal, expected state, and `compute_projections` in
    workflow.py falls back to the baseline for every player in that
    case. Once models exist, a player missing a position's model file,
    or whose team can't be resolved, is simply absent from every
    horizon's dict -- again the caller's cue to fall back to the
    baseline for that one player, not an error for the whole run.
    """
    models = load_models(models_dir)
    if not models:
        return None
    calibration = load_calibration(models_dir)

    max_horizon = max(horizons)
    gameweeks = list(range(from_event, from_event + max_horizon))

    historical.fetch_season(historical.CURRENT_SEASON, historical_dir)
    current_history = historical.load_merged_gw(historical_dir, [historical.CURRENT_SEASON])

    team_names = _team_names_by_id(teams)
    team_strength_by_id = _team_strength_by_id(teams)
    fixture_by_team_by_gw = {gw: _fixture_by_team_for_event(fixtures, gw) for gw in gameweeks}

    prediction_rows = build_prediction_rows(players, fixture_by_team_by_gw[from_event], from_event, team_names)
    if prediction_rows.empty:
        return None

    combined = pd.concat([current_history, prediction_rows], ignore_index=True, sort=False)
    combined["was_home"] = combined["was_home"].fillna(False)
    feat = build_training_frame(combined, historical_dir, [historical.CURRENT_SEASON])

    to_predict = feat[feat["is_prediction_row"].fillna(False)].copy()
    cols = feature_columns()

    # gw -> player_id -> {p_play, pgp, blank}
    gw_points: dict[int, dict[int, dict]] = {gw: {} for gw in gameweeks}
    player_position: dict[int, str] = {}
    player_games_played: dict[int, int] = {}

    for position, group in to_predict.groupby("position"):
        pos_model = models.get(position)
        if pos_model is None:
            logger.warning("no trained ML model for position %s, skipping %d players", position, len(group))
            continue
        group = group.reset_index(drop=True)
        base_X = group[cols].copy()

        for gw in gameweeks:
            if gw == from_event:
                p_play, pgp = pos_model.predict(base_X)
                blank = group["blank_gameweek"].to_numpy()
            else:
                overrides = _strength_overrides_for_gameweek(
                    group, fixture_by_team_by_gw[gw], team_strength_by_id
                )
                merged = group[["team_id"]].merge(overrides, on="team_id", how="left")
                X_gw = base_X.copy()
                for col in [*STRENGTH_FEATURE_COLS, "was_home"]:
                    X_gw[col] = merged[col].to_numpy()
                p_play, pgp = pos_model.predict(X_gw)
                blank = merged["blank_gameweek"].to_numpy()

            for i in range(len(group)):
                player_id = int(group.loc[i, "element"])
                gw_points[gw][player_id] = {
                    "p_play": float(p_play.iloc[i]),
                    "pgp": float(pgp.iloc[i]),
                    "blank": bool(blank[i]),
                }

        for i in range(len(group)):
            player_id = int(group.loc[i, "element"])
            player_position[player_id] = position
            player_games_played[player_id] = int(group.loc[i, "games_played_so_far"])

    live_status_by_id = {
        p.id: (p.availability.status, p.availability.chance_of_playing_next_round) for p in players
    }

    result: dict[int, dict[int, PlayerProjection]] = {h: {} for h in horizons}
    for player_id, position in player_position.items():
        status, chance = live_status_by_id.get(player_id, ("a", None))
        overridden = False

        per_gw_points: dict[int, float] = {}
        from_event_p_play = 0.0
        for gw in gameweeks:
            entry = gw_points[gw].get(player_id)
            if entry is None or entry["blank"]:
                per_gw_points[gw] = 0.0
                continue
            p_play = entry["p_play"]
            if gw == from_event:
                p_play = apply_live_availability(p_play, status, chance)
                overridden = chance is not None or status in CLEARLY_OUT_STATUSES
                from_event_p_play = p_play
            per_gw_points[gw] = p_play * entry["pgp"]

        gw1_entry = gw_points[from_event].get(player_id)
        blank_gw1 = gw1_entry is None or gw1_entry["blank"]
        # Confidence reflects the *actual* P(plays) used for the nearest
        # gameweek -- i.e. after any live-availability override, not the
        # classifier's raw (potentially stale) estimate.
        confidence = round(from_event_p_play, 2) if not blank_gw1 else 0.9
        games_played = player_games_played.get(player_id, 0)

        risk_flags: list[str] = []
        if blank_gw1:
            risk_flags.append("blank_gameweek")
        if games_played < SMALL_SAMPLE_GAMES:
            risk_flags.append("small_sample")
        if overridden:
            risk_flags.append("injury_or_availability_doubt")
        if not blank_gw1 and confidence < 0.5:
            risk_flags.append("rotation_risk")

        band = calibration.get(position, {"residual_p10": 0.0, "residual_p90": 0.0})

        for horizon in horizons:
            included_gws = gameweeks[:horizon]
            expected = round(sum(per_gw_points[gw] for gw in included_gws), 2)
            n_gws = len(included_gws)
            floor_pts = round(max(0.0, expected + n_gws * band["residual_p10"]), 2)
            ceiling_pts = round(max(floor_pts, expected + n_gws * band["residual_p90"]), 2)
            result[horizon][player_id] = PlayerProjection(
                player_id=player_id,
                gameweek=horizon,
                expected_points=expected,
                floor_points=floor_pts,
                ceiling_points=ceiling_pts,
                confidence=confidence,
                risk_flags=risk_flags,
                rationale=(
                    f"ML model (two-stage hurdle): p(plays GW{from_event})={confidence:.2f}, "
                    f"summed over {n_gws} gameweek(s) from real per-gameweek fixtures "
                    f"(not the baseline's average-difficulty simplification)."
                ),
            )

    return result
