"""Walk-forward backtest: ML model vs `projections/baseline_model.py` vs
FPL's own official xP, all evaluated on the same held-out season.

Methodology (see docs/ROADMAP.md "Phase 2"):
  - Chronological split: the ML model trains on every season before
    `TEST_SEASON` and is evaluated only on `TEST_SEASON` -- it never sees
    test-season data during training.
  - The baseline model is *not* trained at all (it has no learned
    parameters), but to score it fairly on historical data this module
    reconstructs, for every (player, gameweek) row in the test season, a
    point-in-time `storage.models.Player` built only from that player's
    strictly-prior season-to-date aggregates -- i.e. exactly what
    `bootstrap-static` would have shown immediately before that
    gameweek's real deadline -- and calls the real, unmodified
    `projections.baseline_model.project_player` on it. This is
    deliberately the actual production function, not a re-derived
    approximation, so the comparison can't silently drift from what the
    app really does.
  - Known simplification shared with the point-in-time reconstruction:
    per-gameweek historical availability/injury status and set-piece
    order are not available in the archived dataset, so every
    reconstructed Player is treated as fully available with no
    penalty/freekick/corner duty. This can only ever help the baseline's
    accuracy (no simulated uncertainty it doesn't already model), so it's
    a conservative bias against the ML model, not for it.
"""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fpl_automate.features.engineering import compute_player_features
from fpl_automate.projections.baseline_model import ModelInputs, project_player
from fpl_automate.projections.ml import historical
from fpl_automate.projections.ml.features import (
    BASELINE_COL,
    TARGET_COL,
    add_cumulative_prior_features,
    build_training_frame,
    feature_columns,
)
from fpl_automate.projections.ml.model import POSITIONS, fit_position_model, mixture_floor_ceiling
from fpl_automate.projections.ml.model import save_models as persist_models
from fpl_automate.risk.covariance import estimate_fixture_correlation
from fpl_automate.risk.covariance import save_correlation as persist_correlation
from fpl_automate.risk.validation import (
    StrategyValidation,
    render_risk_validation_markdown,
    run_risk_validation,
)
from fpl_automate.storage.models import AvailabilityStatus, Fixture, Player, Position, Team

logger = logging.getLogger(__name__)

CODE_TO_POSITION: dict[str, Position] = {
    "GK": Position.GOALKEEPER,
    "DEF": Position.DEFENDER,
    "MID": Position.MIDFIELDER,
    "FWD": Position.FORWARD,
}

NAIVE_FORM_COL = "total_points_roll5"


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "n": len(y_true),
    }


def _build_teams(historical_dir: Path, season: str) -> dict[int, Team]:
    df = pd.read_csv(historical_dir / season / "teams.csv")
    teams: dict[int, Team] = {}
    for row in df.itertuples():
        teams[row.id] = Team(
            id=row.id,
            name=row.name,
            short_name=row.short_name,
            strength_overall_home=row.strength_overall_home,
            strength_overall_away=row.strength_overall_away,
            strength_attack_home=row.strength_attack_home,
            strength_attack_away=row.strength_attack_away,
            strength_defence_home=row.strength_defence_home,
            strength_defence_away=row.strength_defence_away,
        )
    return teams


def _build_fixtures(historical_dir: Path, season: str) -> list[Fixture]:
    df = pd.read_csv(historical_dir / season / "fixtures.csv")
    fixtures: list[Fixture] = []
    for row in df.itertuples():
        event = None if pd.isna(row.event) else int(row.event)
        fixtures.append(
            Fixture(
                id=row.id,
                event=event,
                team_h=row.team_h,
                team_a=row.team_a,
                team_h_difficulty=row.team_h_difficulty,
                team_a_difficulty=row.team_a_difficulty,
                kickoff_time=None,
                finished=bool(row.finished),
            )
        )
    return fixtures


def row_to_player(row, team_name_to_id: dict[str, int]) -> Player | None:
    position = CODE_TO_POSITION.get(row.position)
    team_id = team_name_to_id.get(row.team)
    if position is None or team_id is None:
        return None

    games_played = max(1, int(row.games_played_so_far))
    return Player(
        id=int(row.element),
        web_name=str(row.name),
        full_name=str(row.name),
        team_id=team_id,
        position=position,
        now_cost_tenths=int(row.value),
        selected_by_percent=0.0,
        availability=AvailabilityStatus(status="a", chance_of_playing_next_round=None, news=""),
        # `form` is FPL's own recent-form figure; the rolling-5 average of
        # strictly prior gameweeks is the closest point-in-time equivalent
        # reconstructable from archived data.
        form=round(float(row.total_points_roll5), 2),
        points_per_game=round(float(row.total_points_cum_prior) / games_played, 2),
        total_points=int(row.total_points_cum_prior),
        minutes=int(row.minutes_cum_prior),
        starts=int(row.starts_cum_prior),
        goals_scored=int(row.goals_scored_cum_prior),
        assists=int(row.assists_cum_prior),
        clean_sheets=int(row.clean_sheets_cum_prior),
        goals_conceded=int(row.goals_conceded_cum_prior),
        bonus=int(row.bonus_cum_prior),
        bps=int(row.bps_cum_prior),
        yellow_cards=0,
        red_cards=0,
        saves=int(row.saves_cum_prior),
        expected_goals=round(float(row.expected_goals_cum_prior), 2),
        expected_assists=round(float(row.expected_assists_cum_prior), 2),
        expected_goal_involvements=round(float(row.expected_goal_involvements_cum_prior), 2),
        expected_goals_conceded=round(float(row.expected_goals_conceded_cum_prior), 2),
    )


def _backtest_baseline(
    test_df: pd.DataFrame, historical_dir: Path, season: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (predicted, floor, ceiling, confidence) arrays aligned to
    test_df's row order, using the real baseline_model.project_player on
    point-in-time-reconstructed Players."""
    teams = _build_teams(historical_dir, season)
    team_name_to_id = {t.name: tid for tid, t in teams.items()}
    fixtures = _build_fixtures(historical_dir, season)

    n = len(test_df)
    predicted = np.zeros(n)
    floor = np.zeros(n)
    ceiling = np.zeros(n)
    confidence = np.zeros(n)

    for i, row in enumerate(test_df.itertuples()):
        player = row_to_player(row, team_name_to_id)
        if player is None:
            continue
        team = teams.get(player.team_id)
        if team is None:
            continue
        features = compute_player_features(
            player, team, fixtures, from_event=int(row.GW), fixture_window_gameweeks=1
        )
        projection = project_player(ModelInputs(player=player, features=features, horizon_gameweeks=1))
        predicted[i] = projection.expected_points
        floor[i] = projection.floor_points
        ceiling[i] = projection.ceiling_points
        confidence[i] = projection.confidence

    return predicted, floor, ceiling, confidence


def _calibration_report(actual: np.ndarray, floor: np.ndarray, ceiling: np.ndarray, confidence: np.ndarray) -> dict:
    """Checks whether the baseline model's floor/ceiling band actually
    captures the real range of outcomes, and whether that coverage tracks
    the model's own stated confidence -- directly answering
    docs/ROADMAP.md Phase 2's calibration question."""
    within_band = (actual >= floor) & (actual <= ceiling)
    overall_coverage = round(float(within_band.mean()), 3)

    buckets = []
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    for lo, hi in itertools.pairwise(edges):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        buckets.append(
            {
                "confidence_range": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "n": int(mask.sum()),
                "band_coverage": round(float(within_band[mask].mean()), 3),
                "mean_confidence": round(float(confidence[mask].mean()), 3),
            }
        )

    return {"overall_band_coverage": overall_coverage, "by_confidence_bucket": buckets}


def run_backtest(
    historical_dir: Path,
    seasons: list[str] | None = None,
    test_season: str | None = None,
    save_models: bool = True,
    models_dir: Path | None = None,
) -> dict:
    seasons = seasons or historical.DEFAULT_SEASONS
    test_season = test_season or historical.TEST_SEASON
    if test_season not in seasons:
        raise ValueError(f"test_season {test_season!r} must be included in seasons")

    raw = historical.load_merged_gw(historical_dir, seasons)
    if raw.empty:
        raise RuntimeError(
            "No historical data found -- run `fetch-historical-data` (or "
            "historical.fetch_all_seasons) first."
        )
    feat = build_training_frame(raw, historical_dir, seasons)
    feat = add_cumulative_prior_features(feat)
    cols = feature_columns()

    train_df = feat[feat["season"] != test_season]
    test_df = feat[feat["season"] == test_season]

    results: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seasons_trained_on": [s for s in seasons if s != test_season],
        "test_season": test_season,
        "feature_columns": cols,
        "positions": {},
    }
    overall: dict[str, list[np.ndarray]] = {"y": [], "ml": [], "baseline": [], "xp": [], "naive": []}
    baseline_calibration_inputs: dict[str, list[np.ndarray]] = {
        "actual": [], "floor": [], "ceiling": [], "confidence": []
    }
    ml_calibration_inputs: dict[str, list[np.ndarray]] = {
        "actual": [], "floor": [], "ceiling": [], "confidence": []
    }

    if save_models:
        models_dir = models_dir or (historical_dir.parent / "models")
        models_dir.mkdir(parents=True, exist_ok=True)

    team_name_to_id = {t.name: tid for tid, t in _build_teams(historical_dir, test_season).items()}
    full_test_frames: list[pd.DataFrame] = []

    fitted_models = {}
    for position in POSITIONS:
        pos_train = train_df[train_df["position"] == position]
        pos_test = test_df[test_df["position"] == position].reset_index(drop=True)
        if pos_train.empty or pos_test.empty:
            logger.warning("skipping %s: no train or test rows", position)
            continue

        pos_model = fit_position_model(pos_train)
        fitted_models[position] = pos_model
        ml_p_play, ml_pgp = pos_model.predict(pos_test)
        ml_preds = (ml_p_play * ml_pgp).to_numpy()
        ml_pgp_low, ml_pgp_high = pos_model.predict_quantiles(pos_test)
        ml_floor_ceiling = [
            mixture_floor_ceiling(p, lo, hi)
            for p, lo, hi in zip(ml_p_play, ml_pgp_low, ml_pgp_high, strict=True)
        ]
        ml_floor = np.array([f for f, _ in ml_floor_ceiling])
        ml_ceiling = np.array([c for _, c in ml_floor_ceiling])

        baseline_preds, floor, ceiling, confidence = _backtest_baseline(
            pos_test, historical_dir, test_season
        )

        y_test = pos_test[TARGET_COL].to_numpy()

        results["positions"][position] = {
            "ml_model": _metrics(y_test, ml_preds),
            "baseline_model": _metrics(y_test, baseline_preds),
            "fpl_xp": _metrics(y_test, pos_test[BASELINE_COL].to_numpy()),
            "naive_form": _metrics(y_test, pos_test[NAIVE_FORM_COL].to_numpy()),
            "n_train": len(pos_train),
        }

        overall["y"].append(y_test)
        overall["ml"].append(ml_preds)
        overall["baseline"].append(baseline_preds)
        overall["xp"].append(pos_test[BASELINE_COL].to_numpy())
        overall["naive"].append(pos_test[NAIVE_FORM_COL].to_numpy())
        baseline_calibration_inputs["actual"].append(y_test)
        baseline_calibration_inputs["floor"].append(floor)
        baseline_calibration_inputs["ceiling"].append(ceiling)
        baseline_calibration_inputs["confidence"].append(confidence)
        ml_calibration_inputs["actual"].append(y_test)
        ml_calibration_inputs["floor"].append(ml_floor)
        ml_calibration_inputs["ceiling"].append(ml_ceiling)
        ml_calibration_inputs["confidence"].append(ml_p_play.to_numpy())

        full_test_frames.append(
            pos_test.copy().assign(ml_pred=ml_preds, ml_floor=ml_floor, ml_ceiling=ml_ceiling, actual=y_test)
        )

    if save_models:
        persist_models(fitted_models, models_dir)  # type: ignore[arg-type]

    full_test_df = pd.concat(full_test_frames, ignore_index=True)

    correlation_input = pd.DataFrame(
        {
            "season": full_test_df["season"],
            "GW": full_test_df["GW"],
            "position": full_test_df["position"],
            "team_id": full_test_df["team"].map(team_name_to_id),
            "opponent_team_id": full_test_df["opponent_team"],
            "residual": full_test_df["actual"] - full_test_df["ml_pred"],
        }
    ).dropna(subset=["team_id", "opponent_team_id"])
    fixture_correlation = estimate_fixture_correlation(correlation_input)
    if save_models:
        persist_correlation(fixture_correlation, models_dir)  # type: ignore[arg-type]
    results["fixture_correlation"] = {
        "same_team_rho": fixture_correlation.same_team_rho,
        "opponent_rho": fixture_correlation.opponent_rho,
        "n_same_team_gameweeks": fixture_correlation.n_same_team_gameweeks,
        "n_opponent_gameweeks": fixture_correlation.n_opponent_gameweeks,
    }

    overall_y = np.concatenate(overall["y"])
    results["overall"] = {
        "ml_model": _metrics(overall_y, np.concatenate(overall["ml"])),
        "baseline_model": _metrics(overall_y, np.concatenate(overall["baseline"])),
        "fpl_xp": _metrics(overall_y, np.concatenate(overall["xp"])),
        "naive_form": _metrics(overall_y, np.concatenate(overall["naive"])),
    }
    results["baseline_calibration"] = _calibration_report(
        np.concatenate(baseline_calibration_inputs["actual"]),
        np.concatenate(baseline_calibration_inputs["floor"]),
        np.concatenate(baseline_calibration_inputs["ceiling"]),
        np.concatenate(baseline_calibration_inputs["confidence"]),
    )
    results["ml_calibration"] = _calibration_report(
        np.concatenate(ml_calibration_inputs["actual"]),
        np.concatenate(ml_calibration_inputs["floor"]),
        np.concatenate(ml_calibration_inputs["ceiling"]),
        np.concatenate(ml_calibration_inputs["confidence"]),
    )

    risk_validation = run_risk_validation(full_test_df, team_name_to_id)
    results["risk_validation"] = [asdict(v) for v in risk_validation]

    return results


def _pct_better(baseline_mae: float, model_mae: float) -> float:
    if baseline_mae == 0:
        return 0.0
    return round((baseline_mae - model_mae) / baseline_mae * 100, 1)


def render_report_markdown(results: dict) -> str:
    o = results["overall"]
    ml_mae = o["ml_model"]["mae"]
    baseline_mae = o["baseline_model"]["mae"]
    xp_mae = o["fpl_xp"]["mae"]

    lines = [
        "# Projection models: walk-forward backtest report",
        "",
        f"Generated: {results['generated_at']}",
        f"ML model trained on: {', '.join(results['seasons_trained_on'])}",
        f"Backtested on (held out entirely from ML training): {results['test_season']}",
        "",
        "MAE = mean absolute error in points per player-gameweek (lower is better).",
        "The hand-coded baseline has no training data to hold out -- it is scored",
        "on the same held-out season using only point-in-time reconstructed",
        "player state (see this module's docstring for exactly how).",
        "",
        "## Overall",
        "",
        "| Model | MAE | RMSE | n |",
        "|---|---|---|---|",
        f"| ML model (two-stage hurdle) | {ml_mae} | {o['ml_model']['rmse']} | {o['ml_model']['n']} |",
        f"| Hand-coded baseline | {baseline_mae} | {o['baseline_model']['rmse']} | {o['baseline_model']['n']} |",
        f"| FPL's own official xP | {xp_mae} | {o['fpl_xp']['rmse']} | {o['fpl_xp']['n']} |",
        f"| Naive rolling-5-GW form | {o['naive_form']['mae']} | {o['naive_form']['rmse']} | {o['naive_form']['n']} |",
        "",
    ]

    vs_baseline = _pct_better(baseline_mae, ml_mae)
    vs_xp = _pct_better(xp_mae, ml_mae)
    ml_wins_baseline = ml_mae < baseline_mae
    verdict = (
        f"The ML model beats the hand-coded baseline by {vs_baseline:+.1f}% MAE on this "
        f"backtest ({vs_xp:+.1f}% vs FPL's own xP)."
        if ml_wins_baseline
        else (
            f"The hand-coded baseline still beats the ML model by {-vs_baseline:+.1f}% MAE "
            f"on this backtest ({vs_xp:+.1f}% vs FPL's own xP for the ML model). Per "
            "docs/ROADMAP.md, a learned model should only replace the baseline as the "
            "*active* model once it demonstrably wins here -- see PROJECTION_MODEL below."
        )
    )
    lines += [verdict, ""]

    lines += [
        "## By position",
        "",
        "| Position | ML MAE | Baseline MAE | xP MAE | ML vs baseline | n (train / test) |",
        "|---|---|---|---|---|---|",
    ]
    for position, r in results["positions"].items():
        vs_base_pos = _pct_better(r["baseline_model"]["mae"], r["ml_model"]["mae"])
        lines.append(
            f"| {position} | {r['ml_model']['mae']} | {r['baseline_model']['mae']} | "
            f"{r['fpl_xp']['mae']} | {vs_base_pos:+.1f}% | {r['n_train']} / {r['ml_model']['n']} |"
        )
    lines.append("")

    ml_cal = results["ml_calibration"]
    lines += [
        "## ML model calibration: does the quantile-regression band capture reality?",
        "",
        (
            "Floor/ceiling come from per-player quantile regression (10th/90th "
            "percentile of E[points|plays], `ml/model.py`'s `predict_quantiles`) "
            "combined with the hurdle model's own P(plays) via `mixture_floor_ceiling` "
            "-- a real per-player band, not a single global offset applied to every "
            "player at a position."
        ),
        "",
        (
            f"Overall: actual points fell inside the ML model's [floor, ceiling] band "
            f"**{ml_cal['overall_band_coverage']:.0%}** of the time (target: ~80%, since "
            "this is nominally an 80% central interval)."
        ),
        "",
        "By P(plays) used for that gameweek (bucketed the same way as the",
        "baseline's confidence below, for direct comparison):",
        "",
        "| P(plays) range | n | Band coverage | Mean P(plays) |",
        "|---|---|---|---|",
    ]
    for bucket in ml_cal["by_confidence_bucket"]:
        lines.append(
            f"| {bucket['confidence_range']} | {bucket['n']} | {bucket['band_coverage']:.0%} | "
            f"{bucket['mean_confidence']:.2f} |"
        )
    lines.append("")

    cal = results["baseline_calibration"]
    lines += [
        "## Baseline calibration: does the floor/ceiling band capture reality?",
        "",
        (
            f"Overall: actual points fell inside the baseline's [floor, ceiling] band "
            f"**{cal['overall_band_coverage']:.0%}** of the time."
        ),
        "",
        "By the baseline's own stated confidence (docs/ROADMAP.md asks: are higher-",
        "confidence projections actually more reliable?):",
        "",
        "| Confidence range | n | Band coverage | Mean confidence |",
        "|---|---|---|---|",
    ]
    for bucket in cal["by_confidence_bucket"]:
        lines.append(
            f"| {bucket['confidence_range']} | {bucket['n']} | {bucket['band_coverage']:.0%} | "
            f"{bucket['mean_confidence']:.2f} |"
        )
    lines.append("")

    fc = results["fixture_correlation"]
    lines += [
        "## Fixture correlation: are same-team/same-match players actually correlated?",
        "",
        (
            "`risk/portfolio.py`'s mean-variance optimizer documents an independent-"
            "variance simplification (a squad's variance is just the sum of its "
            "players' variances). This measures whether that's actually a good "
            "approximation, using real backtest residuals -- see `risk/covariance.py` "
            "for the full derivation (Ledoit-Wolf-style shrinkage toward 0)."
        ),
        "",
        "| Relationship | Shrunk correlation (rho) | Gameweek samples |",
        "|---|---|---|",
        f"| Same team | {fc['same_team_rho']:+.3f} | {fc['n_same_team_gameweeks']} |",
        f"| Opposing teams, same fixture | {fc['opponent_rho']:+.3f} | {fc['n_opponent_gameweeks']} |",
        "",
        (
            f"Same-team correlation of {fc['same_team_rho']:+.3f} means the independence "
            "assumption understates real squad variance whenever a lineup concentrates "
            "picks from one team (which most managers do, e.g. 2-3 defenders from one "
            "in-form defence): `risk/covariance.py`'s `portfolio_variance` accounts for "
            "this; `risk/portfolio.py`'s per-player scoring still does not (see "
            "docs/ROADMAP.md's known limitations)."
            if fc["n_same_team_gameweeks"] > 0
            else "Not enough gameweek samples yet to estimate this reliably."
        ),
        "",
    ]

    lines += [
        "## Methodology",
        "",
        (
            "- Chronological holdout: the ML model never sees any "
            f"{results['test_season']} data during training."
        ),
        (
            "- Two-stage hurdle ML model per position (classifier for P(plays) x "
            "regressor for E[points|plays]) -- see `ml/model.py`."
        ),
        (
            "- The baseline is the actual, unmodified `projections.baseline_model."
            "project_player` function, run on point-in-time-reconstructed player "
            "state for every test-season row -- not a re-derived approximation."
        ),
        (
            "- Known simplification: per-gameweek historical availability/injury "
            "status and set-piece order are not present in the archived dataset, so "
            "every reconstructed player is treated as fully available -- this can "
            "only help the baseline's accuracy, never the ML model's, so it is a "
            "conservative bias if anything."
        ),
        (
            "- Known simplification, shared with the baseline model's own documented "
            "assumptions: team strength ratings are a season-level snapshot, not as "
            "they stood before each individual gameweek."
        ),
    ]
    return "\n".join(lines) + "\n"


def run_backtest_and_report(
    historical_dir: Path,
    reports_dir: Path,
    seasons: list[str] | None = None,
    test_season: str | None = None,
    save_models: bool = True,
    models_dir: Path | None = None,
) -> dict:
    results = run_backtest(
        historical_dir,
        seasons=seasons,
        test_season=test_season,
        save_models=save_models,
        models_dir=models_dir,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "model_backtest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (reports_dir / "model_backtest.md").write_text(render_report_markdown(results), encoding="utf-8")

    risk_validation = [StrategyValidation(**v) for v in results["risk_validation"]]
    (reports_dir / "risk_validation.md").write_text(
        render_risk_validation_markdown(risk_validation, results["test_season"]), encoding="utf-8"
    )
    return results
