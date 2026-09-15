"""Orchestrates the full weekly pipeline: fetch -> validate -> features ->
projections -> squad analysis -> transfers -> lineup -> report -> notify.

Kept as plain functions (not a class) so each stage is independently
callable from the CLI (e.g. `fetch-data` alone, or `analyse-squad` alone)
without paying for the full pipeline every time.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

from fpl_automate.config import Settings
from fpl_automate.data.cache import FileCache
from fpl_automate.data.fpl_client import FplClient
from fpl_automate.data.parsers import parse_fixture, parse_gameweek, parse_player, parse_team
from fpl_automate.features.engineering import compute_player_features
from fpl_automate.notifications.base import NullNotifier
from fpl_automate.notifications.email_notifier import EmailNotifier
from fpl_automate.optimization.lineup import Strategy, optimize_lineup
from fpl_automate.projections.baseline_model import ModelInputs, project_player
from fpl_automate.projections.ml import historical as ml_historical
from fpl_automate.projections.ml.live import compute_ml_projections
from fpl_automate.projections.ml.model import load_models
from fpl_automate.reporting.report import (
    WeeklyReport,
    build_weekly_report,
    render_markdown,
    save_report,
)
from fpl_automate.runtime import DEFAULT_HISTORICAL_DIR, DEFAULT_MODELS_DIR
from fpl_automate.squad.state import find_relevant_events, resolve_squad_state
from fpl_automate.storage.db import Database
from fpl_automate.storage.models import Fixture, Gameweek, Player, PlayerProjection, Team
from fpl_automate.transfers.engine import recommend_transfers
from fpl_automate.validation.checks import validate_bootstrap_static, validate_freshness

logger = logging.getLogger(__name__)

STRATEGIES: tuple[Strategy, ...] = ("conservative", "balanced", "aggressive")
DEFAULT_HORIZONS = (1, 3, 5)


def build_client(settings: Settings, cache_dir: Path) -> FplClient:
    return FplClient(
        base_url=settings.fpl_api_base_url,
        timeout_seconds=settings.fpl_http_timeout_seconds,
        min_request_interval_seconds=settings.fpl_min_request_interval_seconds,
        max_retries=settings.fpl_max_retries,
        cache=FileCache(cache_dir),
    )


def build_db(settings: Settings) -> Database:
    path = settings.database_url.removeprefix("sqlite:///")
    return Database(Path(path))


class FetchedData:
    def __init__(
        self,
        teams: list[Team],
        players: list[Player],
        gameweeks: list[Gameweek],
        fixtures: list[Fixture],
    ) -> None:
        self.teams = teams
        self.players = players
        self.gameweeks = gameweeks
        self.fixtures = fixtures
        self.teams_by_id = {t.id: t for t in teams}
        self.players_by_id = {p.id: p for p in players}


def fetch_and_validate(client: FplClient, db: Database) -> FetchedData:
    try:
        bootstrap = client.get_bootstrap_static()
        validate_bootstrap_static(bootstrap)
        fixtures_raw = client.get_fixtures()
    except Exception:
        db.log_fetch("bootstrap-static+fixtures", "error")
        raise
    db.log_fetch("bootstrap-static+fixtures", "ok")

    teams = [parse_team(t) for t in bootstrap["teams"]]
    players = [parse_player(p) for p in bootstrap["elements"]]
    gameweeks = [parse_gameweek(e) for e in bootstrap["events"]]
    fixtures = [parse_fixture(f) for f in fixtures_raw]

    db.upsert_teams(bootstrap["teams"])
    db.upsert_players(bootstrap["elements"])
    db.upsert_gameweeks(bootstrap["events"])
    db.upsert_fixtures(fixtures_raw)

    return FetchedData(teams=teams, players=players, gameweeks=gameweeks, fixtures=fixtures)


def _baseline_projections(
    data: FetchedData, from_event: int, horizons: tuple[int, ...]
) -> dict[int, dict[int, PlayerProjection]]:
    result: dict[int, dict[int, PlayerProjection]] = {}
    for horizon in horizons:
        horizon_projections: dict[int, PlayerProjection] = {}
        for player in data.players:
            team = data.teams_by_id.get(player.team_id)
            if team is None:
                continue
            features = compute_player_features(
                player, team, data.fixtures, from_event=from_event, fixture_window_gameweeks=horizon
            )
            horizon_projections[player.id] = project_player(
                ModelInputs(player=player, features=features, horizon_gameweeks=horizon)
            )
        result[horizon] = horizon_projections
    return result


def compute_projections(
    data: FetchedData,
    from_event: int,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    projection_model: str = "ml",
    historical_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict[int, dict[int, PlayerProjection]]:
    """Returns {horizon_gameweeks: {player_id: PlayerProjection}}.

    `projection_model="ml"` (the default, see Settings.projection_model)
    uses the trained ML model wherever it covers a player, falling back
    to the hand-coded baseline per player otherwise -- never a silent
    substitution, since every PlayerProjection.rationale names which
    model actually produced it. `projection_model="baseline"` skips the
    ML model entirely, e.g. to audit the two side by side or reproduce
    pre-Phase-2 behaviour.
    """
    baseline = _baseline_projections(data, from_event, horizons)
    if projection_model != "ml":
        return baseline

    ml_result = compute_ml_projections(
        players=data.players,
        teams=data.teams,
        fixtures=data.fixtures,
        from_event=from_event,
        horizons=horizons,
        historical_dir=historical_dir or DEFAULT_HISTORICAL_DIR,
        models_dir=models_dir or DEFAULT_MODELS_DIR,
    )
    if ml_result is None:
        logger.info("no trained ML model found -- using the baseline model for every player")
        return baseline

    for horizon in horizons:
        baseline[horizon].update(ml_result.get(horizon, {}))
    return baseline


def reconcile_outcomes(
    client: FplClient, db: Database, team_id: int, gameweeks: list[Gameweek], historical_dir: Path
) -> dict[str, int]:
    """Fills in actual outcomes for past decisions and projections whose
    gameweek has since finished -- the live complement to
    `ml/backtest.py`'s offline walk-forward backtest, and what actually
    makes `show-model-performance` / `show-history`'s "actual points"
    column non-empty. Never raises on a single row it can't resolve yet
    (e.g. vaastav's current-season archive hasn't caught up to that
    gameweek): it just leaves that row unresolved for the next run to
    retry, rather than failing the whole reconciliation pass.

    Returns {"decisions": n_resolved, "projections": n_resolved}.
    """
    finished_events = [g.id for g in gameweeks if g.finished]
    if not finished_events:
        return {"decisions": 0, "projections": 0}
    last_finished = max(finished_events)

    resolved = {"decisions": 0, "projections": 0}

    unresolved_decisions = db.unresolved_decisions(last_finished)
    if unresolved_decisions:
        try:
            history = client.get_entry_history(team_id)
            points_by_event = {g["event"]: g["points"] for g in history.get("current", [])}
        except Exception as exc:  # noqa: BLE001 -- reconciliation is best-effort, never fatal
            logger.warning("could not fetch entry history to reconcile decisions: %s", exc)
            points_by_event = {}
        for row in unresolved_decisions:
            points = points_by_event.get(row["event"])
            if points is None:
                continue
            db.record_decision_actual_points(row["id"], points)
            resolved["decisions"] += 1

    unresolved_projections = db.unresolved_projections(last_finished)
    if unresolved_projections:
        current_history = ml_historical.load_merged_gw(historical_dir, [ml_historical.CURRENT_SEASON])
        actual_by_player_event: dict[tuple[int, int], int] = {}
        if not current_history.empty:
            for row in current_history.itertuples():
                actual_by_player_event[(int(row.element), int(row.GW))] = int(row.total_points)
        for row in unresolved_projections:
            actual = actual_by_player_event.get((row["player_id"], row["event"]))
            if actual is None:
                continue
            db.record_projection_actual_points(row["id"], actual)
            resolved["projections"] += 1

    return resolved


def run_weekly_plan(
    settings: Settings, reports_dir: Path, cache_dir: Path, chosen_strategy: Strategy = "balanced"
) -> tuple[WeeklyReport, Path, Path]:
    settings.require_not_stopped()

    client = build_client(settings, cache_dir)
    db = build_db(settings)

    data = fetch_and_validate(client, db)
    _last_picks_event, next_deadline_event = find_relevant_events(data.gameweeks)

    next_gw = next(g for g in data.gameweeks if g.id == next_deadline_event)
    validate_freshness(next_gw.deadline_time, max_age_before_deadline=timedelta(days=10))

    resolved = reconcile_outcomes(
        client, db, settings.fpl_team_id, data.gameweeks, DEFAULT_HISTORICAL_DIR
    )
    if resolved["decisions"] or resolved["projections"]:
        logger.info(
            "reconciled %d past decision(s) and %d past projection(s) with actual results",
            resolved["decisions"],
            resolved["projections"],
        )

    squad_state = resolve_squad_state(client, settings.fpl_team_id, data.gameweeks)
    db.save_squad_snapshot(
        settings.fpl_team_id, squad_state.as_of_event, squad_state.model_dump_json()
    )

    transfers_history = client.get_entry_transfers(settings.fpl_team_id)
    owned_squad = [data.players_by_id[p.element_id] for p in squad_state.picks]

    projections = compute_projections(
        data, from_event=next_deadline_event, projection_model=settings.projection_model
    )
    # Report which model actually produced these projections, not just which
    # one was requested -- `compute_projections` silently falls back to the
    # baseline for everyone if no trained model exists yet, and the report
    # should say so rather than overclaim "ml" (see docs/ROADMAP.md: never a
    # *silent* substitution).
    actual_projection_model = (
        "ml" if settings.projection_model == "ml" and load_models(DEFAULT_MODELS_DIR) else "baseline"
    )

    # Log the owned squad's next-gameweek projections now, so a future run's
    # reconcile_outcomes() can compare them to what actually happened once
    # this gameweek finishes -- the live complement to ml/backtest.py's
    # offline backtest. Only the owned squad (not the full player pool) to
    # keep this lightweight; see storage/db.py's projection_log docstring.
    db.log_projections(
        event=next_deadline_event,
        model_name=actual_projection_model,
        horizon_gameweeks=1,
        rows=[
            {
                "player_id": p.id,
                "expected_points": proj.expected_points,
                "floor_points": proj.floor_points,
                "ceiling_points": proj.ceiling_points,
                "confidence": proj.confidence,
            }
            for p in owned_squad
            if (proj := projections[1].get(p.id)) is not None
        ],
    )

    scenarios = recommend_transfers(
        squad=owned_squad,
        squad_state=squad_state,
        transfers_history=transfers_history,
        all_players=data.players,
        projections_1gw=projections[1],
        projections_3gw=projections[3],
        projections_5gw=projections[5],
        strategy=chosen_strategy,
        max_transfer_risk=settings.max_transfer_risk,
    )
    recommended = next((s for s in scenarios if s.recommended), scenarios[0])

    resulting_squad = list(owned_squad)
    for move in recommended.moves:
        resulting_squad = [p for p in resulting_squad if p.id != move.sell_player_id]
        resulting_squad.append(data.players_by_id[move.buy_player_id])

    lineups_by_strategy = {
        strat: optimize_lineup(resulting_squad, projections[1], strat) for strat in STRATEGIES
    }

    report = build_weekly_report(
        team_id=settings.fpl_team_id,
        gameweek=next_deadline_event,
        squad_state=squad_state,
        all_players_by_id=data.players_by_id,
        projections_1gw=projections[1],
        projections_5gw=projections[5],
        transfer_scenarios=scenarios,
        lineups_by_strategy=lineups_by_strategy,
        chosen_strategy=chosen_strategy,
        projection_model=actual_projection_model,
    )

    md_path, json_path = save_report(report, reports_dir)
    db.save_decision(next_deadline_event, chosen_strategy, report.recommended_scenario.model_dump_json())

    notifier = EmailNotifier(settings) if settings.email_notifications_enabled else NullNotifier()
    try:
        notifier.send(
            subject=f"FPL GW{next_deadline_event} Plan: {report.approve_state}",
            body_text=render_markdown(report),
        )
    except Exception as exc:  # noqa: BLE001 - never let a notification failure hide the report
        logger.error("Notification failed (report was still generated and saved): %s", exc)

    return report, md_path, json_path
