"""Command-line interface.

Entry point: `fpl-automate <command>` (see pyproject.toml) when installed
normally, or `fpl-automate.exe <command>` (or just double-clicking it) when
run as the packaged Windows executable -- see `main()` at the bottom of this
file and `packaging/fpl-automate.spec` for how that build works.
"""
from __future__ import annotations

import os
import sys

import typer
from rich.console import Console
from rich.table import Table

from fpl_automate.config import ConfigError
from fpl_automate.logging_config import configure_logging
from fpl_automate.optimization.lineup import optimize_lineup
from fpl_automate.projections.ml import historical as ml_historical
from fpl_automate.projections.ml.backtest import run_backtest_and_report
from fpl_automate.risk.classification import RiskTier, classify_squad_risk, tier_counts
from fpl_automate.runtime import (
    APP_BASE_DIR,
    BUNDLE_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_HISTORICAL_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    ENV_FILE_PATH,
    IS_FROZEN,
    ensure_env_file_exists,
    get_app_settings,
)
from fpl_automate.squad.state import find_relevant_events, resolve_squad_state
from fpl_automate.win_console import fix_windows_console_encoding
from fpl_automate.workflow import (
    build_client,
    build_db,
    compute_projections,
    fetch_and_validate,
    run_weekly_plan,
)

fix_windows_console_encoding()

app = typer.Typer(add_completion=False, help="FPL quantitative decision-support CLI (recommendation-only).")
console = Console()

REPO_ROOT = APP_BASE_DIR
_BUNDLE_DIR = BUNDLE_DIR
_get_settings = get_app_settings


def _init() -> None:
    settings = _get_settings()
    configure_logging(settings.log_level)


_RISK_COLORS = {RiskTier.SAFE: "green", RiskTier.BALANCED: "yellow", RiskTier.RISKY: "red"}


def _risk_cell(tier: RiskTier) -> str:
    return f"[{_RISK_COLORS[tier]}]{tier.value}[/{_RISK_COLORS[tier]}]"


@app.command("health-check")
def health_check() -> None:
    """Checks config, database, and FPL API reachability."""
    console.print("[bold]Health check[/bold]")
    try:
        settings = _get_settings()
        console.print("[green]OK[/green] configuration loaded")
    except ConfigError as exc:
        console.print(f"[red]FAIL[/red] configuration: {exc}")
        raise typer.Exit(1)

    try:
        db = build_db(settings)
        db.log_fetch("health-check", "ok", "health-check ping")
        console.print("[green]OK[/green] database writable")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]FAIL[/red] database: {exc}")
        raise typer.Exit(1)

    try:
        client = build_client(settings, DEFAULT_CACHE_DIR)
        bootstrap = client.get_bootstrap_static()
        console.print(f"[green]OK[/green] FPL API reachable ({len(bootstrap.get('elements', []))} players)")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]FAIL[/red] FPL API: {exc}")
        raise typer.Exit(1)

    if settings.emergency_stop:
        console.print("[yellow]NOTE[/yellow] EMERGENCY_STOP is currently ON: all actions are blocked.")
    console.print(f"Auto-execution enabled: {settings.enable_auto_execution} (should be False)")


@app.command("fetch-data")
def fetch_data() -> None:
    """Fetches and persists current FPL data (players, teams, fixtures, gameweeks)."""
    _init()
    settings = _get_settings()
    client = build_client(settings, DEFAULT_CACHE_DIR)
    db = build_db(settings)
    data = fetch_and_validate(client, db)
    console.print(
        f"[green]Fetched[/green] {len(data.players)} players, {len(data.teams)} teams, "
        f"{len(data.fixtures)} fixtures, {len(data.gameweeks)} gameweeks."
    )


@app.command("validate-data")
def validate_data() -> None:
    """Fetches data and reports whether it passes all data-quality checks."""
    _init()
    settings = _get_settings()
    client = build_client(settings, DEFAULT_CACHE_DIR)
    db = build_db(settings)
    try:
        fetch_and_validate(client, db)
        console.print("[green]PASS[/green] all data-quality checks.")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]FAIL[/red] {exc}")
        raise typer.Exit(1)


@app.command("analyse-squad")
def analyse_squad() -> None:
    """Shows your current squad, bank, free transfers, chip status, and
    each player's risk profile (safe/balanced/risky -- see `risk/classification.py`)."""
    _init()
    settings = _get_settings()
    client = build_client(settings, DEFAULT_CACHE_DIR)
    db = build_db(settings)
    data = fetch_and_validate(client, db)
    _last_picks_event, next_deadline_event = find_relevant_events(data.gameweeks)
    squad_state = resolve_squad_state(client, settings.fpl_team_id, data.gameweeks)

    projections = compute_projections(
        data, from_event=next_deadline_event, horizons=(1,), projection_model=settings.projection_model
    )
    risk_profiles = classify_squad_risk(projections[1])

    table = Table(title=f"Squad as of GW{squad_state.as_of_event}")
    table.add_column("Player")
    table.add_column("Pos")
    table.add_column("Price")
    table.add_column("Status")
    table.add_column("Risk")
    for pick in squad_state.picks:
        p = data.players_by_id[pick.element_id]
        tag = " (C)" if pick.is_captain else (" (VC)" if pick.is_vice_captain else "")
        risk = risk_profiles.get(p.id)
        risk_cell = _risk_cell(risk.tier) if risk else "-"
        table.add_row(p.web_name + tag, p.position.short, f"£{p.price:.1f}m", p.availability.status, risk_cell)
    console.print(table)
    counts = tier_counts(risk_profiles)
    console.print(
        f"Squad risk profile: {counts[RiskTier.SAFE]} safe, {counts[RiskTier.BALANCED]} balanced, "
        f"{counts[RiskTier.RISKY]} risky"
    )
    console.print(f"Bank: £{squad_state.bank}m | Squad value: £{squad_state.squad_value}m")
    console.print(f"Free transfers available: {squad_state.free_transfers_available}")
    console.print(
        f"Chips: wildcard_available={squad_state.wildcard_available}, "
        f"free_hit_available={squad_state.free_hit_available}, "
        f"bench_boost_available={squad_state.bench_boost_available}, "
        f"triple_captain_available={squad_state.triple_captain_available}"
    )


@app.command("recommend-transfers")
def recommend_transfers_cmd(
    strategy: str = typer.Option(
        "balanced", help="conservative | balanced | aggressive | risk_adjusted"
    ),
    risk_aversion: float = typer.Option(
        None, help="Only used with --strategy risk_adjusted; overrides RISK_AVERSION from .env."
    ),
) -> None:
    """Shows ranked transfer scenarios (including 'roll') for the next deadline."""
    _init()
    settings = _get_settings()
    effective_risk_aversion = settings.risk_aversion if risk_aversion is None else risk_aversion
    client = build_client(settings, DEFAULT_CACHE_DIR)
    db = build_db(settings)
    data = fetch_and_validate(client, db)
    _last_picks_event, next_deadline_event = find_relevant_events(data.gameweeks)

    squad_state = resolve_squad_state(client, settings.fpl_team_id, data.gameweeks)
    transfers_history = client.get_entry_transfers(settings.fpl_team_id)
    owned_squad = [data.players_by_id[p.element_id] for p in squad_state.picks]
    projections = compute_projections(
        data, from_event=next_deadline_event, projection_model=settings.projection_model
    )

    from fpl_automate.transfers.engine import recommend_transfers as recommend_transfers_fn

    scenarios = recommend_transfers_fn(
        squad=owned_squad,
        squad_state=squad_state,
        transfers_history=transfers_history,
        all_players=data.players,
        projections_1gw=projections[1],
        projections_3gw=projections[3],
        projections_5gw=projections[5],
        strategy=strategy,  # type: ignore[arg-type]
        risk_aversion=effective_risk_aversion,
        max_transfer_risk=settings.max_transfer_risk,
    )

    table = Table(title=f"Transfer scenarios for GW{next_deadline_event} ({strategy})")
    table.add_column("Option")
    table.add_column("Hit")
    table.add_column("5GW gain")
    table.add_column("Net (after hit)")
    table.add_column("Recommended?")
    for s in scenarios:
        table.add_row(
            s.label, str(s.points_hit), f"{s.ep_gain_5gw:+.2f}", f"{s.net_ep_gain_5gw:+.2f}",
            "YES" if s.recommended else "",
        )
    console.print(table)


@app.command("optimise-lineup")
def optimise_lineup_cmd(
    strategy: str = typer.Option(
        "balanced", help="conservative | balanced | aggressive | risk_adjusted"
    ),
    risk_aversion: float = typer.Option(
        None, help="Only used with --strategy risk_adjusted; overrides RISK_AVERSION from .env."
    ),
) -> None:
    """Optimises starting XI / bench / captaincy for your CURRENT squad (no transfers)."""
    _init()
    settings = _get_settings()
    effective_risk_aversion = settings.risk_aversion if risk_aversion is None else risk_aversion
    client = build_client(settings, DEFAULT_CACHE_DIR)
    db = build_db(settings)
    data = fetch_and_validate(client, db)
    _last_picks_event, next_deadline_event = find_relevant_events(data.gameweeks)

    squad_state = resolve_squad_state(client, settings.fpl_team_id, data.gameweeks)
    owned_squad = [data.players_by_id[p.element_id] for p in squad_state.picks]
    projections = compute_projections(
        data, from_event=next_deadline_event, horizons=(1,), projection_model=settings.projection_model
    )

    result = optimize_lineup(owned_squad, projections[1], strategy, effective_risk_aversion)  # type: ignore[arg-type]
    console.print(f"Formation: {result.formation} | Strategy: {strategy}")
    console.print(f"Captain: {data.players_by_id[result.captain_id].web_name}")
    console.print(f"Vice-captain: {data.players_by_id[result.vice_captain_id].web_name}")
    console.print("Starting XI: " + ", ".join(data.players_by_id[pid].web_name for pid in result.starting_xi))
    console.print("Bench order: " + ", ".join(data.players_by_id[pid].web_name for pid in result.bench_order))
    console.print(
        f"Expected {result.total_expected_points} | Floor {result.total_floor_points} | "
        f"Ceiling {result.total_ceiling_points}"
    )
    if result.total_risk_adjusted_score is not None:
        console.print(
            f"Risk-adjusted score: {result.total_risk_adjusted_score} (risk_aversion={result.risk_aversion})"
        )


@app.command("run-weekly-plan")
def run_weekly_plan_cmd(
    strategy: str = typer.Option(
        "balanced", help="conservative | balanced | aggressive | risk_adjusted"
    ),
) -> None:
    """Runs the full pipeline and writes/sends the weekly decision report.
    Uses RISK_AVERSION from .env when --strategy risk_adjusted (no
    per-run CLI override, same as MAX_TRANSFER_RISK/PROJECTION_MODEL)."""
    _init()
    settings = _get_settings()
    report, md_path, json_path = run_weekly_plan(
        settings, DEFAULT_REPORTS_DIR, DEFAULT_CACHE_DIR, chosen_strategy=strategy  # type: ignore[arg-type]
    )
    console.print(f"[bold]{report.approve_state}[/bold]")
    console.print(f"Report written to {md_path} and {json_path}")


@app.command("show-history")
def show_history(limit: int = 20) -> None:
    """Shows past decisions and (if recorded) their outcomes."""
    _init()
    settings = _get_settings()
    db = build_db(settings)
    rows = db.history(limit)
    if not rows:
        console.print("No decision history yet -- run `run-weekly-plan` first.")
        return
    table = Table(title="Decision history")
    table.add_column("GW")
    table.add_column("Strategy")
    table.add_column("Approved?")
    table.add_column("Actual points")
    for row in rows:
        table.add_row(
            str(row["event"]), row["strategy"],
            "" if row["approved"] is None else str(bool(row["approved"])),
            "" if row["actual_points"] is None else str(row["actual_points"]),
        )
    console.print(table)


@app.command("show-model-performance")
def show_model_performance() -> None:
    """Shows the projection model's actual accuracy in production -- MAE
    and bias between logged projections and the real points that landed,
    for every gameweek reconciled so far (see `reconcile_outcomes` in
    workflow.py, which runs automatically at the start of
    `run-weekly-plan`). The live complement to the offline walk-forward
    backtest in reports/model_backtest.md."""
    _init()
    settings = _get_settings()
    db = build_db(settings)
    rows = db.projection_performance()
    if not rows:
        console.print(
            "No resolved projections yet -- run `run-weekly-plan` a few times across "
            "gameweeks so there's something to reconcile against real results."
        )
        return

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model_name"], []).append(row)

    table = Table(title="Live projection accuracy (per model)")
    table.add_column("Model")
    table.add_column("n")
    table.add_column("MAE")
    table.add_column("Bias (mean signed error)")
    for model_name, model_rows in by_model.items():
        errors = [r["expected_points"] - r["actual_points"] for r in model_rows]
        mae = sum(abs(e) for e in errors) / len(errors)
        bias = sum(errors) / len(errors)
        table.add_row(model_name, str(len(model_rows)), f"{mae:.2f}", f"{bias:+.2f}")
    console.print(table)

    console.print(f"\nMost recent {min(10, len(rows))} resolved projections:")
    recent_table = Table()
    recent_table.add_column("GW")
    recent_table.add_column("Model")
    recent_table.add_column("Expected")
    recent_table.add_column("Actual")
    recent_table.add_column("Error")
    for row in rows[:10]:
        error = row["expected_points"] - row["actual_points"]
        recent_table.add_row(
            str(row["event"]), row["model_name"], f"{row['expected_points']:.2f}",
            str(row["actual_points"]), f"{error:+.2f}",
        )
    console.print(recent_table)


def _not_yet_implemented(feature: str, phase: str) -> None:
    console.print(
        f"[yellow]'{feature}' is not implemented yet.[/yellow] It is planned for "
        f"{phase} -- see docs/ROADMAP.md."
    )


@app.command("fetch-historical-data")
def fetch_historical_data_cmd(
    seasons: str = typer.Option(
        ",".join(ml_historical.DEFAULT_SEASONS), help="Comma-separated seasons, e.g. 2021-22,2022-23"
    ),
) -> None:
    """Downloads/refreshes cached historical per-season data used to train the ML model."""
    # Deliberately not _init()/_get_settings(): fetching historical data has
    # nothing to do with any specific user's team or FPL_TEAM_ID, and
    # requiring it here would force e.g. train-model.yml to configure an
    # identity it doesn't need just to train a model.
    configure_logging()
    season_list = [s.strip() for s in seasons.split(",") if s.strip()]
    results = ml_historical.fetch_all_seasons(DEFAULT_HISTORICAL_DIR, seasons=season_list)
    for season, files in results.items():
        missing = [f for f, ok in files.items() if not ok]
        status = "[green]OK[/green]" if not missing else f"[yellow]missing: {', '.join(missing)}[/yellow]"
        console.print(f"{season}: {status}")


@app.command("backtest")
def backtest_cmd(
    seasons: str = typer.Option(
        ",".join(ml_historical.DEFAULT_SEASONS), help="Comma-separated seasons to train/backtest on"
    ),
    test_season: str = typer.Option(ml_historical.TEST_SEASON, help="Held-out season to backtest against"),
) -> None:
    """Walk-forward backtest (docs/ROADMAP.md Phase 2): trains the ML model
    (classifier + mean/quantile regressors) on every season except
    `test_season`, evaluates it against the hand-coded baseline and FPL's
    own xP on `test_season` (held out entirely from training), saves the
    trained models for live use, estimates same-fixture player correlation
    (`risk/covariance.py`), and validates the risk_adjusted strategy
    against realized (not projected) outcomes (`risk/validation.py`)."""
    # Same reasoning as fetch-historical-data-cmd above: no FPL_TEAM_ID needed.
    configure_logging()
    season_list = [s.strip() for s in seasons.split(",") if s.strip()]
    console.print("Fetching historical data (cached files are reused if already present)...")
    ml_historical.fetch_all_seasons(DEFAULT_HISTORICAL_DIR, seasons=season_list)
    console.print("Training and backtesting...")
    results = run_backtest_and_report(
        DEFAULT_HISTORICAL_DIR,
        DEFAULT_REPORTS_DIR,
        seasons=season_list,
        test_season=test_season,
        save_models=True,
        models_dir=DEFAULT_MODELS_DIR,
    )
    o = results["overall"]
    console.print(
        f"ML model MAE: {o['ml_model']['mae']} | Baseline MAE: {o['baseline_model']['mae']} | "
        f"FPL xP MAE: {o['fpl_xp']['mae']}"
    )
    console.print(f"Full report: {DEFAULT_REPORTS_DIR / 'model_backtest.md'}")
    console.print(f"Risk validation report: {DEFAULT_REPORTS_DIR / 'risk_validation.md'}")


@app.command("bursary-search")
def bursary_search() -> None:
    """(Planned) Search/import structured bursary & scholarship opportunities."""
    _not_yet_implemented("bursary-search", "Phase 4 (quant applications module)")


@app.command("bursary-score")
def bursary_score() -> None:
    """(Planned) Score bursary/scholarship opportunities by fit, value, and deadline."""
    _not_yet_implemented("bursary-score", "Phase 4 (quant applications module)")


@app.command("generate-checklist")
def generate_checklist() -> None:
    """(Planned) Generate a personalised document checklist for an opportunity."""
    _not_yet_implemented("generate-checklist", "Phase 4 (quant applications module)")


def _pause_if_interactive() -> None:
    """Keeps a double-clicked console window open until the user reads it. Skipped
    when stdin isn't a real terminal (CI, a script piping input, etc.) -- otherwise
    this would hang or crash a non-interactive run waiting for input that never
    comes."""
    if sys.stdin.isatty():
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass


def _ensure_env_file_exists() -> bool:
    """Returns True if the caller must stop here (a fresh .env was just
    created, or none could be) rather than proceeding to a command that
    needs settings. Prefer the GUI (`fpl-automate-gui`) if you'd rather fill
    in settings through a form than a text file -- see README."""
    created, template_found = ensure_env_file_exists()
    if not created:
        return False
    if template_found:
        console.print(f"[yellow]No .env found -- created one at:[/yellow] {ENV_FILE_PATH}")
        console.print(
            "Open that file in a text editor, confirm FPL_TEAM_ID (and set up email "
            "notifications if you want them), then run this program again. Prefer a "
            "form instead of a text file? Use fpl-automate-gui."
        )
    else:
        console.print(
            f"[red]No .env found at {ENV_FILE_PATH} and no template is bundled.[/red] "
            "Create one there with at least a line: FPL_TEAM_ID=<your team id>"
        )
    return True


def main() -> None:
    """Entry point for both `fpl-automate` (pyproject.toml console_scripts) and
    the packaged .exe (packaging/fpl-automate.spec). Handles the two things a
    plain `app()` call doesn't: making the exe's folder self-contained
    (first-run .env creation, paths relative to the exe not the caller's cwd),
    and keeping a double-clicked console window open long enough to read.
    """
    if IS_FROZEN:
        os.chdir(APP_BASE_DIR)
        if _ensure_env_file_exists():
            _pause_if_interactive()
            sys.exit(1)

    exit_code = 0
    try:
        app()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception as exc:  # noqa: BLE001 - last-resort handler for a double-clicked exe
        console.print(f"[red]Unexpected error:[/red] {exc}")
        console.print_exception()
        exit_code = 1

    if IS_FROZEN:
        _pause_if_interactive()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
