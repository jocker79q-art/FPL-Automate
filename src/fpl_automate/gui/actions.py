"""Pipeline actions formatted as plain text, for the GUI's output panel.

Deliberately tkinter-free (no `import tkinter` anywhere in this file) so it
stays unit-testable in any environment, including one without a Tk display
available. `gui/app.py` is the only module that touches actual widgets;
this module only reuses the same core layers the CLI uses
(`workflow.py`, `optimization/`, `transfers/`) and formats their results as
strings.
"""
from __future__ import annotations

from fpl_automate.config import Settings
from fpl_automate.notifications.email_notifier import EmailNotifier
from fpl_automate.optimization.lineup import Strategy, optimize_lineup
from fpl_automate.reporting.report import render_markdown
from fpl_automate.runtime import DEFAULT_CACHE_DIR, DEFAULT_REPORTS_DIR
from fpl_automate.squad.state import find_relevant_events, resolve_squad_state
from fpl_automate.transfers.engine import recommend_transfers
from fpl_automate.workflow import build_client, build_db, compute_projections, fetch_and_validate
from fpl_automate.workflow import run_weekly_plan as run_weekly_plan_pipeline


def health_check(settings: Settings) -> str:
    lines = ["Health check", ""]
    db = build_db(settings)
    db.log_fetch("health-check", "ok", "GUI health-check")
    lines.append("OK  database writable")

    client = build_client(settings, DEFAULT_CACHE_DIR)
    bootstrap = client.get_bootstrap_static()
    lines.append(f"OK  FPL API reachable ({len(bootstrap.get('elements', []))} players)")

    if settings.emergency_stop:
        lines.append("NOTE  EMERGENCY_STOP is currently ON: all actions are blocked.")
    lines.append(f"Auto-execution enabled: {settings.enable_auto_execution} (should be False)")
    return "\n".join(lines)


def analyse_squad(settings: Settings) -> str:
    client = build_client(settings, DEFAULT_CACHE_DIR)
    db = build_db(settings)
    data = fetch_and_validate(client, db)
    squad_state = resolve_squad_state(client, settings.fpl_team_id, data.gameweeks)

    lines = [f"Squad as of GW{squad_state.as_of_event}", ""]
    for pick in squad_state.picks:
        p = data.players_by_id[pick.element_id]
        tag = " (C)" if pick.is_captain else (" (VC)" if pick.is_vice_captain else "")
        lines.append(f"  {p.web_name}{tag:<5} {p.position.short:<4} £{p.price:>5.1f}m  status={p.availability.status}")

    lines.append("")
    lines.append(f"Bank: £{squad_state.bank}m | Squad value: £{squad_state.squad_value}m")
    lines.append(f"Free transfers available: {squad_state.free_transfers_available}")
    lines.append(
        f"Chips: wildcard_available={squad_state.wildcard_available}, "
        f"free_hit_available={squad_state.free_hit_available}, "
        f"bench_boost_available={squad_state.bench_boost_available}, "
        f"triple_captain_available={squad_state.triple_captain_available}"
    )
    return "\n".join(lines)


def recommend_transfers_action(settings: Settings, strategy: Strategy) -> str:
    client = build_client(settings, DEFAULT_CACHE_DIR)
    db = build_db(settings)
    data = fetch_and_validate(client, db)
    _last_picks_event, next_deadline_event = find_relevant_events(data.gameweeks)

    squad_state = resolve_squad_state(client, settings.fpl_team_id, data.gameweeks)
    transfers_history = client.get_entry_transfers(settings.fpl_team_id)
    owned_squad = [data.players_by_id[p.element_id] for p in squad_state.picks]
    projections = compute_projections(data, from_event=next_deadline_event)

    scenarios = recommend_transfers(
        squad=owned_squad,
        squad_state=squad_state,
        transfers_history=transfers_history,
        all_players=data.players,
        projections_1gw=projections[1],
        projections_3gw=projections[3],
        projections_5gw=projections[5],
        strategy=strategy,
        max_transfer_risk=settings.max_transfer_risk,
    )

    lines = [f"Transfer scenarios for GW{next_deadline_event} ({strategy})", ""]
    for s in scenarios:
        flag = "  <- RECOMMENDED" if s.recommended else ""
        lines.append(
            f"  {s.label:<45} hit=-{s.points_hit:<2} 5GW gain={s.ep_gain_5gw:+6.2f} "
            f"net={s.net_ep_gain_5gw:+6.2f}{flag}"
        )
    return "\n".join(lines)


def optimise_lineup_action(settings: Settings, strategy: Strategy) -> str:
    client = build_client(settings, DEFAULT_CACHE_DIR)
    db = build_db(settings)
    data = fetch_and_validate(client, db)
    _last_picks_event, next_deadline_event = find_relevant_events(data.gameweeks)

    squad_state = resolve_squad_state(client, settings.fpl_team_id, data.gameweeks)
    owned_squad = [data.players_by_id[p.element_id] for p in squad_state.picks]
    projections = compute_projections(data, from_event=next_deadline_event, horizons=(1,))

    result = optimize_lineup(owned_squad, projections[1], strategy)
    lines = [
        f"Formation: {result.formation} | Strategy: {strategy}",
        f"Captain: {data.players_by_id[result.captain_id].web_name}",
        f"Vice-captain: {data.players_by_id[result.vice_captain_id].web_name}",
        "Starting XI: " + ", ".join(data.players_by_id[pid].web_name for pid in result.starting_xi),
        "Bench order: " + ", ".join(data.players_by_id[pid].web_name for pid in result.bench_order),
        (
            f"Expected {result.total_expected_points} | Floor {result.total_floor_points} | "
            f"Ceiling {result.total_ceiling_points}"
        ),
    ]
    return "\n".join(lines)


def send_test_email(settings: Settings) -> str:
    """Used by the Settings tab's "Send Test Email" button, so you can verify SMTP
    details work before relying on them for a real weekly alert."""
    if not settings.email_notifications_enabled:
        raise ValueError("Enable email notifications and save settings first.")
    notifier = EmailNotifier(settings)
    notifier.send(
        subject="FPL Automate: test email",
        body_text=(
            "This is a test email from FPL Automate's Settings tab. If you're "
            "reading this, your SMTP settings work."
        ),
    )
    return f"Test email sent to {settings.email_to}."


def run_weekly_plan_action(settings: Settings, strategy: Strategy) -> str:
    report, md_path, _json_path = run_weekly_plan_pipeline(
        settings, DEFAULT_REPORTS_DIR, DEFAULT_CACHE_DIR, chosen_strategy=strategy
    )
    return f"Report saved to: {md_path}\n\n{render_markdown(report)}"
