"""Weekly decision report: the single human-readable artifact this system produces.

Nothing in this module ever calls the FPL API or submits anything -- it is
pure presentation over already-computed squad state, projections, lineup
options, and transfer scenarios. The "approve" / "do not approve" state it
prints is advisory text for you to read, never a trigger for any action.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fpl_automate.optimization.lineup import LineupResult, Strategy
from fpl_automate.storage.models import Player, PlayerProjection, SquadState
from fpl_automate.transfers.engine import TransferScenario


@dataclass
class WeeklyReport:
    team_id: int
    gameweek: int
    generated_at: str
    squad_state: SquadState
    transfer_scenarios: list[TransferScenario]
    recommended_scenario: TransferScenario
    lineups_by_strategy: dict[Strategy, LineupResult]
    chosen_strategy: Strategy
    player_names: dict[int, str]
    key_risks: list[str] = field(default_factory=list)
    confidence: float = 0.0
    what_would_change_this: list[str] = field(default_factory=list)
    approve_state: str = "DO NOT APPROVE"
    projection_model: str = "baseline"


def _name(names: dict[int, str], player_id: int) -> str:
    return names.get(player_id, f"#{player_id}")


def build_weekly_report(
    team_id: int,
    gameweek: int,
    squad_state: SquadState,
    all_players_by_id: dict[int, Player],
    projections_1gw: dict[int, PlayerProjection],
    projections_5gw: dict[int, PlayerProjection],
    transfer_scenarios: list[TransferScenario],
    lineups_by_strategy: dict[Strategy, LineupResult],
    chosen_strategy: Strategy = "balanced",
    projection_model: str = "baseline",
) -> WeeklyReport:
    player_names = {pid: p.web_name for pid, p in all_players_by_id.items()}
    recommended = next((s for s in transfer_scenarios if s.recommended), transfer_scenarios[0])

    key_risks: list[str] = []
    for s in transfer_scenarios[:1]:
        key_risks.extend(s.risk_notes)
    chosen_lineup = lineups_by_strategy[chosen_strategy]
    for pid in chosen_lineup.starting_xi:
        proj = projections_1gw.get(pid)
        if proj and proj.risk_flags:
            key_risks.append(f"{_name(player_names, pid)}: {', '.join(proj.risk_flags)}")

    confidences = [
        projections_1gw[pid].confidence
        for pid in chosen_lineup.starting_xi
        if pid in projections_1gw
    ]
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    what_would_change = [
        "Late injury/suspension news on any starter or the captain before the deadline.",
        (
            "A confirmed change to predicted lineups (e.g. a key player ruled out in a "
            "press conference) after this report was generated."
        ),
        (
            "A price change that alters what you can afford -- re-run before the deadline "
            "if it's been more than a few hours."
        ),
    ]
    if recommended.moves:
        what_would_change.append(
            "If the sell candidate's price rises before you act, your effective budget "
            "shrinks and this exact transfer may no longer be affordable."
        )

    points_hit = recommended.points_hit
    approve = (
        recommended.recommended
        and avg_confidence >= 0.4
        and not any("exceeds MAX_TRANSFER_RISK" in note for note in recommended.risk_notes)
    )
    approve_state = "APPROVE" if approve else "DO NOT APPROVE"
    if points_hit > 0 and approve:
        approve_state += f" (accepts a -{points_hit} hit)"

    return WeeklyReport(
        team_id=team_id,
        gameweek=gameweek,
        generated_at=datetime.now(UTC).isoformat(),
        squad_state=squad_state,
        transfer_scenarios=transfer_scenarios,
        recommended_scenario=recommended,
        lineups_by_strategy=lineups_by_strategy,
        chosen_strategy=chosen_strategy,
        player_names=player_names,
        key_risks=key_risks[:10],
        confidence=avg_confidence,
        what_would_change_this=what_would_change,
        approve_state=approve_state,
        projection_model=projection_model,
    )


def render_markdown(report: WeeklyReport) -> str:
    names = report.player_names
    lines: list[str] = []
    lines.append(f"# FPL Weekly Plan -- Gameweek {report.gameweek}")
    lines.append(f"_Generated {report.generated_at}_")
    lines.append("")
    lines.append(
        "> This is a recommendation, not a guarantee. Expected points are a modelled "
        "estimate with a documented margin of error -- see floor/ceiling and confidence below."
    )
    lines.append("")
    if report.projection_model == "ml":
        lines.append(
            "> Projections use the trained ML model (docs/ROADMAP.md Phase 2 -- see "
            "reports/model_backtest.md for how it was validated) wherever it covers a "
            "player, falling back to the hand-coded baseline otherwise. Each player's "
            "rationale (not shown in this summary) names which one produced their figure."
        )
    else:
        lines.append(
            "> Projections use the hand-coded baseline model only (PROJECTION_MODEL=baseline)."
        )
    lines.append("")

    lines.append("## Decision")
    lines.append(f"**{report.approve_state}**")
    lines.append("")
    r = report.recommended_scenario
    lines.append(f"**Recommended action:** {r.label}")
    if r.moves:
        for m in r.moves:
            lines.append(
                f"- Sell **{_name(names, m.sell_player_id)}** "
                f"(sell value £{m.sell_value_tenths / 10:.1f}m) -> "
                f"Buy **{_name(names, m.buy_player_id)}** (£{m.buy_price_tenths / 10:.1f}m)"
            )
    lines.append(f"- Transfer cost: **-{r.points_hit} points**")
    lines.append(f"- Resulting bank: £{r.resulting_bank_tenths / 10:.1f}m")
    lines.append(f"- Expected points before: {r.baseline_ep_1gw} (next GW) / {r.baseline_ep_5gw} (5 GW)")
    lines.append(f"- Expected points after: {r.new_ep_1gw} (next GW) / {r.new_ep_5gw} (5 GW)")
    lines.append(f"- Net expected gain (5 GW, after any hit): **{r.net_ep_gain_5gw:+.2f}**")
    lines.append(f"- Rationale: {r.rationale}")
    lines.append("")

    lines.append("## Best alternative options")
    for alt in [s for s in report.transfer_scenarios if s is not r][:4]:
        lines.append(
            f"- {alt.label}: net 5-GW gain {alt.net_ep_gain_5gw:+.2f}, hit -{alt.points_hit}"
        )
    lines.append("")

    chosen = report.lineups_by_strategy[report.chosen_strategy]
    lines.append(f"## Recommended Starting XI ({report.chosen_strategy}, formation {chosen.formation})")
    for pid in chosen.starting_xi:
        tag = ""
        if pid == chosen.captain_id:
            tag = " (C)"
        elif pid == chosen.vice_captain_id:
            tag = " (VC)"
        lines.append(f"- {_name(names, pid)}{tag}")
    lines.append("")
    lines.append("## Bench order")
    for i, pid in enumerate(chosen.bench_order, start=1):
        lines.append(f"{i}. {_name(names, pid)}")
    lines.append("")
    lines.append(f"**Captain:** {_name(names, chosen.captain_id)}  ")
    lines.append(f"**Vice-captain:** {_name(names, chosen.vice_captain_id)}")
    lines.append("")

    lines.append("## Strategy comparison")
    lines.append("| Strategy | Expected pts | Floor | Ceiling |")
    lines.append("|---|---|---|---|")
    for strat, lineup in report.lineups_by_strategy.items():
        lines.append(
            f"| {strat} | {lineup.total_expected_points} | {lineup.total_floor_points} | "
            f"{lineup.total_ceiling_points} |"
        )
    lines.append("")

    lines.append(f"## Confidence: {report.confidence:.0%}")
    lines.append("")
    lines.append("## Key risks")
    for risk in report.key_risks or ["None flagged this week."]:
        lines.append(f"- {risk}")
    lines.append("")

    lines.append("## What would change this recommendation")
    for item in report.what_would_change_this:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Squad state")
    lines.append(f"- Free transfers available: {report.squad_state.free_transfers_available}")
    lines.append(f"- Bank: £{report.squad_state.bank}m")
    lines.append(f"- Squad value: £{report.squad_state.squad_value}m")
    lines.append(f"- Wildcard available: {report.squad_state.wildcard_available}")
    lines.append(f"- Free Hit available: {report.squad_state.free_hit_available}")
    lines.append(f"- Bench Boost available: {report.squad_state.bench_boost_available}")
    lines.append(f"- Triple Captain available: {report.squad_state.triple_captain_available}")

    return "\n".join(lines)


def save_report(report: WeeklyReport, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = reports_dir / f"gw{report.gameweek}_{timestamp}"

    md_path = base.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    json_path = base.with_suffix(".json")
    payload = {
        "team_id": report.team_id,
        "gameweek": report.gameweek,
        "generated_at": report.generated_at,
        "approve_state": report.approve_state,
        "projection_model": report.projection_model,
        "confidence": report.confidence,
        "recommended_scenario": json.loads(report.recommended_scenario.model_dump_json()),
        "transfer_scenarios": [json.loads(s.model_dump_json()) for s in report.transfer_scenarios],
        "lineups_by_strategy": {
            k: json.loads(v.model_dump_json()) for k, v in report.lineups_by_strategy.items()
        },
        "key_risks": report.key_risks,
        "what_would_change_this": report.what_would_change_this,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md_path, json_path
