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
from fpl_automate.risk.classification import (
    RiskProfile,
    RiskTier,
    classify_squad_risk,
    estimate_std_dev,
    tier_counts,
)
from fpl_automate.risk.covariance import FixtureCorrelation, PortfolioPlayer, portfolio_variance
from fpl_automate.storage.models import Fixture, Player, PlayerProjection, SquadState
from fpl_automate.transfers.engine import TransferScenario
from fpl_automate.transfers.planning import TransferTimingPlan


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
    squad_risk_profiles: dict[int, RiskProfile] = field(default_factory=dict)
    fixture_correlation: FixtureCorrelation = field(default_factory=FixtureCorrelation.zero)
    starting_xi_independent_variance: float = 0.0
    starting_xi_correlated_variance: float = 0.0
    transfer_timing: TransferTimingPlan | None = None


def _opponent_team_for_gameweek(fixtures: list[Fixture], team_id: int, gameweek: int) -> int | None:
    """team_id's opponent in `gameweek`, or None for a blank gameweek. A
    team appearing in more than one fixture that gameweek (double
    gameweek) only keeps the first -- the same documented simplification
    `projections/ml/live.py` uses for the same reason."""
    for fx in fixtures:
        if fx.event != gameweek:
            continue
        if fx.team_h == team_id:
            return fx.team_a
        if fx.team_a == team_id:
            return fx.team_h
    return None


def _starting_xi_portfolio_variance(
    starting_xi: list[int],
    projections: dict[int, PlayerProjection],
    all_players_by_id: dict[int, Player],
    fixtures: list[Fixture],
    gameweek: int,
    correlation: FixtureCorrelation,
) -> tuple[float, float]:
    """Returns (independent_variance, correlation_aware_variance) for the
    starting XI's summed points -- see `risk/covariance.py`. The
    independent figure is what `risk/portfolio.py`'s per-player scoring
    implicitly assumes; the correlated figure is the real, measured
    answer wherever a correlation estimate has been computed."""
    players = []
    for pid in starting_xi:
        proj = projections.get(pid)
        player = all_players_by_id.get(pid)
        if proj is None or player is None:
            continue
        opponent = _opponent_team_for_gameweek(fixtures, player.team_id, gameweek)
        players.append(
            PortfolioPlayer(
                player_id=pid, sigma=estimate_std_dev(proj), team_id=player.team_id, opponent_team_id=opponent
            )
        )
    independent = portfolio_variance(players, FixtureCorrelation.zero())
    correlated = portfolio_variance(players, correlation)
    return independent, correlated


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
    fixtures: list[Fixture] | None = None,
    fixture_correlation: FixtureCorrelation | None = None,
    transfer_timing: TransferTimingPlan | None = None,
) -> WeeklyReport:
    fixtures = fixtures or []
    fixture_correlation = fixture_correlation or FixtureCorrelation.zero()
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

    squad_projections = {
        pid: projections_1gw[pid] for pid in [p.element_id for p in squad_state.picks] if pid in projections_1gw
    }
    squad_risk_profiles = classify_squad_risk(squad_projections)

    independent_variance, correlated_variance = _starting_xi_portfolio_variance(
        chosen_lineup.starting_xi, projections_1gw, all_players_by_id, fixtures, gameweek, fixture_correlation
    )

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
        squad_risk_profiles=squad_risk_profiles,
        fixture_correlation=fixture_correlation,
        starting_xi_independent_variance=independent_variance,
        starting_xi_correlated_variance=correlated_variance,
        transfer_timing=transfer_timing,
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
                f"Buy **{_name(names, m.buy_player_id)}** (£{m.buy_price_tenths / 10:.1f}m) "
                f"-- risk: **{m.buy_risk_tier.value}**"
            )
    lines.append(f"- Transfer cost: **-{r.points_hit} points**")
    lines.append(f"- Resulting bank: £{r.resulting_bank_tenths / 10:.1f}m")
    lines.append(f"- Expected points before: {r.baseline_ep_1gw} (next GW) / {r.baseline_ep_5gw} (5 GW)")
    lines.append(f"- Expected points after: {r.new_ep_1gw} (next GW) / {r.new_ep_5gw} (5 GW)")
    lines.append(f"- Net expected gain (5 GW, after any hit): **{r.net_ep_gain_5gw:+.2f}**")
    lines.append(f"- Rationale: {r.rationale}")
    lines.append("")

    tt = report.transfer_timing
    if tt is not None:
        lines.append("### Transfer timing")
        lines.append(
            "Compares making this transfer now against delaying it, using each gameweek's "
            "*own* projection rather than the 5-GW total above -- so a real fixture swing "
            "(a tough game now, an easy one in a couple of weeks) can change the recommended "
            "timing, not just the recommended player. See `transfers/planning.py` for the "
            "documented simplification (assumes stable prices/roles while waiting)."
        )
        lines.append("")
        lines.append(tt.rationale)
        if tt.best_execute_at_offset != 0:
            lines.append(
                f"- Now: {tt.cumulative_gain_now:+.2f} pts captured over this window | "
                f"Wait {tt.best_execute_at_offset} GW(s): {tt.cumulative_gain_at_best:+.2f} pts"
            )
        lines.append("")

    lines.append("## Best alternative options")
    for alt in [s for s in report.transfer_scenarios if s is not r][:4]:
        risk_tags = ", ".join(m.buy_risk_tier.value for m in alt.moves) or "n/a"
        lines.append(
            f"- {alt.label}: net 5-GW gain {alt.net_ep_gain_5gw:+.2f}, hit -{alt.points_hit}, "
            f"risk: {risk_tags}"
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
    lines.append("| Strategy | Expected pts | Floor | Ceiling | Risk-adjusted score |")
    lines.append("|---|---|---|---|---|")
    for strat, lineup in report.lineups_by_strategy.items():
        ra_score = (
            f"{lineup.total_risk_adjusted_score} (aversion={lineup.risk_aversion})"
            if lineup.total_risk_adjusted_score is not None
            else "n/a"
        )
        lines.append(
            f"| {strat} | {lineup.total_expected_points} | {lineup.total_floor_points} | "
            f"{lineup.total_ceiling_points} | {ra_score} |"
        )
    lines.append("")
    if any(lineup.total_risk_adjusted_score is not None for lineup in report.lineups_by_strategy.values()):
        lines.append(
            "> `risk_adjusted` picks captaincy specifically to minimise the extra variance a "
            "captaincy multiplier adds (see `risk/portfolio.py`) -- it can name a different "
            "captain than the other strategies even when its starting XI is identical to "
            "`balanced`'s."
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

    lines.append("## Squad risk profile")
    lines.append(
        "Safe/balanced/risky classification per player (see `risk/classification.py`) -- "
        "based on how wide the model's floor-to-ceiling spread is relative to its expected "
        "points, with injury doubts, rotation risk, and blank gameweeks always counting as "
        "at least as risky as the numbers alone suggest."
    )
    lines.append("")
    counts = tier_counts(report.squad_risk_profiles)
    lines.append(
        f"**{counts[RiskTier.SAFE]} safe, {counts[RiskTier.BALANCED]} balanced, "
        f"{counts[RiskTier.RISKY]} risky** (of {len(report.squad_risk_profiles)} squad players with a projection)."
    )
    lines.append("")
    lines.append("| Player | Risk | Reasons |")
    lines.append("|---|---|---|")
    for pid, profile in sorted(
        report.squad_risk_profiles.items(), key=lambda kv: kv[1].tier.value
    ):
        lines.append(f"| {_name(names, pid)} | {profile.tier.value} | {'; '.join(profile.reasons)} |")
    lines.append("")

    fc = report.fixture_correlation
    indep_sd = report.starting_xi_independent_variance**0.5
    corr_sd = report.starting_xi_correlated_variance**0.5
    lines.append("### Starting XI portfolio variance")
    lines.append(
        "`risk/portfolio.py`'s per-player risk_adjusted scoring assumes each player's "
        "points are independent; this measures the actual starting XI's variance "
        "accounting for same-team/same-fixture correlation (`risk/covariance.py`), which "
        "is almost always higher once picks concentrate on one team's defence or attack."
    )
    lines.append("")
    if fc.n_same_team_gameweeks > 0:
        pct_diff = (
            (corr_sd - indep_sd) / indep_sd * 100 if indep_sd > 0 else 0.0
        )
        lines.append(
            f"- Independent-variance assumption: std dev {indep_sd:.2f} pts "
            f"(variance {report.starting_xi_independent_variance:.2f})"
        )
        lines.append(
            f"- Correlation-aware (same-team rho={fc.same_team_rho:+.2f}, opponent "
            f"rho={fc.opponent_rho:+.2f}): std dev {corr_sd:.2f} pts "
            f"(variance {report.starting_xi_correlated_variance:.2f}, {pct_diff:+.0f}%)"
        )
    else:
        lines.append(
            "- No fixture-correlation estimate available yet (run `fpl-automate backtest` "
            "at least once) -- falling back to the independent-variance assumption: "
            f"std dev {indep_sd:.2f} pts."
        )
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
        "squad_risk_profiles": {
            str(pid): json.loads(profile.model_dump_json())
            for pid, profile in report.squad_risk_profiles.items()
        },
        "fixture_correlation": {
            "same_team_rho": report.fixture_correlation.same_team_rho,
            "opponent_rho": report.fixture_correlation.opponent_rho,
            "n_same_team_gameweeks": report.fixture_correlation.n_same_team_gameweeks,
            "n_opponent_gameweeks": report.fixture_correlation.n_opponent_gameweeks,
        },
        "starting_xi_independent_variance": report.starting_xi_independent_variance,
        "starting_xi_correlated_variance": report.starting_xi_correlated_variance,
        "transfer_timing": (
            {
                "sell_player_id": report.transfer_timing.sell_player_id,
                "buy_player_id": report.transfer_timing.buy_player_id,
                "per_gameweek_differential": report.transfer_timing.per_gameweek_differential,
                "best_execute_at_offset": report.transfer_timing.best_execute_at_offset,
                "cumulative_gain_now": report.transfer_timing.cumulative_gain_now,
                "cumulative_gain_at_best": report.transfer_timing.cumulative_gain_at_best,
                "rationale": report.transfer_timing.rationale,
            }
            if report.transfer_timing is not None
            else None
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md_path, json_path
