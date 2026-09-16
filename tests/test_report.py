from __future__ import annotations

import json
from pathlib import Path

from fpl_automate.optimization.lineup import LineupResult
from fpl_automate.reporting.report import build_weekly_report, save_report
from fpl_automate.risk.classification import RiskTier
from fpl_automate.storage.models import ChipPlay, PlayerProjection, Position, SquadPick, SquadState
from fpl_automate.transfers.engine import TransferMove, TransferScenario
from tests.conftest import make_player


def _proj(player_id, expected, floor_ratio=0.85, ceiling_ratio=1.15):
    """Defaults to a *tight* spread (safe) so tests can opt individual
    players into wider bands explicitly, rather than every fixture player
    accidentally landing in the same tier."""
    return PlayerProjection(
        player_id=player_id,
        gameweek=1,
        expected_points=expected,
        floor_points=expected * floor_ratio,
        ceiling_points=expected * ceiling_ratio,
        confidence=0.8,
    )


def _squad_state(squad):
    picks = [
        SquadPick(element_id=p.id, squad_position=i + 1, multiplier=1, is_captain=i == 0, is_vice_captain=i == 1)
        for i, p in enumerate(squad)
    ]
    return SquadState(
        team_id=9242093,
        as_of_event=5,
        picks=picks,
        bank_tenths=10,
        squad_value_tenths=sum(p.now_cost_tenths for p in squad),
        free_transfers_available=1,
        chips_used=[ChipPlay(name="wildcard", event=3)],
        wildcard_available=False,
        free_hit_available=True,
        bench_boost_available=True,
        triple_captain_available=True,
    )


def _minimal_squad():
    return [
        make_player(1, position=Position.GOALKEEPER, team_id=1, now_cost_tenths=45),
        make_player(2, position=Position.DEFENDER, team_id=2, now_cost_tenths=45),
        make_player(3, position=Position.FORWARD, team_id=3, now_cost_tenths=90),
    ]


def _lineup_result(squad, projections):
    ids = [p.id for p in squad]
    return LineupResult(
        strategy="balanced",
        starting_xi=ids,
        bench_order=[],
        captain_id=ids[0],
        vice_captain_id=ids[1],
        formation="1-1-1",
        total_expected_points=sum(projections[pid].expected_points for pid in ids),
        total_floor_points=sum(projections[pid].floor_points for pid in ids),
        total_ceiling_points=sum(projections[pid].ceiling_points for pid in ids),
    )


def _roll_scenario(baseline_ep):
    return TransferScenario(
        label="Roll transfer (make no changes)",
        moves=[],
        free_transfers_used=0,
        paid_transfers=0,
        points_hit=0,
        resulting_bank_tenths=10,
        baseline_ep_1gw=baseline_ep,
        new_ep_1gw=baseline_ep,
        baseline_ep_5gw=baseline_ep * 5,
        new_ep_5gw=baseline_ep * 5,
        ep_gain_1gw=0.0,
        ep_gain_5gw=0.0,
        net_ep_gain_5gw=0.0,
        rationale="Roll.",
        recommended=True,
    )


def _transfer_scenario_with_move(sell_id, buy_id, buy_tier: RiskTier):
    return TransferScenario(
        label=f"sell{sell_id} -> buy{buy_id}",
        moves=[
            TransferMove(
                sell_player_id=sell_id,
                buy_player_id=buy_id,
                sell_value_tenths=45,
                buy_price_tenths=50,
                buy_risk_tier=buy_tier,
            )
        ],
        free_transfers_used=1,
        paid_transfers=0,
        points_hit=0,
        resulting_bank_tenths=5,
        baseline_ep_1gw=4.0,
        new_ep_1gw=6.0,
        baseline_ep_5gw=20.0,
        new_ep_5gw=30.0,
        ep_gain_1gw=2.0,
        ep_gain_5gw=10.0,
        net_ep_gain_5gw=10.0,
        rationale="Test move.",
        recommended=False,
    )


def test_build_weekly_report_classifies_risk_only_for_owned_squad_players():
    squad = _minimal_squad()
    projections = {p.id: _proj(p.id, expected=5.0) for p in squad}
    # A player NOT in the squad -- must not leak into squad_risk_profiles.
    outsider = make_player(999, position=Position.MIDFIELDER, team_id=9, now_cost_tenths=60)
    projections[outsider.id] = _proj(outsider.id, expected=5.0)

    all_players_by_id = {p.id: p for p in [*squad, outsider]}
    lineup = _lineup_result(squad, projections)
    scenarios = [_roll_scenario(baseline_ep=lineup.total_expected_points)]

    report = build_weekly_report(
        team_id=9242093,
        gameweek=6,
        squad_state=_squad_state(squad),
        all_players_by_id=all_players_by_id,
        projections_1gw=projections,
        projections_5gw=projections,
        transfer_scenarios=scenarios,
        lineups_by_strategy={"balanced": lineup},
        chosen_strategy="balanced",
    )

    assert set(report.squad_risk_profiles.keys()) == {p.id for p in squad}
    assert outsider.id not in report.squad_risk_profiles


def test_render_markdown_includes_squad_risk_profile_section_with_correct_counts():
    squad = _minimal_squad()
    projections = {p.id: _proj(p.id, expected=5.0) for p in squad}
    # Make squad[2] (a forward) clearly risky: wide spread + low expected points.
    projections[squad[2].id] = _proj(squad[2].id, expected=2.0, floor_ratio=0.0, ceiling_ratio=4.0)

    all_players_by_id = {p.id: p for p in squad}
    lineup = _lineup_result(squad, projections)
    scenarios = [_roll_scenario(baseline_ep=lineup.total_expected_points)]

    report = build_weekly_report(
        team_id=9242093,
        gameweek=6,
        squad_state=_squad_state(squad),
        all_players_by_id=all_players_by_id,
        projections_1gw=projections,
        projections_5gw=projections,
        transfer_scenarios=scenarios,
        lineups_by_strategy={"balanced": lineup},
        chosen_strategy="balanced",
    )

    from fpl_automate.reporting.report import render_markdown

    md = render_markdown(report)
    assert "## Squad risk profile" in md
    assert "2 safe, 0 balanced, 1 risky" in md
    assert squad[2].web_name in md


def test_render_markdown_shows_risk_tier_on_recommended_and_alternative_transfers():
    squad = _minimal_squad()
    projections = {p.id: _proj(p.id, expected=5.0) for p in squad}
    all_players_by_id = {p.id: p for p in squad}
    lineup = _lineup_result(squad, projections)

    recommended = _transfer_scenario_with_move(squad[2].id, 9999, RiskTier.RISKY)
    recommended.recommended = True
    alternative = _transfer_scenario_with_move(squad[1].id, 8888, RiskTier.SAFE)
    scenarios = [recommended, alternative]

    report = build_weekly_report(
        team_id=9242093,
        gameweek=6,
        squad_state=_squad_state(squad),
        all_players_by_id=all_players_by_id,
        projections_1gw=projections,
        projections_5gw=projections,
        transfer_scenarios=scenarios,
        lineups_by_strategy={"balanced": lineup},
        chosen_strategy="balanced",
    )

    from fpl_automate.reporting.report import render_markdown

    md = render_markdown(report)
    assert "risk: **risky**" in md
    assert "risk: safe" in md


def test_save_report_json_payload_includes_squad_risk_profiles(tmp_path: Path):
    squad = _minimal_squad()
    projections = {p.id: _proj(p.id, expected=5.0) for p in squad}
    all_players_by_id = {p.id: p for p in squad}
    lineup = _lineup_result(squad, projections)
    scenarios = [_roll_scenario(baseline_ep=lineup.total_expected_points)]

    report = build_weekly_report(
        team_id=9242093,
        gameweek=6,
        squad_state=_squad_state(squad),
        all_players_by_id=all_players_by_id,
        projections_1gw=projections,
        projections_5gw=projections,
        transfer_scenarios=scenarios,
        lineups_by_strategy={"balanced": lineup},
        chosen_strategy="balanced",
    )

    _md_path, json_path = save_report(report, tmp_path)
    payload = json.loads(json_path.read_text())
    assert set(payload["squad_risk_profiles"].keys()) == {str(p.id) for p in squad}
    for entry in payload["squad_risk_profiles"].values():
        assert entry["tier"] in ("safe", "balanced", "risky")
