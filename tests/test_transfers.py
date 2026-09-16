from __future__ import annotations

from fpl_automate.risk.classification import RiskTier
from fpl_automate.storage.models import ChipPlay, PlayerProjection, Position, SquadPick, SquadState
from fpl_automate.transfers.engine import recommend_transfers
from tests.conftest import make_player


def _proj(player_id, expected, floor_ratio=0.7, ceiling_ratio=1.3):
    return PlayerProjection(
        player_id=player_id,
        gameweek=1,
        expected_points=expected,
        floor_points=expected * floor_ratio,
        ceiling_points=expected * ceiling_ratio,
        confidence=0.8,
    )


def _standard_squad(price_for_weak_forward=90, weak_forward_id=15):
    squad = []
    pid = 1
    for _ in range(2):
        squad.append(make_player(pid, position=Position.GOALKEEPER, team_id=pid, now_cost_tenths=45))
        pid += 1
    for _ in range(5):
        squad.append(make_player(pid, position=Position.DEFENDER, team_id=pid, now_cost_tenths=45))
        pid += 1
    for _ in range(5):
        squad.append(make_player(pid, position=Position.MIDFIELDER, team_id=pid, now_cost_tenths=60))
        pid += 1
    for _ in range(3):
        squad.append(make_player(pid, position=Position.FORWARD, team_id=pid, now_cost_tenths=price_for_weak_forward))
        pid += 1
    return squad


def _squad_state(squad, bank_tenths=20, free_transfers=1):
    picks = [
        SquadPick(element_id=p.id, squad_position=i + 1, multiplier=1, is_captain=i == 0, is_vice_captain=i == 1)
        for i, p in enumerate(squad)
    ]
    return SquadState(
        team_id=9242093,
        as_of_event=5,
        picks=picks,
        bank_tenths=bank_tenths,
        squad_value_tenths=sum(p.now_cost_tenths for p in squad),
        free_transfers_available=free_transfers,
        chips_used=[ChipPlay(name="wildcard", event=3)],
        wildcard_available=False,
        free_hit_available=True,
        bench_boost_available=True,
        triple_captain_available=True,
    )


def test_roll_scenario_always_present():
    squad = _standard_squad()
    state = _squad_state(squad)
    projections = {p.id: _proj(p.id, expected=4.0) for p in squad}
    scenarios = recommend_transfers(
        squad, state, transfers_history=[], all_players=squad,
        projections_1gw=projections, projections_3gw=projections, projections_5gw=projections,
    )
    assert any(s.label.startswith("Roll") for s in scenarios)


def test_clear_upgrade_is_recommended_and_respects_budget():
    squad = _standard_squad(price_for_weak_forward=90, weak_forward_id=15)
    weak_forward = squad[-1]
    state = _squad_state(squad, bank_tenths=10, free_transfers=1)

    projections = {p.id: _proj(p.id, expected=4.0) for p in squad}
    projections[weak_forward.id] = _proj(weak_forward.id, expected=0.5)  # clearly the worst

    # A much better, affordable forward not in the squad. Sell value ~90 (no purchase
    # history) + bank 10 = 100 tenths affordable ceiling.
    great_forward = make_player(999, position=Position.FORWARD, team_id=50, now_cost_tenths=95)
    pool = squad + [great_forward]
    projections[great_forward.id] = _proj(great_forward.id, expected=12.0)

    scenarios = recommend_transfers(
        squad, state, transfers_history=[], all_players=pool,
        projections_1gw=projections, projections_3gw=projections, projections_5gw=projections,
    )

    best = scenarios[0]
    assert best.label != "Roll transfer (make no changes)"
    assert best.recommended is True
    assert best.points_hit == 0  # had a free transfer available
    assert best.resulting_bank_tenths >= 0


def test_club_limit_is_never_violated_in_suggestions():
    squad = _standard_squad()
    state = _squad_state(squad, bank_tenths=50, free_transfers=1)
    projections = {p.id: _proj(p.id, expected=4.0) for p in squad}

    # Flood the pool with cheap, high-projection forwards all from the SAME club as an
    # existing squad forward, to try to trigger a 4th-from-one-club violation.
    weak_forward = squad[-1]
    projections[weak_forward.id] = _proj(weak_forward.id, expected=0.1)
    same_club_id = squad[-2].team_id  # another forward's club, already has 1 player in squad
    tempting_players = [
        make_player(2000 + i, position=Position.FORWARD, team_id=same_club_id, now_cost_tenths=50)
        for i in range(5)
    ]
    for tp in tempting_players:
        projections[tp.id] = _proj(tp.id, expected=20.0)

    pool = squad + tempting_players
    scenarios = recommend_transfers(
        squad, state, transfers_history=[], all_players=pool,
        projections_1gw=projections, projections_3gw=projections, projections_5gw=projections,
    )
    for s in scenarios:
        resulting_ids = {p.id for p in squad}
        for m in s.moves:
            resulting_ids.discard(m.sell_player_id)
            resulting_ids.add(m.buy_player_id)
        counts: dict[int, int] = {}
        by_id = {p.id: p for p in pool}
        for pid in resulting_ids:
            counts[by_id[pid].team_id] = counts.get(by_id[pid].team_id, 0) + 1
        assert all(c <= 3 for c in counts.values())


def test_hit_recommended_only_when_gain_clears_threshold():
    squad = _standard_squad()
    weak_forward = squad[-1]
    state = _squad_state(squad, bank_tenths=10, free_transfers=0)  # no free transfer -> any move costs 4
    projections = {p.id: _proj(p.id, expected=4.0) for p in squad}
    projections[weak_forward.id] = _proj(weak_forward.id, expected=3.9)  # barely worse

    slightly_better = make_player(999, position=Position.FORWARD, team_id=50, now_cost_tenths=95)
    projections[slightly_better.id] = _proj(slightly_better.id, expected=4.2)
    pool = squad + [slightly_better]

    scenarios = recommend_transfers(
        squad, state, transfers_history=[], all_players=pool,
        projections_1gw=projections, projections_3gw=projections, projections_5gw=projections,
        min_net_gain_to_recommend=2.0,
    )
    roll = next(s for s in scenarios if s.label.startswith("Roll"))
    assert roll.recommended is True  # tiny gain shouldn't clear a -4 hit


def test_suggested_transfer_moves_carry_the_buy_players_risk_tier():
    squad = _standard_squad()
    weak_forward = squad[-1]
    state = _squad_state(squad, bank_tenths=10, free_transfers=1)
    projections = {p.id: _proj(p.id, expected=4.0) for p in squad}
    projections[weak_forward.id] = _proj(weak_forward.id, expected=0.5)

    # A much better forward, but with a wide floor/ceiling spread -- should
    # come through as "risky", not silently defaulted to something else.
    volatile_forward = make_player(999, position=Position.FORWARD, team_id=50, now_cost_tenths=95)
    projections[volatile_forward.id] = _proj(volatile_forward.id, expected=6.0, floor_ratio=0.0, ceiling_ratio=2.5)
    pool = squad + [volatile_forward]

    scenarios = recommend_transfers(
        squad, state, transfers_history=[], all_players=pool,
        projections_1gw=projections, projections_3gw=projections, projections_5gw=projections,
    )

    best = scenarios[0]
    assert best.moves[0].buy_player_id == volatile_forward.id
    assert best.moves[0].buy_risk_tier == RiskTier.RISKY
