from __future__ import annotations

import pytest

from fpl_automate.optimization.lineup import LineupOptimizationError, optimize_lineup
from fpl_automate.storage.models import PlayerProjection, Position
from tests.conftest import make_player


def _proj(player_id, expected, floor=None, ceiling=None):
    return PlayerProjection(
        player_id=player_id,
        gameweek=1,
        expected_points=expected,
        floor_points=floor if floor is not None else expected * 0.7,
        ceiling_points=ceiling if ceiling is not None else expected * 1.3,
        confidence=0.8,
    )


def _standard_squad():
    squad = []
    pid = 1
    for _ in range(2):
        squad.append(make_player(pid, position=Position.GOALKEEPER, team_id=pid))
        pid += 1
    for _ in range(5):
        squad.append(make_player(pid, position=Position.DEFENDER, team_id=pid))
        pid += 1
    for _ in range(5):
        squad.append(make_player(pid, position=Position.MIDFIELDER, team_id=pid))
        pid += 1
    for _ in range(3):
        squad.append(make_player(pid, position=Position.FORWARD, team_id=pid))
        pid += 1
    return squad


def test_optimize_lineup_respects_formation_limits():
    squad = _standard_squad()
    projections = {p.id: _proj(p.id, expected=5.0 + p.id * 0.1) for p in squad}
    result = optimize_lineup(squad, projections, strategy="balanced")

    by_id = {p.id: p for p in squad}
    starters = [by_id[pid] for pid in result.starting_xi]
    n_gk = sum(1 for p in starters if p.position == Position.GOALKEEPER)
    n_def = sum(1 for p in starters if p.position == Position.DEFENDER)
    n_mid = sum(1 for p in starters if p.position == Position.MIDFIELDER)
    n_fwd = sum(1 for p in starters if p.position == Position.FORWARD)

    assert len(result.starting_xi) == 11
    assert n_gk == 1
    assert 3 <= n_def <= 5
    assert 2 <= n_mid <= 5
    assert 1 <= n_fwd <= 3
    assert n_def + n_mid + n_fwd == 10


def test_captain_is_highest_projected_starter():
    squad = _standard_squad()
    projections = {p.id: _proj(p.id, expected=5.0) for p in squad}
    # Make one midfielder a clear standout so we can assert on it deterministically.
    standout_id = squad[7].id  # a midfielder
    projections[standout_id] = _proj(standout_id, expected=50.0)

    result = optimize_lineup(squad, projections, strategy="balanced")
    assert result.captain_id == standout_id
    assert standout_id in result.starting_xi


def test_bench_gk_is_always_last_in_bench_order():
    squad = _standard_squad()
    projections = {p.id: _proj(p.id, expected=5.0) for p in squad}
    result = optimize_lineup(squad, projections, strategy="balanced")
    by_id = {p.id: p for p in squad}
    assert by_id[result.bench_order[-1]].position == Position.GOALKEEPER


def test_missing_projection_raises():
    squad = _standard_squad()
    projections = {p.id: _proj(p.id, expected=5.0) for p in squad[:-1]}
    with pytest.raises(LineupOptimizationError):
        optimize_lineup(squad, projections, strategy="balanced")


def test_strategies_can_select_different_lineups_via_floor_vs_ceiling():
    squad = _standard_squad()
    # One midfielder: low floor/high ceiling (differential). Another: high floor/low ceiling (safe).
    projections = {p.id: _proj(p.id, expected=4.0, floor=3.0, ceiling=5.0) for p in squad}
    risky_id = squad[7].id
    safe_id = squad[8].id
    projections[risky_id] = _proj(risky_id, expected=4.0, floor=0.5, ceiling=15.0)
    projections[safe_id] = _proj(safe_id, expected=4.0, floor=3.9, ceiling=4.2)

    conservative = optimize_lineup(squad, projections, strategy="conservative")
    aggressive = optimize_lineup(squad, projections, strategy="aggressive")
    assert aggressive.captain_id == risky_id
    assert conservative.captain_id != risky_id


def _squad_with_a_safe_and_a_volatile_standout():
    """A squad where one player has a slightly higher expected value but
    much wider floor/ceiling spread than another -- built so risk_aversion
    can plausibly flip which one gets captained."""
    squad = _standard_squad()
    projections = {p.id: _proj(p.id, expected=5.0, floor=4.5, ceiling=5.5) for p in squad}
    safe_id = squad[-1].id
    volatile_id = squad[-2].id
    projections[safe_id] = _proj(safe_id, expected=8.0, floor=7.0, ceiling=9.0)
    projections[volatile_id] = _proj(volatile_id, expected=8.5, floor=1.0, ceiling=16.0)
    return squad, projections, safe_id, volatile_id


def test_risk_adjusted_with_zero_aversion_matches_balanced_exactly():
    squad, projections, _safe_id, _volatile_id = _squad_with_a_safe_and_a_volatile_standout()

    balanced = optimize_lineup(squad, projections, strategy="balanced")
    risk_adjusted = optimize_lineup(squad, projections, strategy="risk_adjusted", risk_aversion=0.0)

    assert risk_adjusted.captain_id == balanced.captain_id
    assert set(risk_adjusted.starting_xi) == set(balanced.starting_xi)
    assert risk_adjusted.total_risk_adjusted_score == pytest.approx(balanced.total_expected_points)


def test_risk_adjusted_result_carries_the_aversion_used_and_a_score():
    squad, projections, _safe_id, _volatile_id = _squad_with_a_safe_and_a_volatile_standout()

    result = optimize_lineup(squad, projections, strategy="risk_adjusted", risk_aversion=1.5)
    assert result.risk_aversion == 1.5
    assert result.total_risk_adjusted_score is not None


def test_non_risk_adjusted_strategies_leave_the_risk_fields_unset():
    squad = _standard_squad()
    projections = {p.id: _proj(p.id, expected=5.0) for p in squad}
    result = optimize_lineup(squad, projections, strategy="balanced")
    assert result.risk_aversion is None
    assert result.total_risk_adjusted_score is None


def test_higher_risk_aversion_shifts_captaincy_toward_the_lower_variance_player():
    squad, projections, safe_id, volatile_id = _squad_with_a_safe_and_a_volatile_standout()

    # At aversion=0, the higher-mean (volatile) player should win captaincy --
    # same as "balanced" would pick, since variance isn't penalised at all yet.
    low_aversion = optimize_lineup(squad, projections, strategy="risk_adjusted", risk_aversion=0.0)
    assert low_aversion.captain_id == volatile_id

    # At a high enough aversion, the quadratic variance penalty on
    # captaincy should flip the choice to the safer player despite its
    # slightly lower expected points.
    high_aversion = optimize_lineup(squad, projections, strategy="risk_adjusted", risk_aversion=1.0)
    assert high_aversion.captain_id == safe_id
