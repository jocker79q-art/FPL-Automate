from __future__ import annotations

from fpl_automate.storage.models import PlayerProjection
from fpl_automate.transfers.planning import per_gameweek_points, plan_transfer_timing


def _proj(player_id: int, expected: float) -> PlayerProjection:
    return PlayerProjection(
        player_id=player_id, gameweek=1, expected_points=expected, floor_points=expected, ceiling_points=expected,
        confidence=0.8,
    )


def _horizons(player_id: int, cumulative: list[float]) -> dict[int, dict[int, PlayerProjection]]:
    """cumulative[i] = total expected points through (i+1) gameweeks."""
    return {h: {player_id: _proj(player_id, cumulative[h - 1])} for h in range(1, len(cumulative) + 1)}


class TestPerGameweekPoints:
    def test_recovers_individual_gameweek_contributions_by_differencing(self):
        # Cumulative 4.0, 9.0 (gw2=5.0), 11.0 (gw3=2.0), 15.0 (gw4=4.0)
        horizons = _horizons(1, [4.0, 9.0, 11.0, 15.0])
        points = per_gameweek_points(horizons, player_id=1, horizon=4)
        assert points == [4.0, 5.0, 2.0, 4.0]

    def test_missing_player_contributes_zero(self):
        horizons: dict[int, dict[int, PlayerProjection]] = {1: {}, 2: {}}
        points = per_gameweek_points(horizons, player_id=999, horizon=2)
        assert points == [0.0, 0.0]


def _merged_horizons(
    sell_id: int, sell_cumulative: list[float], buy_id: int, buy_cumulative: list[float]
) -> dict[int, dict[int, PlayerProjection]]:
    sell_h = _horizons(sell_id, sell_cumulative)
    buy_h = _horizons(buy_id, buy_cumulative)
    return {h: {**sell_h[h], **buy_h[h]} for h in sell_h}


class TestPlanTransferTiming:
    def test_recommends_acting_now_when_buy_candidate_is_consistently_better(self):
        # Buy candidate outscores sell candidate by 2.0 every single gameweek.
        horizons = _merged_horizons(
            sell_id=1, sell_cumulative=[4.0, 8.0, 12.0, 16.0], buy_id=2, buy_cumulative=[6.0, 12.0, 18.0, 24.0]
        )
        plan = plan_transfer_timing(sell_player_id=1, buy_player_id=2, projections_by_horizon=horizons, horizon=4)
        assert plan.best_execute_at_offset == 0
        assert plan.per_gameweek_differential == [2.0, 2.0, 2.0, 2.0]
        assert "Make the transfer now" in plan.rationale

    def test_recommends_waiting_when_buy_candidate_has_a_bad_fixture_now_and_a_swing_later(self):
        # Sell candidate: 4.0 pts/gw flat. Buy candidate: a rough week 0
        # (1.0), then 6.0 pts/gw for the rest -- a real fixture swing.
        horizons = _merged_horizons(
            sell_id=1,
            sell_cumulative=[4.0, 8.0, 12.0, 16.0],
            buy_id=2,
            buy_cumulative=[1.0, 7.0, 13.0, 19.0],
        )
        plan = plan_transfer_timing(sell_player_id=1, buy_player_id=2, projections_by_horizon=horizons, horizon=4)
        # week-0 differential: 1.0 - 4.0 = -3.0 (buy candidate worse this week)
        # week-1..3 differential: 6.0 - 4.0 = +2.0 each
        assert plan.per_gameweek_differential[0] == -3.0
        assert plan.best_execute_at_offset == 1
        assert plan.cumulative_gain_at_best > plan.cumulative_gain_now
        assert "Consider waiting" in plan.rationale

    def test_ties_resolve_to_acting_now(self):
        # Buy candidate exactly matches sell candidate every gameweek --
        # no benefit either way, so the tie should resolve to "now".
        horizons = _merged_horizons(
            sell_id=1, sell_cumulative=[4.0, 8.0, 12.0], buy_id=2, buy_cumulative=[4.0, 8.0, 12.0]
        )
        plan = plan_transfer_timing(sell_player_id=1, buy_player_id=2, projections_by_horizon=horizons, horizon=3)
        assert plan.best_execute_at_offset == 0
        assert plan.cumulative_gain_now == 0.0
