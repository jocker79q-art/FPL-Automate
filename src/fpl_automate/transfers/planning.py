"""Multi-gameweek transfer TIMING: given a candidate transfer already
identified by `transfers/engine.py`'s single-gameweek search, decides not
just *whether* to make it but *when*, over a short forward lookahead
window -- using genuine per-gameweek (not horizon-summed) projections, so
a real fixture swing (the buy candidate has a tough fixture this week but
a great one in two) can change the recommended *timing*, not just the
recommended player.

Bounded lookahead, not full dynamic programming over the whole season:
for one candidate transfer, evaluates each possible execution gameweek
0..horizon-1 (0 = this gameweek) and picks the one that captures the most
cumulative (buy - sell) points differential across the window -- a small,
tractable search (at most `horizon` options), not an exhaustive search
over sequences of transfers or squads.

Per-gameweek isolation, not a new projection source: `workflow.py`
already requests projections at horizons 1..5 (cumulative points from
now through that many gameweeks); `per_gameweek_points` recovers each
*individual* gameweek's contribution by differencing consecutive
cumulative horizons (`horizon[k] - horizon[k-1]`). This works for both
the ML model (which already walks real per-gameweek fixtures internally)
and the baseline model (whose own multi-gameweek total is closer to an
even split across gameweeks -- differencing it is an honest reflection
of that documented simplification, not a new inaccuracy introduced here).

Known simplification, documented rather than hidden: assumes both
players' price and role stay stable across the window (no simulated
price rises/falls, or a lineup change beyond what the projection model
itself already encodes) -- "waiting" is treated as free except for the
forfeited points differential, which is exactly the risk the weekly
report's own "what would change this" section already warns about (a
price rise while you wait). Chip timing (Wildcard/Free Hit/Bench Boost/
Triple Captain) and multi-transfer sequences are both out of scope here
-- this reasons about the timing of one ordinary transfer only.
"""
from __future__ import annotations

from dataclasses import dataclass

from fpl_automate.storage.models import PlayerProjection


def per_gameweek_points(
    projections_by_horizon: dict[int, dict[int, PlayerProjection]], player_id: int, horizon: int
) -> list[float]:
    """This player's expected points for each *individual* gameweek from
    now (offset 0) through `horizon - 1` gameweeks ahead, recovered by
    differencing consecutive cumulative horizons. Requires
    `projections_by_horizon` to contain every integer horizon from 1 to
    `horizon`. A player absent from a horizon's dict (shouldn't happen
    once the baseline fallback has merged in, but handled defensively)
    contributes 0.0 for that gameweek."""

    def _cumulative(h: int) -> float:
        proj = projections_by_horizon.get(h, {}).get(player_id)
        return proj.expected_points if proj is not None else 0.0

    points = [_cumulative(1)]
    for h in range(2, horizon + 1):
        points.append(_cumulative(h) - _cumulative(h - 1))
    return points


@dataclass(frozen=True)
class TransferTimingPlan:
    sell_player_id: int
    buy_player_id: int
    per_gameweek_differential: list[float]  # index k = k gameweeks from now
    best_execute_at_offset: int  # 0 = this gameweek
    cumulative_gain_now: float
    cumulative_gain_at_best: float
    rationale: str


def plan_transfer_timing(
    sell_player_id: int,
    buy_player_id: int,
    projections_by_horizon: dict[int, dict[int, PlayerProjection]],
    horizon: int,
) -> TransferTimingPlan:
    """Compares executing this transfer now (offset 0) against delaying
    it to each later gameweek within `horizon`, scored by how much of the
    (buy - sell) points differential across the whole window each timing
    actually captures -- delaying forfeits the differential in every
    gameweek before execution, so delay is only ever better when the buy
    candidate's near-term fixture is genuinely worse than the sell
    candidate's and that reverses later in the window."""
    buy_points = per_gameweek_points(projections_by_horizon, buy_player_id, horizon)
    sell_points = per_gameweek_points(projections_by_horizon, sell_player_id, horizon)
    diffs = [b - s for b, s in zip(buy_points, sell_points, strict=True)]

    cumulative_by_start = [sum(diffs[start:]) for start in range(horizon)]
    best_offset = max(range(horizon), key=lambda t: cumulative_by_start[t])
    cumulative_now = cumulative_by_start[0]
    cumulative_best = cumulative_by_start[best_offset]

    if best_offset == 0 or cumulative_best <= cumulative_now:
        rationale = (
            "Make the transfer now -- waiting only forfeits the buy candidate's points "
            "advantage, with no offsetting benefit in this window."
        )
    else:
        gain = cumulative_best - cumulative_now
        rationale = (
            f"Consider waiting {best_offset} gameweek(s): the buy candidate's projection "
            f"this week ({diffs[0]:+.2f} vs. the sell candidate) is weaker than usual "
            f"(a fixture swing), reversing by gameweek +{best_offset} -- delaying captures "
            f"{gain:+.2f} more points over this window than acting immediately."
        )

    return TransferTimingPlan(
        sell_player_id=sell_player_id,
        buy_player_id=buy_player_id,
        per_gameweek_differential=diffs,
        best_execute_at_offset=best_offset,
        cumulative_gain_now=round(cumulative_now, 2),
        cumulative_gain_at_best=round(cumulative_best, 2),
        rationale=rationale,
    )
