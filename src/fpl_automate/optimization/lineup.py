"""Starting-XI / bench / captaincy optimisation as a constrained ILP (via PuLP).

The squad structure (2 GKP, 5 DEF, 5 MID, 3 FWD = 15) is fixed by FPL's own
squad rules, so this module only decides *which 11 of those 15 play*,
subject to formation limits, then orders the bench and picks
captain/vice-captain. It does not touch who is *in* the 15 -- that is the
transfer engine's job (`transfers/engine.py`).

Three strategies change which point estimate the objective (and captaincy
choice) optimises for:
  * conservative -> floor_points  (protects against a bad week)
  * balanced     -> expected_points
  * aggressive   -> ceiling_points (chases upside / differentials)
This is a simplification: it does not directly model variance/covariance
between players (e.g. two players from the same match are correlated), it
just re-optimises against a different single-number target per player.
"""
from __future__ import annotations

from typing import Literal

import pulp
from pydantic import BaseModel

from fpl_automate.storage.models import Player, PlayerProjection, Position

Strategy = Literal["conservative", "balanced", "aggressive"]

_STRATEGY_FIELD: dict[Strategy, str] = {
    "conservative": "floor_points",
    "balanced": "expected_points",
    "aggressive": "ceiling_points",
}

FORMATION_LIMITS: dict[Position, tuple[int, int]] = {
    Position.GOALKEEPER: (1, 1),
    Position.DEFENDER: (3, 5),
    Position.MIDFIELDER: (2, 5),
    Position.FORWARD: (1, 3),
}
STARTING_XI_SIZE = 11


class LineupResult(BaseModel):
    strategy: Strategy
    starting_xi: list[int]  # player IDs
    bench_order: list[int]  # player IDs, index 0 = first sub (GK bench is always last)
    captain_id: int
    vice_captain_id: int
    formation: str
    total_expected_points: float
    total_floor_points: float
    total_ceiling_points: float


class LineupOptimizationError(RuntimeError):
    pass


def _score(projection: PlayerProjection, strategy: Strategy) -> float:
    return getattr(projection, _STRATEGY_FIELD[strategy])


def optimize_lineup(
    squad: list[Player],
    projections: dict[int, PlayerProjection],
    strategy: Strategy = "balanced",
) -> LineupResult:
    missing = [p.id for p in squad if p.id not in projections]
    if missing:
        raise LineupOptimizationError(f"Missing projections for player IDs: {missing}")
    if len(squad) != 15:
        raise LineupOptimizationError(f"Expected a 15-man squad, got {len(squad)}")

    prob = pulp.LpProblem("fpl_lineup", pulp.LpMaximize)
    x = {p.id: pulp.LpVariable(f"start_{p.id}", cat="Binary") for p in squad}

    objective_scores = {p.id: _score(projections[p.id], strategy) for p in squad}
    prob += pulp.lpSum(x[p.id] * objective_scores[p.id] for p in squad)

    prob += pulp.lpSum(x.values()) == STARTING_XI_SIZE

    for position, (min_count, max_count) in FORMATION_LIMITS.items():
        players_in_pos = [p for p in squad if p.position == position]
        prob += pulp.lpSum(x[p.id] for p in players_in_pos) >= min_count
        prob += pulp.lpSum(x[p.id] for p in players_in_pos) <= max_count

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise LineupOptimizationError(
            f"Lineup optimisation did not find an optimal solution (status={pulp.LpStatus[status]})"
        )

    starting_ids = [p.id for p in squad if x[p.id].value() == 1]
    bench_ids = [p.id for p in squad if p.id not in starting_ids]

    by_id = {p.id: p for p in squad}
    counts = {pos: sum(1 for pid in starting_ids if by_id[pid].position == pos) for pos in FORMATION_LIMITS}
    formation = f"{counts[Position.DEFENDER]}-{counts[Position.MIDFIELDER]}-{counts[Position.FORWARD]}"

    ranked_starters = sorted(starting_ids, key=lambda pid: objective_scores[pid], reverse=True)
    captain_id = ranked_starters[0]
    vice_captain_id = ranked_starters[1]

    bench_gk = next(pid for pid in bench_ids if by_id[pid].position == Position.GOALKEEPER)
    bench_outfield = sorted(
        (pid for pid in bench_ids if pid != bench_gk),
        key=lambda pid: objective_scores[pid],
        reverse=True,
    )
    bench_order = bench_outfield + [bench_gk]

    totals = {
        "total_expected_points": sum(projections[pid].expected_points for pid in starting_ids),
        "total_floor_points": sum(projections[pid].floor_points for pid in starting_ids),
        "total_ceiling_points": sum(projections[pid].ceiling_points for pid in starting_ids),
    }
    # Captain doubles their contribution across all three point views.
    totals["total_expected_points"] += projections[captain_id].expected_points
    totals["total_floor_points"] += projections[captain_id].floor_points
    totals["total_ceiling_points"] += projections[captain_id].ceiling_points

    return LineupResult(
        strategy=strategy,
        starting_xi=starting_ids,
        bench_order=bench_order,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        formation=formation,
        total_expected_points=round(totals["total_expected_points"], 2),
        total_floor_points=round(totals["total_floor_points"], 2),
        total_ceiling_points=round(totals["total_ceiling_points"], 2),
    )
