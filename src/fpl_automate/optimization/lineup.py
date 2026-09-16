"""Starting-XI / bench / captaincy optimisation as a constrained ILP (via PuLP).

The squad structure (2 GKP, 5 DEF, 5 MID, 3 FWD = 15) is fixed by FPL's own
squad rules, so this module only decides *which 11 of those 15 play*,
subject to formation limits, then orders the bench and picks
captain/vice-captain. It does not touch who is *in* the 15 -- that is the
transfer engine's job (`transfers/engine.py`).

Four strategies change which point estimate the objective (and captaincy
choice) optimises for:
  * conservative   -> floor_points  (protects against a bad week)
  * balanced       -> expected_points
  * aggressive     -> ceiling_points (chases upside / differentials)
  * risk_adjusted  -> a genuine mean-variance objective (expected_points
    minus a risk-aversion-weighted variance penalty, see
    `risk/portfolio.py`) -- the first three are single-number
    substitutions that don't model variance directly; this one does,
    with the documented independent-variance simplification and a
    mathematically correct (quadratic, not linear) variance treatment for
    the captaincy multiplier specifically.
"""
from __future__ import annotations

from typing import Literal

import pulp
from pydantic import BaseModel

from fpl_automate.risk.portfolio import captain_marginal_score, risk_adjusted_score
from fpl_automate.storage.models import Player, PlayerProjection, Position

Strategy = Literal["conservative", "balanced", "aggressive", "risk_adjusted"]

_STRATEGY_FIELD: dict[str, str] = {
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
    risk_aversion: float | None = None  # only set when strategy == "risk_adjusted"
    total_risk_adjusted_score: float | None = None  # the optimizer's own objective value achieved


class LineupOptimizationError(RuntimeError):
    pass


def _score(projection: PlayerProjection, strategy: Strategy, risk_aversion: float) -> float:
    if strategy == "risk_adjusted":
        return risk_adjusted_score(projection, risk_aversion)
    return getattr(projection, _STRATEGY_FIELD[strategy])


def optimize_lineup(
    squad: list[Player],
    projections: dict[int, PlayerProjection],
    strategy: Strategy = "balanced",
    risk_aversion: float = 1.0,
) -> LineupResult:
    """`risk_aversion` is only used when `strategy == "risk_adjusted"`
    (ignored otherwise) -- see `risk/portfolio.py` for what it means and
    `LineupResult.total_risk_adjusted_score` for the objective value it
    produced.
    """
    missing = [p.id for p in squad if p.id not in projections]
    if missing:
        raise LineupOptimizationError(f"Missing projections for player IDs: {missing}")
    if len(squad) != 15:
        raise LineupOptimizationError(f"Expected a 15-man squad, got {len(squad)}")

    prob = pulp.LpProblem("fpl_lineup", pulp.LpMaximize)
    x = {p.id: pulp.LpVariable(f"start_{p.id}", cat="Binary") for p in squad}

    objective_scores = {p.id: _score(projections[p.id], strategy, risk_aversion) for p in squad}
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

    if strategy == "risk_adjusted":
        # Ranking captaincy candidates by objective_scores here would
        # under-penalise volatile players: that score doesn't know about
        # the captaincy multiplier at all. captain_marginal_score does --
        # see risk/portfolio.py for the quadratic-variance derivation.
        captain_rank = sorted(
            starting_ids,
            key=lambda pid: captain_marginal_score(projections[pid], risk_aversion),
            reverse=True,
        )
    else:
        captain_rank = sorted(starting_ids, key=lambda pid: objective_scores[pid], reverse=True)
    captain_id = captain_rank[0]
    vice_captain_id = captain_rank[1]

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
    # Captain doubles their contribution across all three point views --
    # these are literal points totals regardless of optimization strategy,
    # so this stays a simple doubling even in risk_adjusted mode (the
    # quadratic variance treatment only applies to the risk-adjusted
    # *objective* value below, not to the plain expected/floor/ceiling
    # point totals themselves).
    totals["total_expected_points"] += projections[captain_id].expected_points
    totals["total_floor_points"] += projections[captain_id].floor_points
    totals["total_ceiling_points"] += projections[captain_id].ceiling_points

    total_risk_adjusted_score = None
    if strategy == "risk_adjusted":
        non_captain_sum = sum(objective_scores[pid] for pid in starting_ids)
        total_risk_adjusted_score = round(
            non_captain_sum + captain_marginal_score(projections[captain_id], risk_aversion), 2
        )

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
        risk_aversion=risk_aversion if strategy == "risk_adjusted" else None,
        total_risk_adjusted_score=total_risk_adjusted_score,
    )
