"""Same-fixture player-outcome correlation, estimated from real backtest
residuals and shrunk toward zero with a Ledoit-Wolf-style empirical-Bayes
estimator -- used by `portfolio_variance` to replace `risk/portfolio.py`'s
documented independent-variance simplification (`sum(sigma_i**2)`) with
the real, measured answer: how correlated are two players' outcomes when
they're on the same team, or facing each other in the same match?

Why a *structural* two-parameter model (`same_team_rho`, `opponent_rho`)
rather than a full player-by-player covariance matrix: raw historical data
for any *specific* pair of players is far too sparse to estimate a
pairwise correlation individually (most player pairs share only a handful
of co-appearances across a season, and squads/lineups change every
window). That's exactly the problem covariance shrinkage exists to solve
-- the standard fix is shrinking toward a *structured* target rather than
estimating every entry of a huge, mostly-noise matrix. The structure used
here is pooled across the whole league: every same-team pair shares one
estimated correlation (the intuition: goalkeepers/defenders on the same
team share the same clean-sheet outcome almost exactly, and attackers on
the same team benefit together from a high-scoring win), and every
same-fixture opposing-team pair shares another. This is coarser than a
full Ledoit-Wolf covariance matrix, and is documented as such here -- a
genuine first step past pure independence, not a claim of full
player-level covariance modelling. Extending this to per-player-pair
estimates (or a factor model conditioning on position, e.g. "defenders'
clean-sheet correlation" vs. "attackers' goal-involvement correlation"
separately) is real future work, not attempted here (see docs/ROADMAP.md).

Methodology: for each gameweek, standardize that gameweek's model
residuals (actual - predicted) within each position (dividing by that
position's own pooled residual std dev, so goalkeepers and forwards are on
a comparable scale), then compute the average pairwise product of
standardized residuals across all same-team pairs that gameweek (and,
separately, all cross-team pairs within the same fixture) using the
identity `sum_{i<j} z_i*z_j = ((sum z)^2 - sum z^2) / 2` for same-team
pairs (and the simpler `sum(z_a) * sum(z_b)` for two disjoint groups a/b
in the cross-team case) -- this *is* the empirical correlation estimate
for that gameweek's pooled population, without an explicit O(n^2)
pairwise loop.

Treating each gameweek as one independent sample of the true correlation
gives `G` estimates rho_1..rho_G; the final estimate is shrunk toward 0 by
a single-parameter empirical-Bayes (Ledoit-Wolf-style) formula:

    lambda* = Var(rho_g) / (Var(rho_g) + rho_hat^2)
    rho_shrunk = (1 - lambda*) * rho_hat

-- shrinking harder when the estimate is noisy (high sampling variance of
the per-gameweek estimates) relative to its own magnitude, exactly
Ledoit-Wolf's own logic (trade estimation variance for a controlled bias
toward a simpler target), applied here to a scalar rather than a full
matrix.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CORRELATION_FILE = "fixture_correlation.json"


@dataclass(frozen=True)
class FixtureCorrelation:
    same_team_rho: float
    opponent_rho: float
    n_same_team_gameweeks: int
    n_opponent_gameweeks: int

    @classmethod
    def zero(cls) -> FixtureCorrelation:
        """The independence assumption `risk/portfolio.py` used before this
        module existed -- the safe fallback when no estimate has been
        computed yet (e.g. a fresh checkout before `backtest` has run)."""
        return cls(same_team_rho=0.0, opponent_rho=0.0, n_same_team_gameweeks=0, n_opponent_gameweeks=0)


def _pair_sum_and_count(z: np.ndarray) -> tuple[float, float]:
    n = len(z)
    if n < 2:
        return 0.0, 0.0
    total = float(z.sum())
    sq = float((z**2).sum())
    return (total**2 - sq) / 2, n * (n - 1) / 2


def _cross_sum_and_count(z_a: np.ndarray, z_b: np.ndarray) -> tuple[float, float]:
    if len(z_a) == 0 or len(z_b) == 0:
        return 0.0, 0.0
    return float(z_a.sum() * z_b.sum()), float(len(z_a) * len(z_b))


def _shrink_to_zero(samples: np.ndarray) -> tuple[float, float]:
    """Single-parameter Ledoit-Wolf-style shrinkage of a scalar estimate
    toward 0. Returns (shrunk_estimate, shrinkage_intensity in [0, 1])."""
    g = len(samples)
    if g < 2:
        return 0.0, 1.0
    rho_hat = float(np.mean(samples))
    sampling_var = float(np.var(samples, ddof=1)) / g
    denom = sampling_var + rho_hat**2
    shrinkage = 1.0 if denom == 0 else min(1.0, sampling_var / denom)
    return (1 - shrinkage) * rho_hat, shrinkage


def estimate_fixture_correlation(residuals: pd.DataFrame) -> FixtureCorrelation:
    """`residuals` needs one row per (played) player-gameweek from a
    backtest, with columns: `season`, `GW`, `position`, `team_id`,
    `opponent_team_id`, `residual` (actual - predicted points). See this
    module's docstring for the full derivation."""
    df = residuals.copy()
    pos_std = df.groupby("position")["residual"].transform(lambda s: s.std(ddof=0) or 1.0)
    df["z"] = df["residual"] / pos_std.replace(0, 1.0)

    same_team_rho_per_gw: list[float] = []
    opponent_rho_per_gw: list[float] = []

    for _, gw_group in df.groupby(["season", "GW"]):
        by_team: dict[int, np.ndarray] = {
            int(team_id): g["z"].to_numpy() for team_id, g in gw_group.groupby("team_id")
        }

        same_sum = same_count = 0.0
        for z in by_team.values():
            s, c = _pair_sum_and_count(z)
            same_sum += s
            same_count += c
        if same_count > 0:
            same_team_rho_per_gw.append(same_sum / same_count)

        opp_sum = opp_count = 0.0
        seen_fixtures: set[frozenset[int]] = set()
        pairs = gw_group[["team_id", "opponent_team_id"]].dropna().drop_duplicates()
        for team_id, opp_id in pairs.itertuples(index=False):
            team_id, opp_id = int(team_id), int(opp_id)
            key = frozenset({team_id, opp_id})
            if key in seen_fixtures or opp_id not in by_team:
                continue
            seen_fixtures.add(key)
            s, c = _cross_sum_and_count(by_team[team_id], by_team[opp_id])
            opp_sum += s
            opp_count += c
        if opp_count > 0:
            opponent_rho_per_gw.append(opp_sum / opp_count)

    same_rho, _ = _shrink_to_zero(np.array(same_team_rho_per_gw))
    opp_rho, _ = _shrink_to_zero(np.array(opponent_rho_per_gw))

    return FixtureCorrelation(
        same_team_rho=round(same_rho, 4),
        opponent_rho=round(opp_rho, 4),
        n_same_team_gameweeks=len(same_team_rho_per_gw),
        n_opponent_gameweeks=len(opponent_rho_per_gw),
    )


def save_correlation(correlation: FixtureCorrelation, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / CORRELATION_FILE).write_text(json.dumps(asdict(correlation), indent=2), encoding="utf-8")


def load_correlation(models_dir: Path) -> FixtureCorrelation:
    """Never raises for a missing file -- a fresh checkout before
    `backtest` has ever run is a normal, expected state, and callers
    should treat the zero-correlation fallback as "independence assumed",
    not as an error."""
    path = models_dir / CORRELATION_FILE
    if not path.exists():
        return FixtureCorrelation.zero()
    return FixtureCorrelation(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class PortfolioPlayer:
    player_id: int
    sigma: float
    team_id: int
    opponent_team_id: int | None = None


def portfolio_variance(players: list[PortfolioPlayer], correlation: FixtureCorrelation) -> float:
    """Total variance of the summed points of `players`, accounting for
    same-team and same-fixture-opponent correlation -- the real, measured
    answer wherever a correlation estimate is available, replacing
    `risk/portfolio.py`'s documented independent-variance simplification
    (falls back to that exact independent-sum case, unchanged, when
    `correlation` is `FixtureCorrelation.zero()`).

    Clamped to 0: the pooled two-parameter structural correlation model
    isn't guaranteed positive-semi-definite for every possible squad
    composition the way a properly estimated full covariance matrix would
    be, so this is a documented safety net, not a claim that negative
    variance is ever the "true" answer."""
    total = sum(p.sigma**2 for p in players)
    for i, a in enumerate(players):
        for b in players[i + 1 :]:
            if a.team_id == b.team_id:
                total += 2 * correlation.same_team_rho * a.sigma * b.sigma
            elif (
                a.opponent_team_id is not None
                and a.opponent_team_id == b.team_id
                and b.opponent_team_id == a.team_id
            ):
                total += 2 * correlation.opponent_rho * a.sigma * b.sigma
    return max(0.0, total)
