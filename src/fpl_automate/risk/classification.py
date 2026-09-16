"""Classifies a player's projection as a safe bet, balanced, or risky --
using the uncertainty the projection models already produce, not a new
statistical model of its own.

=====================================================================
ASSUMPTIONS (read this before trusting a classification)
=====================================================================
1. `floor_points`/`ceiling_points` are treated as an *approximate* 80%
   central interval (10th-90th percentile) around `expected_points`, and
   converted to a standard deviation under a normality assumption:
   `sigma = (ceiling - floor) / 2.5632` (2.5632 = 2 x the z-score for the
   90th percentile, 1.2816). This is exact by construction for the ML
   model (`projections/ml/backtest.py` derives floor/ceiling from actual
   empirical residual p10/p90 quantiles) but only an approximation for the
   hand-built baseline, whose own docstring calls its band "a heuristic
   uncertainty band... not a fitted prediction interval." Both are treated
   the same way here for a single, consistent risk scale across whichever
   model produced a given projection -- a documented simplification, not
   a hidden one.
2. Risk is scored by *coefficient of variation* (sigma / expected_points),
   not raw sigma -- a low-scoring player with a small absolute spread can
   still be relatively unpredictable, and a high-scoring player with a
   large absolute spread can still be relatively reliable. Dividing by
   expected_points normalises for that.
3. `risk_flags` (see `features/engineering.py`) are qualitative signals
   the numeric spread doesn't always fully capture on its own (e.g. a
   confirmed injury doubt can still show a moderate spread if the model's
   own p(play) estimate isn't very low) and act as a floor on the tier:
   they can only push a classification *up* in risk, never down.
=====================================================================
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from fpl_automate.storage.models import PlayerProjection

# z-score for the 90th percentile of a standard normal, doubled to span a
# 10th-90th (80% central) interval.
_Z90 = 1.2816
INTERVAL_TO_STD_DEV = 2 * _Z90

# Coefficient-of-variation (sigma / expected_points) thresholds. Tuned
# against real model output (see tests/test_risk_classification.py and
# reports/model_backtest.md's own error distributions) rather than picked
# arbitrarily -- a nailed, in-form starter typically lands well under 0.3;
# a rotation-risk or minutes-uncertain player typically clears 0.6.
SAFE_MAX_CV = 0.30
RISKY_MIN_CV = 0.60

MIN_EXPECTED_POINTS_FOR_CV = 0.5  # below this, CV is numerically unstable/meaningless


class RiskTier(str, Enum):
    SAFE = "safe"
    BALANCED = "balanced"
    RISKY = "risky"


_TIER_ORDER = {RiskTier.SAFE: 0, RiskTier.BALANCED: 1, RiskTier.RISKY: 2}


def _max_tier(a: RiskTier, b: RiskTier) -> RiskTier:
    return a if _TIER_ORDER[a] >= _TIER_ORDER[b] else b


class RiskProfile(BaseModel):
    player_id: int
    tier: RiskTier
    std_dev: float
    coefficient_of_variation: float | None  # None when expected_points is ~0 (undefined)
    reasons: list[str] = []


def estimate_std_dev(projection: PlayerProjection) -> float:
    """Approximate standard deviation implied by the projection's own
    floor/ceiling band -- see this module's docstring, assumption 1."""
    return max(0.0, (projection.ceiling_points - projection.floor_points) / INTERVAL_TO_STD_DEV)


def classify_player_risk(projection: PlayerProjection) -> RiskProfile:
    std_dev = estimate_std_dev(projection)
    reasons: list[str] = []

    if projection.expected_points < MIN_EXPECTED_POINTS_FOR_CV:
        cv = None
    else:
        cv = round(std_dev / projection.expected_points, 3)

    if cv is None:
        # Near-zero expected points with real spread above it (e.g. a
        # fringe player who might still start) is itself a risk signal,
        # not something CV can measure meaningfully.
        tier = RiskTier.RISKY if std_dev > MIN_EXPECTED_POINTS_FOR_CV else RiskTier.BALANCED
        reasons.append(f"Expected points too low ({projection.expected_points:.1f}) for a stable CV estimate.")
    elif cv <= SAFE_MAX_CV:
        tier = RiskTier.SAFE
        reasons.append(f"Tight spread relative to expectation (CV={cv:.2f}).")
    elif cv >= RISKY_MIN_CV:
        tier = RiskTier.RISKY
        reasons.append(f"Wide spread relative to expectation (CV={cv:.2f}).")
    else:
        tier = RiskTier.BALANCED
        reasons.append(f"Moderate spread relative to expectation (CV={cv:.2f}).")

    # Qualitative overrides: can only push the tier up, never down (see
    # module docstring, assumption 3).
    if "blank_gameweek" in projection.risk_flags:
        tier = _max_tier(tier, RiskTier.RISKY)
        reasons.append("Confirmed blank gameweek -- 0 points is the near-certain outcome, not a spread.")
    if "injury_or_availability_doubt" in projection.risk_flags:
        tier = _max_tier(tier, RiskTier.RISKY)
        reasons.append("Live injury/availability doubt -- real chance of not playing at all.")
    if "rotation_risk" in projection.risk_flags:
        tier = _max_tier(tier, RiskTier.BALANCED)
        reasons.append("Inconsistent minutes when selected -- rotation risk.")
    if "small_sample" in projection.risk_flags:
        tier = _max_tier(tier, RiskTier.BALANCED)
        reasons.append("Limited appearances this season -- the projection itself is less certain.")

    return RiskProfile(
        player_id=projection.player_id,
        tier=tier,
        std_dev=round(std_dev, 3),
        coefficient_of_variation=cv,
        reasons=reasons,
    )


def classify_squad_risk(projections: dict[int, PlayerProjection]) -> dict[int, RiskProfile]:
    return {player_id: classify_player_risk(proj) for player_id, proj in projections.items()}


def tier_counts(profiles: dict[int, RiskProfile]) -> dict[RiskTier, int]:
    counts = dict.fromkeys(RiskTier, 0)
    for profile in profiles.values():
        counts[profile.tier] += 1
    return counts
