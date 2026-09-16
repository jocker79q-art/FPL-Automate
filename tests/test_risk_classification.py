from __future__ import annotations

from fpl_automate.risk.classification import (
    INTERVAL_TO_STD_DEV,
    RiskTier,
    classify_player_risk,
    classify_squad_risk,
    estimate_std_dev,
    tier_counts,
)
from fpl_automate.storage.models import PlayerProjection


def _proj(
    player_id: int = 1,
    expected: float = 6.0,
    floor: float = 5.0,
    ceiling: float = 7.0,
    confidence: float = 0.8,
    risk_flags: list[str] | None = None,
) -> PlayerProjection:
    return PlayerProjection(
        player_id=player_id,
        gameweek=1,
        expected_points=expected,
        floor_points=floor,
        ceiling_points=ceiling,
        confidence=confidence,
        risk_flags=risk_flags or [],
    )


def test_estimate_std_dev_matches_the_documented_formula():
    proj = _proj(expected=6.0, floor=4.0, ceiling=8.0)
    assert estimate_std_dev(proj) == (8.0 - 4.0) / INTERVAL_TO_STD_DEV


def test_estimate_std_dev_never_negative_even_if_floor_exceeds_ceiling():
    # Shouldn't happen in practice, but a classification function must not
    # blow up or return a nonsensical negative "spread" on bad input.
    proj = _proj(floor=8.0, ceiling=4.0)
    assert estimate_std_dev(proj) == 0.0


def test_tight_spread_relative_to_expectation_is_classified_safe():
    proj = _proj(expected=6.0, floor=5.2, ceiling=6.8)  # narrow band, no risk flags
    profile = classify_player_risk(proj)
    assert profile.tier == RiskTier.SAFE
    assert profile.coefficient_of_variation is not None
    assert profile.coefficient_of_variation <= 0.30


def test_wide_spread_relative_to_expectation_is_classified_risky():
    proj = _proj(expected=3.0, floor=0.0, ceiling=8.0)  # wide band vs. a low expectation
    profile = classify_player_risk(proj)
    assert profile.tier == RiskTier.RISKY
    assert profile.coefficient_of_variation is not None
    assert profile.coefficient_of_variation >= 0.60


def test_moderate_spread_is_classified_balanced():
    proj = _proj(expected=6.0, floor=3.5, ceiling=8.9)
    profile = classify_player_risk(proj)
    assert profile.tier == RiskTier.BALANCED


def test_risk_flags_can_only_push_the_tier_up_never_down():
    # A numerically "safe" spread, but a live injury doubt -- the flag
    # must win, not the spread.
    tight_but_injured = _proj(expected=6.0, floor=5.2, ceiling=6.8, risk_flags=["injury_or_availability_doubt"])
    profile = classify_player_risk(tight_but_injured)
    assert profile.tier == RiskTier.RISKY
    assert any("injury" in r.lower() for r in profile.reasons)


def test_rotation_risk_flag_pushes_a_safe_spread_to_at_least_balanced():
    proj = _proj(expected=6.0, floor=5.2, ceiling=6.8, risk_flags=["rotation_risk"])
    profile = classify_player_risk(proj)
    assert profile.tier in (RiskTier.BALANCED, RiskTier.RISKY)
    assert profile.tier != RiskTier.SAFE


def test_small_sample_flag_prevents_a_safe_classification():
    proj = _proj(expected=6.0, floor=5.2, ceiling=6.8, risk_flags=["small_sample"])
    profile = classify_player_risk(proj)
    assert profile.tier != RiskTier.SAFE


def test_blank_gameweek_flag_forces_risky_regardless_of_spread():
    # A blank-gameweek projection is expected=floor=ceiling=0 in practice
    # (see baseline_model.project_player), which would otherwise look
    # deceptively "safe" (zero spread) if the flag didn't override it.
    proj = _proj(expected=0.0, floor=0.0, ceiling=0.0, risk_flags=["blank_gameweek"])
    profile = classify_player_risk(proj)
    assert profile.tier == RiskTier.RISKY


def test_multiple_risk_flags_do_not_downgrade_each_other():
    proj = _proj(
        expected=6.0, floor=5.2, ceiling=6.8,
        risk_flags=["small_sample", "injury_or_availability_doubt"],
    )
    profile = classify_player_risk(proj)
    assert profile.tier == RiskTier.RISKY  # the stronger flag wins, not the weaker one


def test_near_zero_expected_points_has_no_coefficient_of_variation():
    proj = _proj(expected=0.1, floor=0.0, ceiling=0.3)
    profile = classify_player_risk(proj)
    assert profile.coefficient_of_variation is None


def test_classify_squad_risk_and_tier_counts_cover_every_player():
    projections = {
        1: _proj(player_id=1, expected=6.0, floor=5.2, ceiling=6.8),  # safe
        2: _proj(player_id=2, expected=3.0, floor=0.0, ceiling=8.0),  # risky
        3: _proj(player_id=3, expected=6.0, floor=3.5, ceiling=8.9),  # balanced
    }
    profiles = classify_squad_risk(projections)
    assert set(profiles.keys()) == {1, 2, 3}
    counts = tier_counts(profiles)
    assert counts[RiskTier.SAFE] == 1
    assert counts[RiskTier.RISKY] == 1
    assert counts[RiskTier.BALANCED] == 1
    assert sum(counts.values()) == 3


def test_tier_counts_includes_zero_for_tiers_with_no_players():
    profiles = classify_squad_risk({1: _proj(player_id=1, expected=6.0, floor=5.2, ceiling=6.8)})
    counts = tier_counts(profiles)
    assert counts[RiskTier.RISKY] == 0
    assert counts[RiskTier.BALANCED] == 0
    assert counts[RiskTier.SAFE] == 1
