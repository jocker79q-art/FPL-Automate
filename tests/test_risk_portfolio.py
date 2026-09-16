from __future__ import annotations

import pytest

from fpl_automate.risk.classification import estimate_std_dev
from fpl_automate.risk.portfolio import captain_marginal_score, risk_adjusted_score
from fpl_automate.storage.models import PlayerProjection


def _proj(expected, floor, ceiling):
    return PlayerProjection(
        player_id=1, gameweek=1, expected_points=expected, floor_points=floor, ceiling_points=ceiling,
        confidence=0.8,
    )


def test_risk_adjusted_score_with_zero_aversion_equals_expected_points():
    proj = _proj(expected=6.0, floor=0.0, ceiling=20.0)  # huge variance, shouldn't matter at aversion=0
    assert risk_adjusted_score(proj, risk_aversion=0.0) == proj.expected_points


def test_risk_adjusted_score_penalizes_variance_as_aversion_increases():
    proj = _proj(expected=6.0, floor=2.0, ceiling=10.0)
    sigma = estimate_std_dev(proj)
    assert sigma > 0

    low = risk_adjusted_score(proj, risk_aversion=0.5)
    high = risk_adjusted_score(proj, risk_aversion=2.0)
    assert high < low < proj.expected_points


def test_risk_adjusted_score_matches_the_documented_formula():
    proj = _proj(expected=6.0, floor=2.0, ceiling=10.0)
    sigma = estimate_std_dev(proj)
    expected_score = proj.expected_points - 1.5 * sigma**2
    assert risk_adjusted_score(proj, risk_aversion=1.5) == pytest.approx(expected_score)


def test_captain_marginal_score_with_zero_aversion_equals_expected_points():
    proj = _proj(expected=6.0, floor=0.0, ceiling=20.0)
    assert captain_marginal_score(proj, risk_aversion=0.0) == proj.expected_points


def test_captain_marginal_score_penalizes_variance_three_times_as_hard_as_the_starter_score():
    """The captaincy multiplier scales variance quadratically (Var(2X) =
    4*Var(X)), so the *marginal* captaincy contribution subtracts
    3*risk_aversion*sigma^2, not risk_aversion*sigma^2 -- three times the
    penalty a plain (non-captain) risk_adjusted_score applies."""
    proj = _proj(expected=6.0, floor=2.0, ceiling=10.0)
    aversion = 1.0

    starter_score = risk_adjusted_score(proj, aversion)
    captain_score = captain_marginal_score(proj, aversion)

    starter_penalty = proj.expected_points - starter_score
    captain_penalty = proj.expected_points - captain_score
    assert captain_penalty == pytest.approx(3 * starter_penalty)
    assert captain_score < starter_score


def test_captain_marginal_score_is_not_simply_double_the_starter_score():
    """A naive (incorrect) implementation might just do
    2 * risk_adjusted_score(proj, aversion) for captaincy -- this checks
    the actual formula diverges from that naive shortcut whenever there's
    real variance to penalise."""
    proj = _proj(expected=6.0, floor=1.0, ceiling=11.0)
    aversion = 1.0
    naive_double = 2 * risk_adjusted_score(proj, aversion)
    actual = captain_marginal_score(proj, aversion)
    assert actual != pytest.approx(naive_double)
    assert actual < naive_double  # the correct formula penalises volatility harder
