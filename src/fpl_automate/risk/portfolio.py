"""Mean-variance ("Markowitz-style") risk-adjusted scoring, used by
`optimization/lineup.py`'s `"risk_adjusted"` strategy.

The objective for a non-captain starter is the classical mean-variance
trade-off: `expected_points - risk_aversion * variance`. `risk_aversion`
is the dial -- 0 recovers plain expected-points maximisation (identical to
the `"balanced"` strategy), and larger values increasingly favour a lower-
variance player over a higher-mean one.

Known simplification, stated plainly rather than hidden: this treats each
player's points as *independent* of every other player's, so a squad's
total variance is just the sum of its players' variances
(`Var(sum X_i) = sum Var(X_i)` only holds for independent X_i). In reality
players are correlated -- two players from the same match share outcome
risk (a heavy loss drags both team's players down together), and a real
covariance-aware optimizer would need a covariance matrix estimated from
historical results. That's real future work, not attempted here: it needs
enough same-fixture player-pair history to estimate reliably, which this
project doesn't yet compute (see docs/ROADMAP.md). Treating variance as
additive is the standard, honest starting simplification for exactly this
reason -- it's what a first pass at mean-variance optimization looks like
before adding covariance.

Captaincy needs its own formula, not just "double the score": doubling a
player's points (Var(2X) = 4 x Var(X)) scales variance *quadratically*,
while it scales the mean only *linearly*. Naming player c captain instead
of leaving them as a normal (non-captain) starter changes the squad's
risk-adjusted total by:

    [2*mu_c - risk_aversion*(2*sigma_c)^2] - [mu_c - risk_aversion*sigma_c^2]
  = mu_c - 3*risk_aversion*sigma_c^2

-- i.e. captaincy's marginal contribution is `mu_c - 3*risk_aversion*sigma_c^2`,
not `2 * (mu_c - risk_aversion*sigma_c^2)`. A volatile player becomes a
*much* riskier captaincy pick than the "just double it" arithmetic a naive
implementation would use suggests, since the variance penalty roughly
triples rather than doubles.
"""
from __future__ import annotations

from fpl_automate.risk.classification import estimate_std_dev
from fpl_automate.storage.models import PlayerProjection

DEFAULT_RISK_AVERSION = 1.0


def risk_adjusted_score(projection: PlayerProjection, risk_aversion: float) -> float:
    """The mean-variance objective value for `projection` as a normal
    (non-captain) starter."""
    sigma = estimate_std_dev(projection)
    return projection.expected_points - risk_aversion * sigma**2


def captain_marginal_score(projection: PlayerProjection, risk_aversion: float) -> float:
    """The marginal risk-adjusted contribution of naming this player
    captain, on top of what they'd already contribute as a normal starter
    -- see this module's docstring for the derivation. Used to rank
    captaincy candidates specifically; ranking by `risk_adjusted_score`
    alone would under-penalise volatile players for the captaincy
    decision specifically, since that function doesn't know about the
    multiplier at all.
    """
    sigma = estimate_std_dev(projection)
    return projection.expected_points - 3 * risk_aversion * sigma**2
