from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_automate.risk.covariance import (
    FixtureCorrelation,
    PortfolioPlayer,
    estimate_fixture_correlation,
    load_correlation,
    portfolio_variance,
    save_correlation,
)


def _correlated_residuals(seed: int = 0, n_gw: int = 30) -> pd.DataFrame:
    """Two teams, 3 players each, every gameweek. Same-team players share a
    per-gameweek team-level shock (shock variance 1 vs. 0.09 idiosyncratic
    noise, so the true same-team correlation is 1/1.09 ~= 0.92); the two
    teams' shocks are independent of each other, so the true opponent
    correlation is ~0."""
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, n_gw + 1):
        team1_shock = rng.normal(0, 1)
        team2_shock = rng.normal(0, 1)
        for shock, team_id, opp_id in [(team1_shock, 1, 2), (team2_shock, 2, 1)]:
            for _ in range(3):
                rows.append(
                    {
                        "season": "2099-00",
                        "GW": gw,
                        "position": "MID",
                        "team_id": team_id,
                        "opponent_team_id": opp_id,
                        "residual": shock + rng.normal(0, 0.3),
                    }
                )
    return pd.DataFrame(rows)


def _independent_residuals(seed: int = 1, n_gw: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, n_gw + 1):
        for team_id, opp_id in [(1, 2), (2, 1)]:
            for _ in range(3):
                rows.append(
                    {
                        "season": "2099-00",
                        "GW": gw,
                        "position": "MID",
                        "team_id": team_id,
                        "opponent_team_id": opp_id,
                        "residual": rng.normal(0, 1),
                    }
                )
    return pd.DataFrame(rows)


class TestEstimateFixtureCorrelation:
    def test_detects_strong_same_team_correlation_from_a_shared_shock(self):
        result = estimate_fixture_correlation(_correlated_residuals())
        assert result.same_team_rho > 0.5
        assert result.n_same_team_gameweeks == 30

    def test_opponent_correlation_stays_near_zero_when_team_shocks_are_independent(self):
        result = estimate_fixture_correlation(_correlated_residuals())
        assert abs(result.opponent_rho) < 0.2

    def test_shrinks_toward_zero_for_genuinely_independent_residuals(self):
        result = estimate_fixture_correlation(_independent_residuals())
        assert abs(result.same_team_rho) < 0.2
        assert abs(result.opponent_rho) < 0.2

    def test_zero_gameweek_samples_when_no_rows_share_a_team_or_fixture(self):
        # Every row a lone player on its own team -- no same-team pairs,
        # no resolvable opponent (opponent never appears as its own team_id).
        df = pd.DataFrame(
            {
                "season": ["2099-00"] * 3,
                "GW": [1, 2, 3],
                "position": ["MID"] * 3,
                "team_id": [1, 2, 3],
                "opponent_team_id": [9, 9, 9],
                "residual": [0.5, -0.3, 0.1],
            }
        )
        result = estimate_fixture_correlation(df)
        assert result.n_same_team_gameweeks == 0
        assert result.n_opponent_gameweeks == 0
        assert result.same_team_rho == 0.0
        assert result.opponent_rho == 0.0


class TestFixtureCorrelationZero:
    def test_zero_factory_has_no_correlation_and_no_samples(self):
        zero = FixtureCorrelation.zero()
        assert zero.same_team_rho == 0.0
        assert zero.opponent_rho == 0.0
        assert zero.n_same_team_gameweeks == 0
        assert zero.n_opponent_gameweeks == 0


class TestSaveLoadCorrelation:
    def test_round_trip(self, tmp_path: Path):
        original = FixtureCorrelation(
            same_team_rho=0.15, opponent_rho=-0.05, n_same_team_gameweeks=38, n_opponent_gameweeks=38
        )
        save_correlation(original, tmp_path)
        loaded = load_correlation(tmp_path)
        assert loaded == original

    def test_returns_zero_when_never_saved(self, tmp_path: Path):
        assert load_correlation(tmp_path) == FixtureCorrelation.zero()


class TestPortfolioVariance:
    def test_matches_independent_sum_when_correlation_is_zero(self):
        players = [
            PortfolioPlayer(player_id=1, sigma=2.0, team_id=1),
            PortfolioPlayer(player_id=2, sigma=3.0, team_id=2),
        ]
        variance = portfolio_variance(players, FixtureCorrelation.zero())
        assert variance == pytest.approx(2.0**2 + 3.0**2)

    def test_positive_same_team_correlation_increases_variance_above_independent_sum(self):
        players = [
            PortfolioPlayer(player_id=1, sigma=2.0, team_id=1),
            PortfolioPlayer(player_id=2, sigma=2.0, team_id=1),
        ]
        correlation = FixtureCorrelation(same_team_rho=0.5, opponent_rho=0.0, n_same_team_gameweeks=10, n_opponent_gameweeks=10)
        independent = portfolio_variance(players, FixtureCorrelation.zero())
        correlated = portfolio_variance(players, correlation)
        assert correlated > independent
        assert correlated == pytest.approx(independent + 2 * 0.5 * 2.0 * 2.0)

    def test_negative_opponent_correlation_decreases_variance_below_independent_sum(self):
        players = [
            PortfolioPlayer(player_id=1, sigma=2.0, team_id=1, opponent_team_id=2),
            PortfolioPlayer(player_id=2, sigma=2.0, team_id=2, opponent_team_id=1),
        ]
        correlation = FixtureCorrelation(same_team_rho=0.0, opponent_rho=-0.3, n_same_team_gameweeks=10, n_opponent_gameweeks=10)
        independent = portfolio_variance(players, FixtureCorrelation.zero())
        correlated = portfolio_variance(players, correlation)
        assert correlated < independent

    def test_unrelated_players_are_never_covaried(self):
        """Different teams and not opponents this gameweek (e.g. one has a
        blank gameweek) -- no correlation term should apply between them."""
        players = [
            PortfolioPlayer(player_id=1, sigma=2.0, team_id=1, opponent_team_id=None),
            PortfolioPlayer(player_id=2, sigma=2.0, team_id=2, opponent_team_id=3),
        ]
        correlation = FixtureCorrelation(same_team_rho=0.5, opponent_rho=0.5, n_same_team_gameweeks=10, n_opponent_gameweeks=10)
        assert portfolio_variance(players, correlation) == pytest.approx(2.0**2 + 2.0**2)

    def test_never_returns_negative_variance(self):
        players = [PortfolioPlayer(player_id=i, sigma=5.0, team_id=1) for i in range(6)]
        # An implausibly extreme same_team_rho, to exercise the safety clamp.
        correlation = FixtureCorrelation(same_team_rho=-5.0, opponent_rho=0.0, n_same_team_gameweeks=10, n_opponent_gameweeks=10)
        assert portfolio_variance(players, correlation) == 0.0
