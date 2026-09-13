"""Interpretable baseline expected-points model.

This is deliberately NOT a machine-learned model. Every coefficient below
is a documented, hand-set assumption so the whole projection can be
explained in plain English -- "why does this player project to 5.2
points?" should always be answerable by reading this file. Machine
learning (Phase 2+) should be benchmarked against this baseline, not
replace it silently.

=====================================================================
ASSUMPTIONS (read this before trusting a projection)
=====================================================================
1. Scoring table below is the standard FPL points table. If the game's
   rules change mid-season, this file must be updated or projections for
   the changed events will be wrong.
2. "Effective appearance probability" blends the player's own status/news
   signal with their recent minutes pattern. It is a heuristic, not a
   calibrated probability -- treat p_play as directional, not exact.
3. The attacking-output estimate blends *actual* returns (goals/assists
   scored) with *underlying* output (expected goals/assists) 50/50. This
   is a deliberate regression-to-process signal: a player heavily
   over-performing their xG is nudged down, one under-performing is
   nudged up. Small-sample players lean on `blended_form`, which is
   itself already shrunk towards season points-per-game (see
   `features/engineering.py`).
4. Clean-sheet and goals-conceded rates are each player's own empirical
   per-90 rate while on the pitch, not a team-level defensive model. This
   avoids re-deriving a separate win-probability model on top of FPL's
   own fixture-difficulty ratings, but means it will not react quickly to
   e.g. a new signing improving a leaky defence.
5. Multi-gameweek horizons multiply the single-fixture expectation by the
   fixture *count* in that window, using the *average* difficulty across
   the window rather than fixture-by-fixture difficulty. Good enough to
   compare players; not precise for e.g. "GW1 easy, GW2 very hard".
6. Floor/ceiling are a heuristic uncertainty band driven by availability
   risk, minutes reliability, and small-sample size -- not a fitted
   prediction interval. Treat them as "plausible bad case" / "plausible
   good case", not a statistical confidence interval.
7. This model has NOT been backtested yet (see docs/ROADMAP.md, Phase 2).
   Until it has, treat its absolute point values as approximate and its
   *relative ranking* of similar players as more trustworthy than any
   single number.
=====================================================================
"""
from __future__ import annotations

from pydantic import BaseModel

from fpl_automate.features.engineering import PlayerFeatures
from fpl_automate.storage.models import Player, PlayerProjection, Position

GOAL_POINTS: dict[Position, int] = {
    Position.GOALKEEPER: 6,
    Position.DEFENDER: 6,
    Position.MIDFIELDER: 5,
    Position.FORWARD: 4,
}
ASSIST_POINTS = 3
CLEAN_SHEET_POINTS: dict[Position, int] = {
    Position.GOALKEEPER: 4,
    Position.DEFENDER: 4,
    Position.MIDFIELDER: 1,
    Position.FORWARD: 0,
}
GOALS_CONCEDED_PENALTY_PER_2 = 1  # -1 point per 2 goals conceded, GK/DEF only
SAVE_POINTS_PER_3 = 1  # GK only

ATTACKING_BLEND_ACTUAL_WEIGHT = 0.5  # vs. underlying (xG/xA) weight of 0.5
EMPIRICAL_VS_COMPONENT_WEIGHT = 0.55  # weight on blended_form vs. bottom-up component model

DIFFICULTY_COEFFICIENT = 0.06  # +/-6% expected points per difficulty point away from neutral (3)
HOME_ADVANTAGE_COEFFICIENT = 0.05  # up to +5% if fully home across the window


class ModelInputs(BaseModel):
    """Everything the model needs for one player/horizon; kept explicit for testability."""

    player: Player
    features: PlayerFeatures
    horizon_gameweeks: int  # 1 for "next GW", or e.g. 5 for a medium-term horizon


def _effective_appearance_probability(player: Player, features: PlayerFeatures) -> float:
    p_play = 1 - features.availability_risk
    # Soften for fringe players who are technically "available" but rarely start.
    reliability_factor = 0.5 + 0.5 * features.minutes_reliability
    return round(max(0.0, min(1.0, p_play * reliability_factor)), 3)


def _expected_minutes_given_appearance(features: PlayerFeatures) -> float:
    return 45 + 45 * features.minutes_reliability


def _attacking_rate_per_90(player: Player) -> float:
    minutes90 = player.minutes / 90 if player.minutes > 0 else 0.0
    if minutes90 < 0.5:
        return 0.0
    goal_pts = GOAL_POINTS[player.position]
    actual = (player.goals_scored * goal_pts + player.assists * ASSIST_POINTS) / minutes90
    underlying = (
        player.expected_goals * goal_pts + player.expected_assists * ASSIST_POINTS
    ) / minutes90
    return ATTACKING_BLEND_ACTUAL_WEIGHT * actual + (1 - ATTACKING_BLEND_ACTUAL_WEIGHT) * underlying


def _defensive_rate_per_90(player: Player) -> float:
    minutes90 = player.minutes / 90 if player.minutes > 0 else 0.0
    if minutes90 < 0.5:
        return 0.0
    cs_pts = CLEAN_SHEET_POINTS[player.position]
    cs_rate = (player.clean_sheets * cs_pts) / minutes90
    if player.position in (Position.GOALKEEPER, Position.DEFENDER):
        conceded_penalty = (player.goals_conceded / 2) * GOALS_CONCEDED_PENALTY_PER_2 / minutes90
    else:
        conceded_penalty = 0.0
    return cs_rate - conceded_penalty


def _save_rate_per_90(player: Player) -> float:
    if player.position != Position.GOALKEEPER:
        return 0.0
    minutes90 = player.minutes / 90 if player.minutes > 0 else 0.0
    if minutes90 < 0.5:
        return 0.0
    return (player.saves / 3) * SAVE_POINTS_PER_3 / minutes90


def _bonus_rate_per_90(player: Player) -> float:
    minutes90 = player.minutes / 90 if player.minutes > 0 else 0.0
    if minutes90 < 0.5:
        return 0.0
    return player.bonus / minutes90


def _component_points_per_appearance(player: Player, features: PlayerFeatures) -> float:
    rate_per_90 = (
        _attacking_rate_per_90(player)
        + _defensive_rate_per_90(player)
        + _save_rate_per_90(player)
        + _bonus_rate_per_90(player)
    )
    expected_minutes = _expected_minutes_given_appearance(features)
    frac_60_plus = max(0.0, min(1.0, (expected_minutes - 45) / 45))
    appearance_flat_points = 1 + frac_60_plus  # interpolates between 1 (sub cameo) and 2 (60+ mins)
    return rate_per_90 * (expected_minutes / 90) + appearance_flat_points


def _fixture_multiplier(features: PlayerFeatures) -> float:
    difficulty_delta = 3 - features.fixture_difficulty_avg  # positive = easier than neutral
    difficulty_mult = 1 + difficulty_delta * DIFFICULTY_COEFFICIENT
    home_mult = 1 + (features.home_fixture_fraction - 0.5) * 2 * HOME_ADVANTAGE_COEFFICIENT
    return difficulty_mult * home_mult


def _uncertainty_spread(player: Player, features: PlayerFeatures) -> float:
    small_sample_penalty = 0.15 if player.starts < 6 else 0.0
    spread = (
        0.20
        + 0.35 * features.availability_risk
        + 0.25 * (1 - features.minutes_reliability)
        + small_sample_penalty
    )
    return max(0.15, min(0.9, spread))


def project_player(inputs: ModelInputs) -> PlayerProjection:
    player, features = inputs.player, inputs.features
    risk_flags: list[str] = list(features.data_notes)

    if features.num_fixtures_next_n == 0:
        return PlayerProjection(
            player_id=player.id,
            gameweek=inputs.horizon_gameweeks,
            expected_points=0.0,
            floor_points=0.0,
            ceiling_points=0.0,
            confidence=0.9,  # confidently near-zero: there is simply no fixture
            risk_flags=["blank_gameweek"],
            rationale="No fixture in this window (blank gameweek) -- projected points are zero.",
        )

    p_play = _effective_appearance_probability(player, features)
    per_appearance_component = _component_points_per_appearance(player, features)
    per_appearance_blended = (
        EMPIRICAL_VS_COMPONENT_WEIGHT * features.blended_form
        + (1 - EMPIRICAL_VS_COMPONENT_WEIGHT) * per_appearance_component
    )
    fixture_mult = _fixture_multiplier(features)

    per_fixture_expected = p_play * per_appearance_blended * fixture_mult
    expected = round(per_fixture_expected * features.num_fixtures_next_n, 2)

    spread = _uncertainty_spread(player, features)
    floor_pts = round(max(0.0, expected * (1 - spread)), 2)
    ceiling_pts = round(expected * (1 + spread * 1.3), 2)

    confidence = round(
        max(
            0.0,
            min(
                1.0,
                1
                - (
                    0.5 * features.availability_risk
                    + 0.3 * (1 - features.minutes_reliability)
                    + 0.2 * min(1.0, spread)
                ),
            ),
        ),
        2,
    )

    if features.minutes_reliability < 0.5 and player.starts > 0:
        risk_flags.append("rotation_risk")
    if features.availability_risk > 0:
        risk_flags.append("injury_or_availability_doubt")
    if features.num_fixtures_next_n > inputs.horizon_gameweeks:
        risk_flags.append("double_gameweek")
    if player.starts < 6:
        risk_flags.append("small_sample")

    rationale = (
        f"p(appearance)={p_play:.2f}, blended form={features.blended_form:.2f}, "
        f"component/appearance={per_appearance_component:.2f}, "
        f"fixture x home multiplier={fixture_mult:.2f}, "
        f"fixtures in window={features.num_fixtures_next_n}."
    )

    return PlayerProjection(
        player_id=player.id,
        gameweek=inputs.horizon_gameweeks,
        expected_points=expected,
        floor_points=floor_pts,
        ceiling_points=ceiling_pts,
        confidence=confidence,
        risk_flags=risk_flags,
        rationale=rationale,
    )


def apply_captain_multiplier(projection: PlayerProjection, multiplier: int = 2) -> PlayerProjection:
    """Captaincy/triple-captain exactly doubles/triples realised points -- an FPL rule, not an
    assumption."""
    return projection.model_copy(
        update={
            "expected_points": round(projection.expected_points * multiplier, 2),
            "floor_points": round(projection.floor_points * multiplier, 2),
            "ceiling_points": round(projection.ceiling_points * multiplier, 2),
        }
    )
