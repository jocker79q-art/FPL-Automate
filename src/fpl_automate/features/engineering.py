"""Feature engineering: turn raw FPL data into signals the projection model uses.

Design note on API cost: per-gameweek granular history for a player only
comes from a dedicated `element-summary/{id}` call. Calling that for the
full ~700-player pool on every run would be slow and impolite to a third
party's free API. So by default this module works off the season-aggregate
fields bootstrap-static already gives for every player (form, points per
game, minutes, starts, season-total expected-goal-involvement, etc.), and
only uses per-gameweek `recent_history` (from element-summary) when the
caller supplies it -- e.g. for a shortlist of the user's squad + realistic
transfer candidates, not the entire player pool.
"""
from __future__ import annotations

from pydantic import BaseModel

from fpl_automate.storage.models import Fixture, Player, Team

# How many recent match "logs" count as a small sample worth shrinking
# towards the season-long rate, so one or two big/blank games don't swing
# the projection too far (explicit anti-overfitting-to-recent-form measure).
FORM_SHRINKAGE_MIN_STARTS = 6


class PlayerFeatures(BaseModel):
    player_id: int
    position: int
    price: float

    # Output signals
    blended_form: float  # shrunk blend of recent `form` and season points-per-game
    points_per_90: float
    xgi_per_90: float
    xgc_per_90: float  # expected goals conceded per 90 (defenders/keepers: lower is better)

    # Fixtures (next N gameweeks from "now")
    fixture_difficulty_avg: float  # 1 (easiest) - 5 (hardest), lower is better for the player
    home_fixture_fraction: float
    num_fixtures_next_n: int  # 0 = blank gameweek run, >1 in one GW = double gameweek present

    # Risk
    availability_risk: float  # 0 (nailed & fit) - 1 (will not play)
    minutes_reliability: float  # 0 (fringe/rotation risk) - 1 (nailed starter)
    on_penalties: bool
    on_direct_freekicks: bool
    on_corners: bool

    # Value
    value_score: float  # season total points per £1m, for context/tie-breaks

    data_notes: list[str] = []


def _shrink(recent: float, baseline: float, sample_weight: float) -> float:
    """Weighted-average `recent` towards `baseline`; sample_weight in [0, 1]."""
    sample_weight = max(0.0, min(1.0, sample_weight))
    return recent * sample_weight + baseline * (1 - sample_weight)


MINUTES_RELIABILITY_SAMPLE_SIZE = 5  # starts needed before we trust the ratio fully
NEUTRAL_RELIABILITY_PRIOR = 0.5  # "unknown" prior for a player we've barely seen


def compute_minutes_reliability(player: Player) -> float:
    if player.starts == 0:
        return 0.0
    per_start = player.minutes_per_start
    # Reward players who, when they start, tend to play (close to) the full match.
    completion = min(1.0, per_start / 80.0)
    # Reward players who start most of the games their team has played, approximated
    # by comparing starts to minutes/90 as a proxy for appearances made.
    appearances_est = max(player.starts, round(player.minutes / 90))
    start_share = player.starts / appearances_est if appearances_est else 0.0
    raw_reliability = 0.6 * completion + 0.4 * start_share

    # A player with only 1-2 starts can look "nailed" by the ratio above purely by
    # chance -- pull small samples towards a neutral prior, same anti-overfitting
    # principle as the form-shrinkage in `compute_player_features`.
    sample_confidence = min(1.0, player.starts / MINUTES_RELIABILITY_SAMPLE_SIZE)
    blended = sample_confidence * raw_reliability + (1 - sample_confidence) * NEUTRAL_RELIABILITY_PRIOR
    return round(blended, 3)


def compute_fixture_window(
    team_id: int, fixtures: list[Fixture], from_event: int, num_gameweeks: int
) -> tuple[float, float, int]:
    """Returns (avg_difficulty, home_fraction, fixture_count) for a team's next window."""
    window_events = set(range(from_event, from_event + num_gameweeks))
    relevant = [
        f
        for f in fixtures
        if f.event in window_events and (f.team_h == team_id or f.team_a == team_id)
    ]
    if not relevant:
        return (3.0, 0.5, 0)  # neutral defaults for a blank run; caller should flag this

    difficulties = []
    home_count = 0
    for f in relevant:
        if f.team_h == team_id:
            difficulties.append(f.team_h_difficulty)
            home_count += 1
        else:
            difficulties.append(f.team_a_difficulty)
    avg_difficulty = sum(difficulties) / len(difficulties)
    home_fraction = home_count / len(relevant)
    return (round(avg_difficulty, 2), round(home_fraction, 2), len(relevant))


def compute_player_features(
    player: Player,
    team: Team,
    all_fixtures: list[Fixture],
    from_event: int,
    fixture_window_gameweeks: int = 5,
) -> PlayerFeatures:
    notes: list[str] = []
    minutes90 = player.minutes / 90 if player.minutes > 0 else 0.0

    points_per_90 = round(player.total_points / minutes90, 2) if minutes90 > 0.5 else 0.0
    xgi_per_90 = round(player.expected_goal_involvements / minutes90, 3) if minutes90 > 0.5 else 0.0
    xgc_per_90 = round(player.expected_goals_conceded / minutes90, 3) if minutes90 > 0.5 else 0.0

    sample_weight = min(1.0, player.starts / FORM_SHRINKAGE_MIN_STARTS)
    if sample_weight < 1.0:
        notes.append(
            f"Only {player.starts} start(s) this season; form is shrunk towards "
            "points-per-game to avoid overweighting a small sample."
        )
    blended_form = round(_shrink(player.form, player.points_per_game, sample_weight), 2)

    avg_difficulty, home_fraction, fixture_count = compute_fixture_window(
        player.team_id, all_fixtures, from_event, fixture_window_gameweeks
    )
    if fixture_count == 0:
        notes.append(f"No fixtures found for this team in GW{from_event}-"
                      f"{from_event + fixture_window_gameweeks - 1}: likely a blank gameweek.")
    elif fixture_count > fixture_window_gameweeks:
        notes.append("More fixtures than gameweeks in this window: a double gameweek is present.")

    availability_risk = player.availability.rotation_or_injury_risk
    if availability_risk > 0:
        notes.append(f"Availability flag: status={player.availability.status!r}, "
                      f"news={player.availability.news!r}")

    minutes_reliability = compute_minutes_reliability(player)
    if minutes_reliability < 0.5 and player.starts > 0:
        notes.append("Inconsistent minutes when selected: rotation risk.")

    value_score = round(player.total_points / player.price, 2) if player.price > 0 else 0.0

    return PlayerFeatures(
        player_id=player.id,
        position=int(player.position),
        price=player.price,
        blended_form=blended_form,
        points_per_90=points_per_90,
        xgi_per_90=xgi_per_90,
        xgc_per_90=xgc_per_90,
        fixture_difficulty_avg=avg_difficulty,
        home_fixture_fraction=home_fraction,
        num_fixtures_next_n=fixture_count,
        availability_risk=round(availability_risk, 2),
        minutes_reliability=minutes_reliability,
        on_penalties=player.penalties_order == 1,
        on_direct_freekicks=player.direct_freekicks_order == 1,
        on_corners=player.corners_and_indirect_freekicks_order == 1,
        value_score=value_score,
        data_notes=notes,
    )
