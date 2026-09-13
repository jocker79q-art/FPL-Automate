from __future__ import annotations

from fpl_automate.features.engineering import (
    compute_fixture_window,
    compute_minutes_reliability,
    compute_player_features,
)
from tests.conftest import make_fixture, make_player, make_team


def test_fixture_window_averages_difficulty_and_home_fraction():
    fixtures = [
        make_fixture(1, event=10, team_h=1, team_a=2, difficulty=2),  # team 1 home: difficulty 2
        # team 1 away here; make_fixture sets team_a_difficulty = 6 - difficulty, so
        # difficulty=2 gives team 1 (away) a difficulty of 4.
        make_fixture(2, event=11, team_h=3, team_a=1, difficulty=2),
    ]
    avg_diff, home_frac, count = compute_fixture_window(team_id=1, fixtures=fixtures, from_event=10, num_gameweeks=2)
    assert count == 2
    assert avg_diff == 3.0  # (2 + 4) / 2
    assert home_frac == 0.5


def test_fixture_window_blank_gameweek_returns_zero_fixtures():
    fixtures = [make_fixture(1, event=10, team_h=1, team_a=2)]
    _avg, _home, count = compute_fixture_window(team_id=5, fixtures=fixtures, from_event=10, num_gameweeks=1)
    assert count == 0


def test_minutes_reliability_high_for_nailed_starter():
    p = make_player(1, minutes=900, starts=10)  # 90 min/start, played every game
    assert compute_minutes_reliability(p) > 0.9


def test_minutes_reliability_low_for_fringe_player():
    p = make_player(1, minutes=90, starts=1)
    assert compute_minutes_reliability(p) < compute_minutes_reliability(make_player(2, minutes=900, starts=10))


def test_small_sample_form_is_shrunk_towards_points_per_game():
    team = make_team(1)
    fixtures = [make_fixture(1, event=10, team_h=1, team_a=2)]
    # Explosive one-game form (10) but season points_per_game much lower (2.0), few starts.
    p = make_player(1, form=10.0, points_per_game=2.0, starts=1, minutes=90)
    features = compute_player_features(p, team, fixtures, from_event=10)
    assert features.blended_form < 10.0
    assert any("shrunk" in note for note in features.data_notes)


def test_availability_risk_flows_through():
    team = make_team(1)
    fixtures = [make_fixture(1, event=10, team_h=1, team_a=2)]
    injured = make_player(1, status="i", chance_of_playing_next_round=0)
    features = compute_player_features(injured, team, fixtures, from_event=10)
    assert features.availability_risk == 1.0
