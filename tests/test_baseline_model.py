from __future__ import annotations

from fpl_automate.features.engineering import compute_player_features
from fpl_automate.projections.baseline_model import (
    ModelInputs,
    apply_captain_multiplier,
    project_player,
)
from tests.conftest import make_fixture, make_player, make_team


def _project(player, team, fixtures, from_event=10, horizon=1):
    features = compute_player_features(player, team, fixtures, from_event=from_event, fixture_window_gameweeks=horizon)
    return project_player(ModelInputs(player=player, features=features, horizon_gameweeks=horizon))


def test_blank_gameweek_gives_zero_points():
    team = make_team(1)
    player = make_player(1, team_id=1)
    fixtures = [make_fixture(1, event=10, team_h=2, team_a=3)]  # doesn't involve team 1
    projection = _project(player, team, fixtures, from_event=10)
    assert projection.expected_points == 0.0
    assert "blank_gameweek" in projection.risk_flags


def test_easier_fixture_scores_higher_than_harder_fixture():
    team = make_team(1)
    player = make_player(1, team_id=1, form=6.0, points_per_game=6.0, starts=10, minutes=900)
    easy_fixtures = [make_fixture(1, event=10, team_h=1, team_a=2, difficulty=1)]
    hard_fixtures = [make_fixture(1, event=10, team_h=1, team_a=2, difficulty=5)]
    easy_proj = _project(player, team, easy_fixtures, from_event=10)
    hard_proj = _project(player, team, hard_fixtures, from_event=10)
    assert easy_proj.expected_points > hard_proj.expected_points


def test_injured_player_projects_lower_than_fit_equivalent():
    team = make_team(1)
    fixtures = [make_fixture(1, event=10, team_h=1, team_a=2)]
    fit = make_player(1, team_id=1, status="a", chance_of_playing_next_round=None)
    injured = make_player(2, team_id=1, status="i", chance_of_playing_next_round=0)
    fit_proj = _project(fit, team, fixtures, from_event=10)
    injured_proj = _project(injured, team, fixtures, from_event=10)
    assert injured_proj.expected_points < fit_proj.expected_points
    assert "injury_or_availability_doubt" in injured_proj.risk_flags


def test_floor_never_exceeds_expected_and_ceiling_never_below():
    team = make_team(1)
    fixtures = [make_fixture(1, event=10, team_h=1, team_a=2)]
    player = make_player(1, team_id=1)
    projection = _project(player, team, fixtures, from_event=10)
    assert projection.floor_points <= projection.expected_points <= projection.ceiling_points


def test_captain_multiplier_doubles_all_three_estimates():
    team = make_team(1)
    fixtures = [make_fixture(1, event=10, team_h=1, team_a=2)]
    player = make_player(1, team_id=1)
    projection = _project(player, team, fixtures, from_event=10)
    captained = apply_captain_multiplier(projection, multiplier=2)
    assert captained.expected_points == round(projection.expected_points * 2, 2)
    assert captained.floor_points == round(projection.floor_points * 2, 2)
    assert captained.ceiling_points == round(projection.ceiling_points * 2, 2)


def test_double_gameweek_flagged_and_scores_higher():
    team = make_team(1)
    player = make_player(1, team_id=1, form=5.0, points_per_game=5.0, starts=10, minutes=900)
    single = [make_fixture(1, event=10, team_h=1, team_a=2)]
    double = [
        make_fixture(1, event=10, team_h=1, team_a=2),
        make_fixture(2, event=10, team_h=3, team_a=1),
    ]
    single_proj = _project(player, team, single, from_event=10)
    double_proj = _project(player, team, double, from_event=10)
    assert "double_gameweek" in double_proj.risk_flags
    assert double_proj.expected_points > single_proj.expected_points
