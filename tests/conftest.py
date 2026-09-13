from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fpl_automate.storage.models import (
    AvailabilityStatus,
    Fixture,
    Gameweek,
    Player,
    Position,
    Team,
)


def make_team(team_id: int = 1, name: str = "Test Town") -> Team:
    return Team(
        id=team_id,
        name=name,
        short_name=name[:3].upper(),
        strength_overall_home=1100,
        strength_overall_away=1100,
        strength_attack_home=1100,
        strength_attack_away=1100,
        strength_defence_home=1100,
        strength_defence_away=1100,
    )


def make_player(
    player_id: int,
    team_id: int = 1,
    position: Position = Position.MIDFIELDER,
    now_cost_tenths: int = 80,
    status: str = "a",
    chance_of_playing_next_round: int | None = None,
    form: float = 5.0,
    points_per_game: float = 4.5,
    total_points: int = 45,
    minutes: int = 900,
    starts: int = 10,
    **overrides,
) -> Player:
    base = {
        "id": player_id,
        "web_name": f"Player{player_id}",
        "full_name": f"Player {player_id}",
        "team_id": team_id,
        "position": position,
        "now_cost_tenths": now_cost_tenths,
        "selected_by_percent": 10.0,
        "availability": AvailabilityStatus(
            status=status, chance_of_playing_next_round=chance_of_playing_next_round, news=""
        ),
        "form": form,
        "points_per_game": points_per_game,
        "total_points": total_points,
        "minutes": minutes,
        "starts": starts,
        "goals_scored": 3,
        "assists": 2,
        "clean_sheets": 3,
        "goals_conceded": 10,
        "bonus": 5,
        "bps": 300,
        "yellow_cards": 1,
        "red_cards": 0,
        "saves": 0,
        "expected_goals": 2.5,
        "expected_assists": 1.8,
        "expected_goal_involvements": 4.3,
        "expected_goals_conceded": 9.0,
    }
    base.update(overrides)
    return Player(**base)


def make_fixture(
    fixture_id: int, event: int, team_h: int, team_a: int, difficulty: int = 3, finished: bool = False
) -> Fixture:
    return Fixture(
        id=fixture_id,
        event=event,
        team_h=team_h,
        team_a=team_a,
        team_h_difficulty=difficulty,
        team_a_difficulty=6 - difficulty,
        kickoff_time=datetime.now(UTC) + timedelta(days=event),
        finished=finished,
    )


def make_gameweek(event_id: int, is_current: bool = False, is_next: bool = False, finished: bool = False) -> Gameweek:
    return Gameweek(
        id=event_id,
        name=f"Gameweek {event_id}",
        deadline_time=datetime.now(UTC) + timedelta(days=event_id),
        finished=finished,
        is_current=is_current,
        is_next=is_next,
        average_entry_score=50,
    )


def make_raw_team(team_id: int) -> dict:
    return {
        "id": team_id,
        "name": f"Team {team_id}",
        "short_name": f"T{team_id:02d}",
        "strength_overall_home": 1100,
        "strength_overall_away": 1100,
        "strength_attack_home": 1100,
        "strength_attack_away": 1100,
        "strength_defence_home": 1100,
        "strength_defence_away": 1100,
    }


def make_raw_player(player_id: int, team_id: int) -> dict:
    return {
        "id": player_id,
        "web_name": f"Player{player_id}",
        "first_name": "First",
        "second_name": f"Last{player_id}",
        "team": team_id,
        "element_type": (player_id % 4) + 1,
        "now_cost": 50 + (player_id % 100),
        "selected_by_percent": "5.0",
        "status": "a",
        "news": "",
        "news_added": None,
        "chance_of_playing_next_round": None,
        "form": "3.0",
        "points_per_game": "3.0",
        "total_points": 30,
        "minutes": 900,
        "starts": 10,
        "goals_scored": 1,
        "assists": 1,
        "clean_sheets": 1,
        "goals_conceded": 10,
        "bonus": 2,
        "bps": 200,
        "yellow_cards": 0,
        "red_cards": 0,
        "saves": 0,
        "expected_goals": "1.0",
        "expected_assists": "1.0",
        "expected_goal_involvements": "2.0",
        "expected_goals_conceded": "8.0",
        "ep_this": "3.0",
        "ep_next": "3.0",
        "transfers_in_event": 0,
        "transfers_out_event": 0,
        "penalties_order": None,
        "direct_freekicks_order": None,
        "corners_and_indirect_freekicks_order": None,
    }


@pytest.fixture
def bootstrap_static_payload() -> dict:
    teams = [make_raw_team(i) for i in range(1, 21)]
    players = [make_raw_player(i, ((i - 1) % 20) + 1) for i in range(1, 451)]
    events = [
        {
            "id": i,
            "name": f"Gameweek {i}",
            "deadline_time": (datetime.now(UTC) + timedelta(days=i)).isoformat().replace("+00:00", "Z"),
            "finished": i < 5,
            "is_current": i == 5,
            "is_next": i == 6,
            "average_entry_score": 50,
        }
        for i in range(1, 39)
    ]
    element_types = [{"id": i} for i in range(1, 5)]
    return {"teams": teams, "elements": players, "events": events, "element_types": element_types}
