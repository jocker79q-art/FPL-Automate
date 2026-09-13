"""Translate raw FPL API JSON (mostly strings-for-numbers) into typed domain models."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fpl_automate.storage.models import (
    AvailabilityStatus,
    Fixture,
    Gameweek,
    Player,
    Position,
    Team,
)


def _f(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def parse_team(raw: dict[str, Any]) -> Team:
    return Team(
        id=raw["id"],
        name=raw["name"],
        short_name=raw["short_name"],
        strength_overall_home=raw["strength_overall_home"],
        strength_overall_away=raw["strength_overall_away"],
        strength_attack_home=raw["strength_attack_home"],
        strength_attack_away=raw["strength_attack_away"],
        strength_defence_home=raw["strength_defence_home"],
        strength_defence_away=raw["strength_defence_away"],
    )


def parse_player(raw: dict[str, Any]) -> Player:
    news_added = None
    if raw.get("news_added"):
        try:
            news_added = datetime.fromisoformat(raw["news_added"])
        except ValueError:
            news_added = None

    availability = AvailabilityStatus(
        status=raw["status"],
        chance_of_playing_next_round=raw.get("chance_of_playing_next_round"),
        news=raw.get("news", "") or "",
        news_added=news_added,
    )

    return Player(
        id=raw["id"],
        web_name=raw["web_name"],
        full_name=f"{raw.get('first_name', '')} {raw.get('second_name', '')}".strip(),
        team_id=raw["team"],
        position=Position(raw["element_type"]),
        now_cost_tenths=raw["now_cost"],
        selected_by_percent=_f(raw.get("selected_by_percent")),
        availability=availability,
        form=_f(raw.get("form")),
        points_per_game=_f(raw.get("points_per_game")),
        total_points=raw.get("total_points", 0),
        minutes=raw.get("minutes", 0),
        starts=raw.get("starts", 0),
        goals_scored=raw.get("goals_scored", 0),
        assists=raw.get("assists", 0),
        clean_sheets=raw.get("clean_sheets", 0),
        goals_conceded=raw.get("goals_conceded", 0),
        bonus=raw.get("bonus", 0),
        bps=raw.get("bps", 0),
        yellow_cards=raw.get("yellow_cards", 0),
        red_cards=raw.get("red_cards", 0),
        saves=raw.get("saves", 0),
        expected_goals=_f(raw.get("expected_goals")),
        expected_assists=_f(raw.get("expected_assists")),
        expected_goal_involvements=_f(raw.get("expected_goal_involvements")),
        expected_goals_conceded=_f(raw.get("expected_goals_conceded")),
        ep_this=_f(raw.get("ep_this")) if raw.get("ep_this") not in (None, "") else None,
        ep_next=_f(raw.get("ep_next")) if raw.get("ep_next") not in (None, "") else None,
        transfers_in_event=raw.get("transfers_in_event", 0),
        transfers_out_event=raw.get("transfers_out_event", 0),
        penalties_order=raw.get("penalties_order"),
        direct_freekicks_order=raw.get("direct_freekicks_order"),
        corners_and_indirect_freekicks_order=raw.get("corners_and_indirect_freekicks_order"),
    )


def parse_fixture(raw: dict[str, Any]) -> Fixture:
    kickoff = None
    if raw.get("kickoff_time"):
        try:
            kickoff = datetime.fromisoformat(raw["kickoff_time"])
        except ValueError:
            kickoff = None
    return Fixture(
        id=raw["id"],
        event=raw.get("event"),
        team_h=raw["team_h"],
        team_a=raw["team_a"],
        team_h_difficulty=raw.get("team_h_difficulty", 3),
        team_a_difficulty=raw.get("team_a_difficulty", 3),
        kickoff_time=kickoff,
        finished=bool(raw.get("finished", False)),
        team_h_score=raw.get("team_h_score"),
        team_a_score=raw.get("team_a_score"),
    )


def parse_gameweek(raw: dict[str, Any]) -> Gameweek:
    return Gameweek(
        id=raw["id"],
        name=raw["name"],
        deadline_time=datetime.fromisoformat(raw["deadline_time"]),
        finished=bool(raw["finished"]),
        is_current=bool(raw["is_current"]),
        is_next=bool(raw["is_next"]),
        average_entry_score=raw.get("average_entry_score"),
    )
