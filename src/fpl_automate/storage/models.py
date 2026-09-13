"""Typed domain models shared across ingestion, features, projections, and reporting.

These are deliberately independent of both the raw FPL JSON shape and the
SQLite schema: `data/fpl_client.py` + parsing functions translate raw API
responses into these; `storage/db.py` translates these to/from SQLite rows.
"""
from __future__ import annotations

from datetime import datetime
from enum import IntEnum

from pydantic import BaseModel, Field


class Position(IntEnum):
    GOALKEEPER = 1
    DEFENDER = 2
    MIDFIELDER = 3
    FORWARD = 4

    @property
    def short(self) -> str:
        return {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[self.value]


class AvailabilityStatus(BaseModel):
    """FPL's own availability signal for a player."""

    status: str  # "a" available, "d" doubtful, "i" injured, "s" suspended, "u" unavailable, "n" not in squad
    chance_of_playing_next_round: int | None = None  # 0/25/50/75/100, None = presumed 100
    news: str = ""
    news_added: datetime | None = None

    @property
    def is_available(self) -> bool:
        return self.status == "a" and (
            self.chance_of_playing_next_round is None or self.chance_of_playing_next_round >= 75
        )

    @property
    def rotation_or_injury_risk(self) -> float:
        """0.0 (nailed) to 1.0 (essentially will not play)."""
        if self.status in ("i", "s", "u"):
            return 1.0
        if self.chance_of_playing_next_round is not None:
            return max(0.0, min(1.0, 1.0 - self.chance_of_playing_next_round / 100))
        return 0.0


class Team(BaseModel):
    id: int
    name: str
    short_name: str
    strength_overall_home: int
    strength_overall_away: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int


class Player(BaseModel):
    id: int
    web_name: str
    full_name: str
    team_id: int
    position: Position
    now_cost_tenths: int  # FPL prices are in tenths of a million (e.g. 105 = £10.5m)
    selected_by_percent: float
    availability: AvailabilityStatus
    form: float
    points_per_game: float
    total_points: int
    minutes: int
    starts: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    bonus: int
    bps: int
    yellow_cards: int
    red_cards: int
    saves: int
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    ep_this: float | None = None
    ep_next: float | None = None
    transfers_in_event: int = 0
    transfers_out_event: int = 0
    penalties_order: int | None = None
    direct_freekicks_order: int | None = None
    corners_and_indirect_freekicks_order: int | None = None

    @property
    def price(self) -> float:
        return self.now_cost_tenths / 10

    @property
    def minutes_per_start(self) -> float:
        return self.minutes / self.starts if self.starts > 0 else 0.0


class Fixture(BaseModel):
    id: int
    event: int | None
    team_h: int
    team_a: int
    team_h_difficulty: int
    team_a_difficulty: int
    kickoff_time: datetime | None
    finished: bool
    team_h_score: int | None = None
    team_a_score: int | None = None


class Gameweek(BaseModel):
    id: int
    name: str
    deadline_time: datetime
    finished: bool
    is_current: bool
    is_next: bool
    average_entry_score: int | None = None


class SquadPick(BaseModel):
    element_id: int
    squad_position: int  # 1-15, FPL's own ordering (1-11 starters incl. GK, 12-15 bench)
    multiplier: int  # 0 (benched), 1, 2 (captain), 3 (triple captain)
    is_captain: bool
    is_vice_captain: bool


class ChipPlay(BaseModel):
    name: str
    event: int


class SquadState(BaseModel):
    """Resolved current-state snapshot for one manager's team."""

    team_id: int
    as_of_event: int  # last completed gameweek this state reflects
    picks: list[SquadPick]
    bank_tenths: int
    squad_value_tenths: int
    free_transfers_available: int
    chips_used: list[ChipPlay]
    wildcard_available: bool
    free_hit_available: bool
    bench_boost_available: bool
    triple_captain_available: bool

    @property
    def bank(self) -> float:
        return self.bank_tenths / 10

    @property
    def squad_value(self) -> float:
        return self.squad_value_tenths / 10


class PlayerProjection(BaseModel):
    """Output of the expected-points model for one player over one horizon."""

    player_id: int
    gameweek: int
    expected_points: float
    floor_points: float
    ceiling_points: float
    confidence: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)
    rationale: str = ""
