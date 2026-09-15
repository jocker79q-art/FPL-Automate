"""SQLite persistence layer (stdlib sqlite3, no ORM).

The schema favours simplicity over normalisation: most tables store a
`raw_json` column alongside a handful of indexed columns used for
filtering. This keeps the ingestion layer resilient to the FPL API adding
fields we don't yet model, while still letting the rest of the app query
efficiently on the columns it actually needs.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    status TEXT NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL,
    updated_at REAL NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gameweeks (
    event_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    deadline_time TEXT NOT NULL,
    finished INTEGER NOT NULL,
    is_current INTEGER NOT NULL,
    is_next INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    team_id INTEGER NOT NULL,
    web_name TEXT NOT NULL,
    position INTEGER NOT NULL,
    now_cost_tenths INTEGER NOT NULL,
    status TEXT NOT NULL,
    selected_by_percent REAL NOT NULL,
    updated_at REAL NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id INTEGER PRIMARY KEY,
    event INTEGER,
    team_h INTEGER NOT NULL,
    team_a INTEGER NOT NULL,
    finished INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS squad_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    as_of_event INTEGER NOT NULL,
    fetched_at REAL NOT NULL,
    state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event INTEGER NOT NULL,
    created_at REAL NOT NULL,
    strategy TEXT NOT NULL,
    recommendation_json TEXT NOT NULL,
    approved INTEGER,
    actual_points INTEGER,
    outcome_notes TEXT
);

CREATE TABLE IF NOT EXISTS projection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    horizon_gameweeks INTEGER NOT NULL,
    expected_points REAL NOT NULL,
    floor_points REAL,
    ceiling_points REAL,
    confidence REAL,
    logged_at REAL NOT NULL,
    actual_points INTEGER
);

CREATE INDEX IF NOT EXISTS idx_decisions_event ON decisions(event);
CREATE INDEX IF NOT EXISTS idx_squad_snapshots_team_event
    ON squad_snapshots(team_id, as_of_event);
CREATE INDEX IF NOT EXISTS idx_projection_log_event ON projection_log(event);
CREATE INDEX IF NOT EXISTS idx_projection_log_unresolved
    ON projection_log(event) WHERE actual_points IS NULL;
"""


class Database:
    def __init__(self, sqlite_path: Path) -> None:
        self._path = sqlite_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # -- fetch log ----------------------------------------------------

    def log_fetch(self, endpoint: str, status: str, note: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO fetch_log (endpoint, fetched_at, status, note) VALUES (?, ?, ?, ?)",
                (endpoint, time.time(), status, note),
            )

    def last_successful_fetch(self, endpoint: str) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT fetched_at FROM fetch_log WHERE endpoint = ? AND status = 'ok' "
                "ORDER BY fetched_at DESC LIMIT 1",
                (endpoint,),
            ).fetchone()
            return row["fetched_at"] if row else None

    # -- reference data snapshots --------------------------------------

    def upsert_teams(self, teams: list[dict[str, Any]]) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO teams (team_id, name, short_name, updated_at, raw_json) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(team_id) DO UPDATE SET name=excluded.name, "
                "short_name=excluded.short_name, updated_at=excluded.updated_at, "
                "raw_json=excluded.raw_json",
                [(t["id"], t["name"], t["short_name"], now, json.dumps(t)) for t in teams],
            )

    def upsert_gameweeks(self, events: list[dict[str, Any]]) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO gameweeks (event_id, name, deadline_time, finished, is_current, "
                "is_next, updated_at, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(event_id) DO UPDATE SET name=excluded.name, "
                "deadline_time=excluded.deadline_time, finished=excluded.finished, "
                "is_current=excluded.is_current, is_next=excluded.is_next, "
                "updated_at=excluded.updated_at, raw_json=excluded.raw_json",
                [
                    (
                        e["id"],
                        e["name"],
                        e["deadline_time"],
                        int(e["finished"]),
                        int(e["is_current"]),
                        int(e["is_next"]),
                        now,
                        json.dumps(e),
                    )
                    for e in events
                ],
            )

    def upsert_players(self, players: list[dict[str, Any]]) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO players (player_id, team_id, web_name, position, now_cost_tenths, "
                "status, selected_by_percent, updated_at, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(player_id) DO UPDATE SET team_id=excluded.team_id, "
                "web_name=excluded.web_name, position=excluded.position, "
                "now_cost_tenths=excluded.now_cost_tenths, status=excluded.status, "
                "selected_by_percent=excluded.selected_by_percent, updated_at=excluded.updated_at, "
                "raw_json=excluded.raw_json",
                [
                    (
                        p["id"],
                        p["team"],
                        p["web_name"],
                        p["element_type"],
                        p["now_cost"],
                        p["status"],
                        float(p["selected_by_percent"]),
                        now,
                        json.dumps(p),
                    )
                    for p in players
                ],
            )

    def upsert_fixtures(self, fixtures: list[dict[str, Any]]) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO fixtures (fixture_id, event, team_h, team_a, finished, updated_at, "
                "raw_json) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(fixture_id) DO UPDATE SET event=excluded.event, "
                "team_h=excluded.team_h, team_a=excluded.team_a, finished=excluded.finished, "
                "updated_at=excluded.updated_at, raw_json=excluded.raw_json",
                [
                    (f["id"], f.get("event"), f["team_h"], f["team_a"], int(f["finished"]), now, json.dumps(f))
                    for f in fixtures
                ],
            )

    def get_all_raw(self, table: str) -> list[dict[str, Any]]:
        assert table in {"teams", "gameweeks", "players", "fixtures"}, f"unknown table {table}"
        with self._connect() as conn:
            rows = conn.execute(f"SELECT raw_json FROM {table}").fetchall()
            return [json.loads(r["raw_json"]) for r in rows]

    # -- squad snapshots -------------------------------------------------

    def save_squad_snapshot(self, team_id: int, as_of_event: int, state_json: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO squad_snapshots (team_id, as_of_event, fetched_at, state_json) "
                "VALUES (?, ?, ?, ?)",
                (team_id, as_of_event, time.time(), state_json),
            )

    def latest_squad_snapshot(self, team_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM squad_snapshots WHERE team_id = ? "
                "ORDER BY as_of_event DESC, fetched_at DESC LIMIT 1",
                (team_id,),
            ).fetchone()
            return json.loads(row["state_json"]) if row else None

    # -- decisions / outcome tracking -------------------------------------

    def save_decision(self, event: int, strategy: str, recommendation_json: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO decisions (event, created_at, strategy, recommendation_json, "
                "approved, actual_points, outcome_notes) VALUES (?, ?, ?, ?, NULL, NULL, NULL)",
                (event, time.time(), strategy, recommendation_json),
            )
            assert cur.lastrowid is not None  # guaranteed after a successful INSERT
            return cur.lastrowid

    def record_outcome(self, decision_id: int, approved: bool, actual_points: int | None, notes: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE decisions SET approved = ?, actual_points = ?, outcome_notes = ? WHERE id = ?",
                (int(approved), actual_points, notes, decision_id),
            )

    def record_decision_actual_points(self, decision_id: int, actual_points: int) -> None:
        """Fills in just `actual_points` for a past decision, leaving
        `approved` untouched -- unlike `record_outcome`, this is for
        *automatic* reconciliation (see `workflow.reconcile_outcomes`),
        which has no way to know whether the user actually followed the
        recommendation and must never guess/overwrite that field."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE decisions SET actual_points = ? WHERE id = ?", (actual_points, decision_id)
            )

    def unresolved_decisions(self, up_to_event: int) -> list[dict[str, Any]]:
        """Decisions for a gameweek that has since finished but whose
        actual_points is still NULL -- i.e. ready to reconcile."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE actual_points IS NULL AND event <= ? "
                "ORDER BY event",
                (up_to_event,),
            ).fetchall()
            return [dict(r) for r in rows]

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # -- per-player projection accuracy tracking --------------------------
    #
    # Distinct from `decisions` (whole-squad recommendation + outcome):
    # this tracks the projection model's actual accuracy, player by
    # player and gameweek by gameweek, in production -- the live
    # complement to `ml/backtest.py`'s offline walk-forward backtest.
    # Only the owned squad is logged each run (not the full ~700-player
    # pool) to keep this lightweight and avoid unbounded DB growth.

    def log_projections(
        self, event: int, model_name: str, horizon_gameweeks: int, rows: list[dict[str, Any]]
    ) -> None:
        """`rows`: [{player_id, expected_points, floor_points, ceiling_points, confidence}, ...]."""
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO projection_log (event, player_id, model_name, horizon_gameweeks, "
                "expected_points, floor_points, ceiling_points, confidence, logged_at, actual_points) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                [
                    (
                        event,
                        r["player_id"],
                        model_name,
                        horizon_gameweeks,
                        r["expected_points"],
                        r.get("floor_points"),
                        r.get("ceiling_points"),
                        r.get("confidence"),
                        now,
                    )
                    for r in rows
                ],
            )

    def unresolved_projections(self, up_to_event: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM projection_log WHERE actual_points IS NULL AND event <= ? "
                "ORDER BY event",
                (up_to_event,),
            ).fetchall()
            return [dict(r) for r in rows]

    def record_projection_actual_points(self, projection_id: int, actual_points: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE projection_log SET actual_points = ? WHERE id = ?", (actual_points, projection_id)
            )

    def projection_performance(self, model_name: str | None = None) -> list[dict[str, Any]]:
        """Resolved (actual_points filled in) rows, most recent first --
        callers compute MAE/bias themselves (see `cli.show_model_performance_cmd`)
        since aggregating in SQL would obscure the per-row detail a user
        might want to inspect."""
        with self._connect() as conn:
            if model_name:
                rows = conn.execute(
                    "SELECT * FROM projection_log WHERE actual_points IS NOT NULL AND model_name = ? "
                    "ORDER BY event DESC",
                    (model_name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM projection_log WHERE actual_points IS NOT NULL ORDER BY event DESC"
                ).fetchall()
            return [dict(r) for r in rows]
