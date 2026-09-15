"""Download past-season FPL data from the vaastav/Fantasy-Premier-League
public archive (raw.githubusercontent.com) and cache it locally.

This is the training data source for the ML projection model:
gameweek-by-gameweek actual results for every player, several seasons
back. Entirely separate from `data/fpl_client.py`, which only ever talks
to the live, current-season FPL API -- this module never touches that
endpoint and never needs a login.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

HISTORICAL_ARCHIVE_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

# Seasons to pull for model training. Extend as new (completed) seasons
# become available -- never include the season currently being played
# (it's incomplete and would bias/leak into a "final season" merged_gw).
DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# Chronological holdout for backtesting: train on everything before this
# season, evaluate on this season only. Always the most recently
# *completed* season -- bump this (and add it to DEFAULT_SEASONS) once a
# new season finishes.
TEST_SEASON = "2025-26"

# The season currently being played -- deliberately not in DEFAULT_SEASONS
# (it's incomplete, and must never be used for training/backtesting since
# it's what live inference predicts into). vaastav's archive updates this
# season's merged_gw.csv gameweek by gameweek as real results land, which
# is what makes it usable as *live* rolling-feature history (see
# ml/live.py) without needing hundreds of extra FPL API calls.
CURRENT_SEASON = "2026-27"

# path (relative to a season's data/ folder) -> local filename
SEASON_FILES = {
    "players_raw.csv": "players_raw.csv",
    "fixtures.csv": "fixtures.csv",
    "teams.csv": "teams.csv",
    "gws/merged_gw.csv": "merged_gw.csv",
}

REQUEST_TIMEOUT = 30


def _fetch_text(url: str, session: requests.Session) -> str | None:
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    # A handful of older-season files are latin-1, not utf-8.
    resp.encoding = resp.encoding or "utf-8"
    try:
        return resp.content.decode("utf-8")
    except UnicodeDecodeError:
        return resp.content.decode("latin-1")


def fetch_season(
    season: str, historical_dir: Path, session: requests.Session | None = None
) -> dict[str, bool]:
    """Download the known files for one season into `historical_dir/{season}/`.

    Returns {local_filename: was_downloaded} so callers can tell which
    files were missing upstream (not every season has every file).
    """
    session = session or requests.Session()
    season_dir = historical_dir / season
    season_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    for remote_path, local_name in SEASON_FILES.items():
        url = f"{HISTORICAL_ARCHIVE_BASE}/{season}/{remote_path}"
        text = _fetch_text(url, session)
        if text is None:
            logger.warning("season %s: %s not found upstream, skipping", season, remote_path)
            results[local_name] = False
            continue
        (season_dir / local_name).write_text(text, encoding="utf-8")
        results[local_name] = True

    return results


def fetch_all_seasons(
    historical_dir: Path,
    seasons: list[str] | None = None,
    session: requests.Session | None = None,
) -> dict[str, dict[str, bool]]:
    seasons = seasons or DEFAULT_SEASONS
    session = session or requests.Session()
    return {season: fetch_season(season, historical_dir, session) for season in seasons}


def load_merged_gw(historical_dir: Path, seasons: list[str] | None = None) -> pd.DataFrame:
    """Load cached per-gameweek player rows across seasons into one frame.

    Requires fetch_season/fetch_all_seasons to have been run first for
    each season requested.
    """
    seasons = seasons or DEFAULT_SEASONS
    frames = []
    for season in seasons:
        path = historical_dir / season / "merged_gw.csv"
        if not path.exists():
            logger.warning("no cached merged_gw.csv for season %s, skipping", season)
            continue
        df = pd.read_csv(path)
        df["season"] = season
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_teams(historical_dir: Path, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = historical_dir / season / "teams.csv"
        if not path.exists():
            logger.warning("no cached teams.csv for season %s, skipping", season)
            continue
        df = pd.read_csv(path)
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_fixtures(historical_dir: Path, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = historical_dir / season / "fixtures.csv"
        if not path.exists():
            logger.warning("no cached fixtures.csv for season %s, skipping", season)
            continue
        df = pd.read_csv(path)
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
