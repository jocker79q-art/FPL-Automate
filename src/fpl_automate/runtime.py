"""Shared runtime path resolution for every frontend (CLI, GUI) and both
packaged forms (a normal `pip install -e .` and a PyInstaller .exe).

A packaged .exe's working directory isn't reliable (a double-click's cwd
varies by how Windows launched it), so everything -- .env, data/, reports/
-- is resolved relative to the executable's own folder when frozen, and
relative to the repo root otherwise. Keeping this in one place means the
CLI and the GUI can never quietly disagree about where "home" is.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from fpl_automate.config import Settings, get_settings

IS_FROZEN = getattr(sys, "frozen", False)

if IS_FROZEN:
    APP_BASE_DIR = Path(sys.executable).resolve().parent
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_BASE_DIR))
else:
    APP_BASE_DIR = Path(__file__).resolve().parents[2]
    BUNDLE_DIR = APP_BASE_DIR

ENV_FILE_PATH = APP_BASE_DIR / ".env"
ENV_EXAMPLE_PATH = BUNDLE_DIR / ".env.example"
DEFAULT_REPORTS_DIR = APP_BASE_DIR / "reports"
DEFAULT_CACHE_DIR = APP_BASE_DIR / "data" / "cache"


def get_app_settings() -> Settings:
    """The one call every frontend should use instead of `config.get_settings()`
    directly -- it points at the exe-relative (or repo-relative) .env."""
    return get_settings(env_file=str(ENV_FILE_PATH))


def ensure_env_file_exists() -> tuple[bool, bool]:
    """Creates .env from the bundled template if neither it nor a real
    FPL_TEAM_ID environment variable exists yet.

    Returns (created, template_found). `created` is True only when a fresh
    .env was just written -- callers should treat that as "stop here and let
    the user fill it in" for a console-only frontend; a GUI frontend can
    ignore this entirely and just let its Settings tab write .env directly.
    """
    if ENV_FILE_PATH.exists() or os.environ.get("FPL_TEAM_ID"):
        return False, True
    if ENV_EXAMPLE_PATH.exists():
        shutil.copyfile(ENV_EXAMPLE_PATH, ENV_FILE_PATH)
        return True, True
    return True, False
