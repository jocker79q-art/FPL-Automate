from __future__ import annotations

import pytest
from pydantic import ValidationError

from fpl_automate.config import ConfigError, Settings


def test_missing_team_id_raises(monkeypatch):
    monkeypatch.delenv("FPL_TEAM_ID", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_valid_minimal_settings():
    s = Settings(_env_file=None, fpl_team_id=9242093)  # type: ignore[call-arg]
    assert s.fpl_team_id == 9242093
    assert s.enable_auto_execution is False
    assert s.emergency_stop is False


def test_email_enabled_without_credentials_raises():
    with pytest.raises(ConfigError):
        Settings(  # type: ignore[call-arg]
            _env_file=None,
            fpl_team_id=9242093,
            email_notifications_enabled=True,
        )


def test_email_enabled_with_credentials_ok():
    s = Settings(  # type: ignore[call-arg]
        _env_file=None,
        fpl_team_id=9242093,
        email_notifications_enabled=True,
        smtp_username="user",
        smtp_password="pass",
        email_from="a@example.com",
        email_to="b@example.com",
    )
    assert s.email_notifications_enabled is True


def test_negative_team_id_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, fpl_team_id=-1)  # type: ignore[call-arg]


def test_blank_mini_league_id_treated_as_none():
    s = Settings(_env_file=None, fpl_team_id=9242093, fpl_mini_league_id="")  # type: ignore[call-arg]
    assert s.fpl_mini_league_id is None


def test_emergency_stop_blocks():
    s = Settings(_env_file=None, fpl_team_id=9242093, emergency_stop=True)  # type: ignore[call-arg]
    with pytest.raises(ConfigError):
        s.require_not_stopped()
