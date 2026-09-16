"""Central configuration, loaded from environment variables / .env.

No secret is ever given a default value here: anything sensitive (SMTP
credentials) must come from the environment, and misconfiguration fails
loudly at startup rather than silently degrading.
"""
from __future__ import annotations

from functools import cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when configuration is missing, contradictory, or unsafe."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Identity
    fpl_team_id: int = Field(..., description="Your FPL entry/team ID")
    fpl_mini_league_id: int | None = None

    # Safety switches
    emergency_stop: bool = False
    enable_auto_execution: bool = False
    max_transfer_risk: int = 4

    # Projection model (docs/ROADMAP.md Phase 2): "ml" uses the trained
    # two-stage hurdle model (ml/backtest.py's walk-forward backtest is
    # what justifies this default -- see reports/model_backtest.md),
    # falling back to the hand-coded baseline per player whenever the ML
    # model can't cover them (no trained bundle yet, a brand-new team,
    # etc.) or no trained model exists at all. "baseline" forces the
    # hand-coded model only, e.g. to reproduce pre-Phase-2 behaviour or
    # audit the two side by side. Never a *silent* switch either way --
    # every projection's rationale states which model produced it.
    projection_model: Literal["baseline", "ml"] = "ml"

    # Mean-variance risk aversion (optimization/lineup.py's "risk_adjusted"
    # strategy, see risk/portfolio.py) -- 0 is equivalent to plain
    # expected-points maximisation ("balanced"); larger values increasingly
    # favour lower-variance players and captaincy picks over higher-mean
    # but more volatile ones. Only changes decisions actually made under
    # strategy="risk_adjusted" -- it's always computed for the report's
    # strategy-comparison table regardless of the chosen strategy.
    risk_aversion: float = 1.0

    # Storage
    database_url: str = "sqlite:///data/fpl_automate.db"

    # Networking
    fpl_api_base_url: str = "https://fantasy.premierleague.com/api"
    fpl_http_timeout_seconds: float = 15.0
    fpl_min_request_interval_seconds: float = 1.0
    fpl_max_retries: int = 3

    # Email notifications
    email_notifications_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    email_from: str = ""
    email_to: str = ""

    log_level: str = "INFO"

    @field_validator("fpl_mini_league_id", mode="before")
    @classmethod
    def _blank_league_id_means_none(cls, v: object) -> object:
        # An unset .env value (FPL_MINI_LEAGUE_ID=) arrives as "", which int-parsing
        # would otherwise reject -- treat blank the same as omitted.
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("fpl_team_id")
    @classmethod
    def _positive_team_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("FPL_TEAM_ID must be a positive integer")
        return v

    @field_validator("max_transfer_risk")
    @classmethod
    def _sane_risk(cls, v: int) -> int:
        if v < 0 or v > 16:
            raise ValueError("MAX_TRANSFER_RISK must be between 0 and 16")
        return v

    @field_validator("risk_aversion")
    @classmethod
    def _non_negative_risk_aversion(cls, v: float) -> float:
        if v < 0:
            raise ValueError("RISK_AVERSION must be >= 0 (0 means no risk penalty at all)")
        return v

    @model_validator(mode="after")
    def _validate_email_config(self) -> Settings:
        if self.email_notifications_enabled:
            missing = [
                name
                for name, val in [
                    ("SMTP_USERNAME", self.smtp_username),
                    ("SMTP_PASSWORD", self.smtp_password),
                    ("EMAIL_FROM", self.email_from),
                    ("EMAIL_TO", self.email_to),
                ]
                if not val
            ]
            if missing:
                raise ConfigError(
                    "EMAIL_NOTIFICATIONS_ENABLED=true but missing: " + ", ".join(missing)
                )
        return self

    def require_not_stopped(self) -> None:
        """Call before any non-trivial action; raises if the kill switch is on."""
        if self.emergency_stop:
            raise ConfigError(
                "EMERGENCY_STOP is set to true. Refusing to run. "
                "Set EMERGENCY_STOP=false in .env to resume."
            )


@cache
def get_settings(env_file: str | None = None) -> Settings:
    """Loads settings from `env_file` (defaults to ".env" in the current working
    directory). The packaged .exe passes an absolute path next to the executable
    itself, since a double-clicked exe's working directory isn't reliable."""
    try:
        kwargs = {"_env_file": env_file} if env_file is not None else {}
        return Settings(**kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        raise ConfigError(
            "Configuration is invalid or incomplete. Copy .env.example to .env "
            f"and fill in the required values. Details: {exc}"
        ) from exc


def clear_settings_cache() -> None:
    """Call after writing new values to .env (e.g. the GUI's Settings tab saving)
    so the next get_settings() call re-reads the file instead of returning a
    stale cached Settings object."""
    get_settings.cache_clear()
