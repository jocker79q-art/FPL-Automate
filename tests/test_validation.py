from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fpl_automate.validation.checks import (
    DataValidationError,
    validate_bootstrap_static,
    validate_freshness,
    validate_squad_state,
)


def _make_picks(n=15, captain_idx=0, vice_idx=1):
    picks = []
    for i in range(n):
        picks.append(
            {
                "element": i + 1,
                "position": i + 1,
                "multiplier": 1,
                "is_captain": i == captain_idx,
                "is_vice_captain": i == vice_idx,
            }
        )
    return picks


def test_valid_squad_state_passes():
    validate_squad_state(_make_picks(), bank_tenths=10)


def test_wrong_squad_size_fails():
    with pytest.raises(DataValidationError, match="15"):
        validate_squad_state(_make_picks(n=14), bank_tenths=10)


def test_negative_bank_fails():
    with pytest.raises(DataValidationError, match="Negative"):
        validate_squad_state(_make_picks(), bank_tenths=-5)


def test_duplicate_players_fail():
    picks = _make_picks()
    picks[1]["element"] = picks[0]["element"]
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_squad_state(picks, bank_tenths=10)


def test_captain_and_vice_must_differ():
    picks = _make_picks(captain_idx=0, vice_idx=0)
    with pytest.raises(DataValidationError, match="cannot be the same"):
        validate_squad_state(picks, bank_tenths=10)


def test_freshness_rejects_past_deadline():
    past = datetime.now(UTC) - timedelta(hours=1)
    with pytest.raises(DataValidationError, match="already passed"):
        validate_freshness(past, timedelta(days=10))


def test_freshness_accepts_future_deadline():
    future = datetime.now(UTC) + timedelta(hours=1)
    validate_freshness(future, timedelta(days=10))  # should not raise


def test_valid_payload_passes(bootstrap_static_payload):
    validate_bootstrap_static(bootstrap_static_payload)  # should not raise


def test_wrong_team_count_fails(bootstrap_static_payload):
    bootstrap_static_payload["teams"] = bootstrap_static_payload["teams"][:19]
    with pytest.raises(DataValidationError, match="teams"):
        validate_bootstrap_static(bootstrap_static_payload)


def test_too_few_players_fails(bootstrap_static_payload):
    bootstrap_static_payload["elements"] = bootstrap_static_payload["elements"][:10]
    with pytest.raises(DataValidationError, match="players"):
        validate_bootstrap_static(bootstrap_static_payload)


def test_missing_key_fails(bootstrap_static_payload):
    del bootstrap_static_payload["events"]
    with pytest.raises(DataValidationError, match="events"):
        validate_bootstrap_static(bootstrap_static_payload)


def test_widespread_bad_prices_fail(bootstrap_static_payload):
    for p in bootstrap_static_payload["elements"]:
        p["now_cost"] = -5
    with pytest.raises(DataValidationError):
        validate_bootstrap_static(bootstrap_static_payload)


def test_handful_of_bad_records_tolerated(bootstrap_static_payload):
    bootstrap_static_payload["elements"][0]["now_cost"] = -5
    validate_bootstrap_static(bootstrap_static_payload)  # should not raise (below threshold)
