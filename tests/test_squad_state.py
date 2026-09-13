from __future__ import annotations

import pytest

from fpl_automate.squad.state import estimate_free_transfers, find_relevant_events
from fpl_automate.validation.checks import DataValidationError
from tests.conftest import make_gameweek


def test_find_relevant_events_picks_last_finished_and_next():
    events = [
        make_gameweek(1, finished=True),
        make_gameweek(2, finished=True),
        make_gameweek(3, is_current=True),
        make_gameweek(4, is_next=True),
        make_gameweek(5),
    ]
    last, nxt = find_relevant_events(events)
    assert last == 3
    assert nxt == 4


def test_find_relevant_events_raises_without_next():
    events = [make_gameweek(1, finished=True)]
    with pytest.raises(DataValidationError):
        find_relevant_events(events)


def _history(entries):
    return [{"event": e, "event_transfers": t, "event_transfers_cost": c} for e, t, c in entries]


def test_free_transfers_accumulate_when_unused():
    # GW2, GW3 both unused -> should have banked up to 3 by GW4 (1 -> 2 -> 3)
    history = _history([(2, 0, 0), (3, 0, 0)])
    balance, warnings = estimate_free_transfers(history, chips=[], up_to_event=3)
    assert balance == 3
    assert warnings == []


def test_free_transfer_used_resets_to_one_next_week():
    history = _history([(2, 1, 0)])  # used the 1 free transfer, no hit
    balance, _ = estimate_free_transfers(history, chips=[], up_to_event=2)
    assert balance == 1


def test_taking_a_hit_resets_to_one():
    history = _history([(2, 2, 4)])  # 1 free + 1 paid (-4)
    balance, _ = estimate_free_transfers(history, chips=[], up_to_event=2)
    assert balance == 1


def test_balance_caps_at_five():
    history = _history([(e, 0, 0) for e in range(2, 12)])
    balance, _ = estimate_free_transfers(history, chips=[], up_to_event=11)
    assert balance == 5


def test_wildcard_week_leaves_balance_untouched():
    # GW2 unused (-> balance 2 going into GW3), GW3 is a wildcard with many transfers (should not
    # affect the ledger), so balance going into GW4 should still be 2 -> then +1 for GW3 = wait:
    # wildcard weeks are skipped entirely, so the balance from GW2 (2) carries straight to GW4.
    history = _history([(2, 0, 0), (3, 11, 0)])
    chips = [{"name": "wildcard", "event": 3}]
    balance, _ = estimate_free_transfers(history, chips=chips, up_to_event=3)
    assert balance == 2


def test_missing_history_defaults_conservatively():
    balance, warnings = estimate_free_transfers([], chips=[], up_to_event=5)
    assert balance == 1
    assert warnings
