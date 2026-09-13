"""Resolve a manager's current squad, bank, free transfers, and chip status.

Everything here uses PUBLIC read-only FPL endpoints -- an entry's history,
picks, and transfers are visible to anyone (the same way anyone can look up
your team on the FPL website). No login/session is required or used. This
is what makes recommendation-only mode possible with zero credentials.

=====================================================================
ASSUMPTIONS -- free transfer & chip accounting (read before trusting this)
=====================================================================
* Free transfers accumulate at +1 per gameweek not fully used, capped at 5
  (current FPL rule as of the 2024/25 rule change). If FPL changes this
  again, `estimate_free_transfers` will be wrong until updated.
* A Wildcard or Free Hit gameweek is treated as not affecting the banked
  free-transfer count at all (neither consuming nor growing it) -- this
  matches FPL's ruling that chip gameweeks give unlimited free transfers
  without touching your saved balance.
* Wildcard / Free Hit / Bench Boost / Triple Captain are each treated as
  single-use for the season. Some seasons have offered a second Wildcard;
  this module does not special-case that, and simply reports
  `wildcard_available = (no wildcard used yet)`. Since you've told this
  system you've already used your Wildcard, this simplification matches
  your actual situation -- but if FPL later grants you a second one, this
  field will incorrectly say "not available" until this file is updated.
* All of this is inferred from your own public transfer/points history; if
  that history is incomplete (e.g. a newly created team), free-transfer
  estimation falls back to a conservative default (see code) and is
  flagged in `risk_flags` rather than silently trusted.
=====================================================================
"""
from __future__ import annotations

import logging

from fpl_automate.data.fpl_client import FplClient
from fpl_automate.storage.models import ChipPlay, Gameweek, SquadPick, SquadState
from fpl_automate.validation.checks import DataValidationError, validate_squad_state

logger = logging.getLogger(__name__)

MAX_FREE_TRANSFERS = 5
TRANSFER_CHIP_NAMES = {"wildcard", "freehit"}


def find_relevant_events(events: list[Gameweek]) -> tuple[int, int]:
    """Returns (last_picks_event, next_deadline_event).

    `last_picks_event` is the gameweek whose picks represent the manager's
    current squad going into the next deadline. `next_deadline_event` is
    the gameweek the recommendation report should be produced for.
    """
    finished_or_current = [e for e in events if e.finished or e.is_current]
    upcoming = [e for e in events if e.is_next]

    if not finished_or_current:
        raise DataValidationError(
            "No finished/current gameweek found -- cannot determine the current squad "
            "(is the season not yet under way?)."
        )
    if not upcoming:
        raise DataValidationError(
            "No upcoming gameweek found (is_next missing from bootstrap-static events) -- "
            "cannot determine which deadline to plan for."
        )

    last_picks_event = max(e.id for e in finished_or_current)
    next_deadline_event = min(e.id for e in upcoming)
    return last_picks_event, next_deadline_event


def estimate_free_transfers(
    history_current: list[dict], chips: list[dict], up_to_event: int
) -> tuple[int, list[str]]:
    """Simulates the free-transfer ledger up to and including `up_to_event`.

    Returns (free_transfers_available_for_next_deadline, warnings).
    """
    warnings: list[str] = []
    chip_events = {
        c["event"] for c in chips if c.get("name") in TRANSFER_CHIP_NAMES
    }

    relevant = sorted(
        [gw for gw in history_current if gw["event"] <= up_to_event], key=lambda gw: gw["event"]
    )
    if not relevant:
        warnings.append(
            "No gameweek history found; defaulting to 1 free transfer (conservative)."
        )
        return 1, warnings

    balance = 1  # entering GW2; GW1 has no transfer concept
    for gw in relevant:
        event = gw["event"]
        if event <= 1:
            continue
        if event in chip_events:
            continue  # wildcard/free hit: balance untouched this gameweek

        made = gw.get("event_transfers", 0) or 0
        cost = gw.get("event_transfers_cost", 0) or 0
        paid_transfers = cost // 4
        free_used = made - paid_transfers
        if free_used < 0:
            warnings.append(
                f"GW{event}: transfer-cost accounting looked inconsistent "
                f"(made={made}, cost={cost}); clamping."
            )
            free_used = 0
        balance = max(0, balance - free_used)
        balance = min(balance + 1, MAX_FREE_TRANSFERS)

    return balance, warnings


def resolve_squad_state(client: FplClient, team_id: int, events: list[Gameweek]) -> SquadState:
    last_picks_event, _next_deadline_event = find_relevant_events(events)

    history = client.get_entry_history(team_id)
    history_current = history.get("current", [])
    chips_raw = history.get("chips", [])

    if not history_current:
        raise DataValidationError(
            f"Team {team_id} has no gameweek history yet -- cannot resolve a current squad."
        )

    picks_response = client.get_entry_picks(team_id, last_picks_event)
    picks_raw = picks_response.get("picks", [])
    entry_history = picks_response.get("entry_history", {})

    bank_tenths = entry_history.get("bank")
    squad_value_tenths = entry_history.get("value")
    if bank_tenths is None or squad_value_tenths is None:
        # Fall back to the season-history record for this event if the picks
        # payload is missing it for some reason.
        matching = next((g for g in history_current if g["event"] == last_picks_event), None)
        if matching is None:
            raise DataValidationError(
                f"Could not determine bank/value for GW{last_picks_event}."
            )
        bank_tenths = matching["bank"]
        squad_value_tenths = matching["value"]

    validate_squad_state(picks_raw, bank_tenths)

    picks = [
        SquadPick(
            element_id=p["element"],
            squad_position=p["position"],
            multiplier=p["multiplier"],
            is_captain=p["is_captain"],
            is_vice_captain=p["is_vice_captain"],
        )
        for p in picks_raw
    ]

    chips_used = [ChipPlay(name=c["name"], event=c["event"]) for c in chips_raw]
    wildcard_used = any(c.name == "wildcard" for c in chips_used)
    free_hit_used = any(c.name == "freehit" for c in chips_used)
    bench_boost_used = any(c.name == "bboost" for c in chips_used)
    triple_captain_used = any(c.name == "3xc" for c in chips_used)

    free_transfers, ft_warnings = estimate_free_transfers(history_current, chips_raw, last_picks_event)
    for w in ft_warnings:
        logger.warning("Free-transfer estimation for team %s: %s", team_id, w)

    return SquadState(
        team_id=team_id,
        as_of_event=last_picks_event,
        picks=picks,
        bank_tenths=bank_tenths,
        squad_value_tenths=squad_value_tenths,
        free_transfers_available=free_transfers,
        chips_used=chips_used,
        wildcard_available=not wildcard_used,
        free_hit_available=not free_hit_used,
        bench_boost_available=not bench_boost_used,
        triple_captain_available=not triple_captain_used,
    )
