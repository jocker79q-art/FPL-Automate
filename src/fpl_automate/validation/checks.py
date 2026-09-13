"""Data-quality gate: refuse to proceed on missing, stale, or contradictory data.

Every ingestion call routes its result through here before it is persisted
or used for projections. Failing safe means raising `DataValidationError`
rather than silently continuing with partial or nonsensical data -- per the
project's safety requirements, a bad recommendation is worse than no
recommendation.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

EXPECTED_TEAM_COUNT = 20
MIN_EXPECTED_PLAYERS = 400  # generous floor; a real season has 600-700+
VALID_POSITIONS = {1, 2, 3, 4}
VALID_STATUSES = {"a", "d", "i", "s", "u", "n"}


class DataValidationError(RuntimeError):
    """Raised when incoming data fails a safety check; caller must not proceed."""


def validate_bootstrap_static(data: dict[str, Any]) -> None:
    for key in ("elements", "teams", "events", "element_types"):
        if key not in data:
            raise DataValidationError(f"bootstrap-static response missing '{key}'")
        if not isinstance(data[key], list):
            raise DataValidationError(f"bootstrap-static '{key}' is not a list")

    if len(data["teams"]) != EXPECTED_TEAM_COUNT:
        raise DataValidationError(
            f"Expected {EXPECTED_TEAM_COUNT} teams, got {len(data['teams'])}. "
            "Refusing to proceed with an incomplete season snapshot."
        )

    if len(data["elements"]) < MIN_EXPECTED_PLAYERS:
        raise DataValidationError(
            f"Only {len(data['elements'])} players returned (expected >= "
            f"{MIN_EXPECTED_PLAYERS}). Data looks truncated or the API shape changed."
        )

    team_ids = {t["id"] for t in data["teams"]}
    issues: list[str] = []
    for p in data["elements"]:
        pid = p.get("id")
        if p.get("element_type") not in VALID_POSITIONS:
            issues.append(f"player {pid} has invalid position {p.get('element_type')!r}")
        if p.get("team") not in team_ids:
            issues.append(f"player {pid} references unknown team {p.get('team')!r}")
        if p.get("status") not in VALID_STATUSES:
            issues.append(f"player {pid} has unrecognised status {p.get('status')!r}")
        now_cost = p.get("now_cost")
        if not isinstance(now_cost, int) or not (30 <= now_cost <= 200):
            issues.append(f"player {pid} has implausible price {now_cost!r} (tenths of £m)")
        selected_by = p.get("selected_by_percent")
        try:
            if not (0.0 <= float(selected_by) <= 100.0):
                issues.append(f"player {pid} has out-of-range ownership {selected_by!r}")
        except (TypeError, ValueError):
            issues.append(f"player {pid} has non-numeric ownership {selected_by!r}")

    if issues:
        # A handful of odd records (e.g. a player with a data-entry quirk) shouldn't
        # block the whole run, but a large fraction failing signals a real problem.
        threshold = max(5, int(0.02 * len(data["elements"])))
        if len(issues) > threshold:
            raise DataValidationError(
                f"{len(issues)} player records failed validation (threshold {threshold}); "
                f"first few: {issues[:5]}"
            )
        logger.warning("Ignoring %d minor player validation issues: %s", len(issues), issues[:5])


def validate_freshness(deadline_time: datetime, max_age_before_deadline: timedelta) -> None:
    """Ensure we are not relying on data fetched long before/after where we think we are."""
    now = datetime.now(UTC)
    if deadline_time.tzinfo is None:
        deadline_time = deadline_time.replace(tzinfo=UTC)
    if now > deadline_time:
        raise DataValidationError(
            f"Target gameweek deadline {deadline_time.isoformat()} has already passed "
            f"(now {now.isoformat()}). Refusing to produce a pre-deadline recommendation."
        )
    if deadline_time - now > max_age_before_deadline:
        logger.info(
            "Deadline %s is more than %s away; recommendation will be treated as provisional.",
            deadline_time.isoformat(),
            max_age_before_deadline,
        )


def validate_squad_state(picks: list[dict[str, Any]], bank_tenths: int) -> None:
    if len(picks) != 15:
        raise DataValidationError(f"Squad has {len(picks)} picks, expected exactly 15")
    if bank_tenths < 0:
        raise DataValidationError(f"Negative bank balance ({bank_tenths / 10:.1f}m) is impossible")
    element_ids = [p["element"] for p in picks]
    if len(set(element_ids)) != len(element_ids):
        raise DataValidationError("Squad contains duplicate players")
    captains = [p for p in picks if p.get("is_captain")]
    vice_captains = [p for p in picks if p.get("is_vice_captain")]
    if len(captains) != 1:
        raise DataValidationError(f"Expected exactly 1 captain, found {len(captains)}")
    if len(vice_captains) != 1:
        raise DataValidationError(f"Expected exactly 1 vice-captain, found {len(vice_captains)}")
    if captains[0]["element"] == vice_captains[0]["element"]:
        raise DataValidationError("Captain and vice-captain cannot be the same player")
