"""Execution layer: the ONLY place this project would ever act on your FPL account.

Current state: fully inert. There is no code anywhere in this repository
that logs into fantasy.premierleague.com or submits transfers/lineup
changes. `ENABLE_AUTO_EXECUTION` exists as a config field so that if a
real execution backend is built in a future phase, its wiring and default
(off) are already auditable -- but flipping that flag today does nothing,
because `execute_transfer_scenario` below always raises.

Building real execution would require handling your FPL login (email +
password -> session cookie), since FPL has no public write API. That is a
materially bigger trust and security surface than anything else in this
project, and per the project's own safety requirements it must never
happen silently: it needs its own explicit design/review pass and your
explicit go-ahead before a single line of submission code is written.
"""
from __future__ import annotations

from fpl_automate.config import Settings
from fpl_automate.transfers.engine import TransferScenario


class ExecutionDisabledError(RuntimeError):
    pass


class ExecutionLayer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def execute_transfer_scenario(self, scenario: TransferScenario) -> None:
        self._settings.require_not_stopped()
        if not self._settings.enable_auto_execution:
            raise ExecutionDisabledError(
                "ENABLE_AUTO_EXECUTION is false. This system is recommendation-only. "
                "Review the report and make the transfer yourself on the FPL website/app."
            )
        # Even if a future user sets ENABLE_AUTO_EXECUTION=true, there is deliberately
        # no submission logic implemented yet -- see module docstring.
        raise NotImplementedError(
            "Automatic execution is not implemented in this version of the project. "
            "No transfer has been submitted anywhere."
        )
