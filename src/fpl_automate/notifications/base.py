"""Notifier interface. New channels (Discord, Telegram, ...) implement this."""
from __future__ import annotations

from typing import Protocol


class Notifier(Protocol):
    def send(self, subject: str, body_text: str, body_html: str | None = None) -> None: ...


class NullNotifier:
    """Used when notifications are disabled; reports are still written to disk."""

    def send(self, subject: str, body_text: str, body_html: str | None = None) -> None:
        return None
