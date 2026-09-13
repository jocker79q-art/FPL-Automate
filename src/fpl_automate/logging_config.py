"""Logging setup shared by the CLI and all layers.

Guardrail: a filter strips anything that looks like a credential from log
records before they are emitted, as a defence-in-depth measure on top of
"never pass secrets into log calls" discipline elsewhere in the codebase.
"""
from __future__ import annotations

import logging
import re
import sys

_SECRET_PATTERNS = [
    re.compile(r"(password|passwd|pwd|token|secret|apikey|api_key)=([^&\s]+)", re.IGNORECASE),
]


class RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        redacted = msg
        for pattern in _SECRET_PATTERNS:
            redacted = pattern.sub(lambda m: f"{m.group(1)}=***REDACTED***", redacted)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    if root.handlers:
        return  # already configured (e.g. re-entrant CLI calls / tests)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    handler.addFilter(RedactSecretsFilter())
    root.addHandler(handler)
