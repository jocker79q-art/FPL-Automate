"""Windows console encoding fix, shared by any console-based frontend.

Windows' legacy console codepage (often cp1252 or cp437, not UTF-8) mangles
the "£" in every price we print -- confirmed via a real build's output
showing garbled bytes before this fix. Not relevant to the GUI frontend
(it has no console to reconfigure).
"""
from __future__ import annotations

import contextlib
import sys


def fix_windows_console_encoding() -> None:
    if sys.platform != "win32":
        return
    with contextlib.suppress(Exception):
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
