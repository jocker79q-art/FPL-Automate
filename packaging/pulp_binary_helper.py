"""Shared between both PyInstaller specs: finds PuLP's CBC solver binary for
whatever platform we're building on right now and returns PyInstaller
`binaries`/`datas` entries for it.

Why this matters: PuLP ships a real compiled CBC solver executable per
platform under pulp/solverdir/cbc/<os>/<arch>/ (cbc.exe on Windows). It is a
*data file*, not a Python import, so PyInstaller's static import analysis
can't find it on its own. It must be declared as a `binaries` entry, not
`datas`: PyInstaller only preserves the executable bit for `binaries`
entries, and CBC has to actually run as a subprocess at report-generation
time, not just sit on disk. (Verified with a real build: left in `datas`,
PULP_CBC_CMD fails at runtime with "Not Available (check permissions on
.../cbc)".) Every other platform's solver binary that collect_data_files()
also picks up is left as inert `datas` -- harmless bloat, never executed.
"""
from __future__ import annotations

from pathlib import Path

import pulp
from PyInstaller.utils.hooks import collect_data_files


def get_pulp_binaries_and_datas() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    pulp_root = Path(pulp.__file__).resolve().parent
    solver_path = Path(pulp.PULP_CBC_CMD().path).resolve()
    solver_dest_dir = str(Path("pulp") / solver_path.parent.relative_to(pulp_root))

    datas = [d for d in collect_data_files("pulp") if Path(d[0]).resolve() != solver_path]
    binaries = [(str(solver_path), solver_dest_dir)]
    return binaries, datas
