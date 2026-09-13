# PyInstaller spec for the fpl-automate CLI, packaged as a single console
# executable. PyInstaller does NOT cross-compile: run this with
# `pyinstaller packaging/fpl-automate.spec` on the target OS (Windows for a
# .exe -- see .github/workflows/build-windows-exe.yml, which does exactly
# that on a windows-latest runner).
#
# Why PuLP's data files matter: PuLP ships a real compiled CBC solver binary
# per platform under pulp/solverdir/cbc/<os>/<arch>/ (cbc.exe on Windows).
# It is a *data file*, not a Python import, so PyInstaller's static import
# analysis can't find it on its own -- collect_data_files pulls in the whole
# solverdir tree (all platforms; harmless bloat, but simple and robust) so
# the bundled app can still launch a solver process at runtime.
from pathlib import Path

import pulp
from PyInstaller.utils.hooks import collect_data_files

REPO_ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is injected by PyInstaller
ENTRY_SCRIPT = str(REPO_ROOT / "src" / "fpl_automate" / "cli.py")

# The CBC solver binary for whatever platform we're building on right now (PyInstaller
# never cross-compiles, so "right now" is always the target platform) MUST be declared
# as `binaries`, not `datas`: PyInstaller only preserves the executable bit for
# `binaries` entries, and CBC has to actually run as a subprocess at report-generation
# time, not just sit on disk. (Verified: with it left in `datas`, PULP_CBC_CMD fails at
# runtime with "Not Available (check permissions on .../cbc)".) Every other platform's
# solver binary that collect_data_files() also picks up is left as inert `datas` --
# harmless bloat, never executed.
_pulp_root = Path(pulp.__file__).resolve().parent
_solver_path = Path(pulp.PULP_CBC_CMD().path).resolve()
_solver_dest_dir = str(Path("pulp") / _solver_path.parent.relative_to(_pulp_root))

datas = [d for d in collect_data_files("pulp") if Path(d[0]).resolve() != _solver_path]
datas.append((str(REPO_ROOT / ".env.example"), "."))
binaries = [(str(_solver_path), _solver_dest_dir)]

a = Analysis(  # noqa: F821 - Analysis/PYZ/EXE are injected by PyInstaller's spec exec
    [ENTRY_SCRIPT],
    pathex=[str(REPO_ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="fpl-automate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # keep the console window: this is a CLI, not a GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
)
