# PyInstaller spec for the fpl-automate desktop GUI (Settings + Dashboard
# tabs), packaged as a single windowed executable (no console box). See
# fpl-automate.spec for the CLI build and pulp_binary_helper.py for why the
# CBC solver binary needs special handling -- identical reasoning applies
# here since the GUI runs the same optimiser.
import sys
from pathlib import Path

REPO_ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is injected by PyInstaller
sys.path.insert(0, str(REPO_ROOT / "packaging"))
from pulp_binary_helper import get_pulp_binaries_and_datas  # noqa: E402

ENTRY_SCRIPT = str(REPO_ROOT / "src" / "fpl_automate" / "gui" / "app.py")

binaries, datas = get_pulp_binaries_and_datas()
datas.append((str(REPO_ROOT / ".env.example"), "."))

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
    name="fpl-automate-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed: no console box behind the GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
)
