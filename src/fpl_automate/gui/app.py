"""Desktop GUI: a Settings tab (writes .env for you -- no text editor needed)
and a Dashboard tab (runs the same pipeline the CLI does, shown inline).

This is a second presentation layer over the exact same core logic the CLI
uses (`gui/actions.py` calls straight into `workflow.py` /
`optimization/lineup.py` / `transfers/engine.py`) -- nothing about the
underlying analysis differs between the two frontends.

IMPORTANT (PyInstaller windowed builds): a `console=False` .exe has no
console attached, so `sys.stdout`/`sys.stderr` can be `None` at startup.
Any code that prints or sets up a logging StreamHandler on them would
crash immediately. The guard below must run before any other
`fpl_automate` import.
"""
from __future__ import annotations

import io
import sys

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

import os
import subprocess
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, scrolledtext, ttk

from fpl_automate.config import ConfigError, Settings, clear_settings_cache
from fpl_automate.dotenv_editor import read_env_values, update_env_file
from fpl_automate.gui import actions
from fpl_automate.logging_config import configure_logging
from fpl_automate.runtime import (
    DEFAULT_REPORTS_DIR,
    ENV_EXAMPLE_PATH,
    ENV_FILE_PATH,
    get_app_settings,
)

# (key, label, is_secret) -- drives both the Settings form layout and what
# gets written back to .env. Kept in one place so the two can't drift apart.
IDENTITY_FIELDS = [
    ("FPL_TEAM_ID", "FPL Team ID", False),
    ("FPL_MINI_LEAGUE_ID", "Mini League ID (optional)", False),
]
SAFETY_FIELDS = [
    ("MAX_TRANSFER_RISK", "Max points-hit to auto-recommend", False),
]
PROJECTION_MODEL_CHOICES = ["ml", "baseline"]
RISK_FIELDS = [
    ("RISK_AVERSION", "Risk aversion (0 = same as balanced strategy)", False),
]
EMAIL_FIELDS = [
    ("SMTP_HOST", "SMTP host", False),
    ("SMTP_PORT", "SMTP port", False),
    ("SMTP_USERNAME", "SMTP username", False),
    ("SMTP_PASSWORD", "SMTP password / app password", True),
    ("EMAIL_FROM", "From address", False),
    ("EMAIL_TO", "To address (usually yourself)", False),
]


def _load_effective_env_values() -> dict[str, str]:
    """Bundled template defaults, overridden by whatever's actually in .env."""
    values = read_env_values(ENV_EXAMPLE_PATH)
    values.update(read_env_values(ENV_FILE_PATH))
    return values


def _open_folder(path: os.PathLike[str] | str) -> None:
    path = str(path)
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


class SettingsTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, status_var: tk.StringVar) -> None:
        super().__init__(parent, padding=12)
        self._status_var = status_var
        self._vars: dict[str, tk.Variable] = {}

        values = _load_effective_env_values()

        row = 0
        row = self._add_section(row, "Identity", IDENTITY_FIELDS, values)
        row = self._add_checkbox(row, "EMERGENCY_STOP", "Emergency stop (blocks everything)", values)
        row = self._add_section(row, "Safety", SAFETY_FIELDS, values)
        row = self._add_dropdown(
            row, "PROJECTION_MODEL", "Projection model", PROJECTION_MODEL_CHOICES, values, default="ml"
        )
        row = self._add_section(row, "Risk", RISK_FIELDS, values)
        row = self._add_checkbox(
            row, "EMAIL_NOTIFICATIONS_ENABLED", "Enable email notifications", values
        )
        row = self._add_section(row, "Email (SMTP)", EMAIL_FIELDS, values)
        row = self._add_checkbox(row, "SMTP_USE_TLS", "Use TLS", values)

        button_row = ttk.Frame(self)
        button_row.grid(row=row, column=0, columnspan=2, pady=(16, 0), sticky="w")
        ttk.Button(button_row, text="Save Settings", command=self._save).pack(side="left")
        ttk.Button(button_row, text="Send Test Email", command=self._send_test_email).pack(
            side="left", padx=(8, 0)
        )

        self.columnconfigure(1, weight=1)

    def _add_section(
        self, row: int, title: str, fields: list[tuple[str, str, bool]], values: dict[str, str]
    ) -> int:
        ttk.Label(self, text=title, font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(12, 4)
        )
        row += 1
        for key, label, is_secret in fields:
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=values.get(key, ""))
            entry = ttk.Entry(self, textvariable=var, show="*" if is_secret else "", width=40)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            self._vars[key] = var
            row += 1
        return row

    def _add_dropdown(
        self, row: int, key: str, label: str, choices: list[str], values: dict[str, str], default: str
    ) -> int:
        ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", pady=2)
        current = values.get(key, "").strip() or default
        var = tk.StringVar(value=current if current in choices else default)
        combo = ttk.Combobox(self, textvariable=var, values=choices, state="readonly", width=37)
        combo.grid(row=row, column=1, sticky="ew", pady=2)
        self._vars[key] = var
        return row + 1

    def _add_checkbox(self, row: int, key: str, label: str, values: dict[str, str]) -> int:
        var = tk.BooleanVar(value=values.get(key, "false").strip().lower() == "true")
        ttk.Checkbutton(self, text=label, variable=var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(8, 2)
        )
        self._vars[key] = var
        return row + 1

    def _collect_updates(self) -> dict[str, str]:
        updates: dict[str, str] = {}
        for key, var in self._vars.items():
            if isinstance(var, tk.BooleanVar):
                updates[key] = "true" if var.get() else "false"
            else:
                updates[key] = var.get().strip()
        return updates

    def _save(self) -> None:
        updates = self._collect_updates()
        update_env_file(ENV_FILE_PATH, updates)
        clear_settings_cache()
        try:
            get_app_settings()
        except ConfigError as exc:
            self._status_var.set(f"Saved, but settings are invalid: {exc}")
            return
        self._status_var.set(f"Settings saved to {ENV_FILE_PATH}")

    def _send_test_email(self) -> None:
        self._save()
        try:
            settings = get_app_settings()
            result = actions.send_test_email(settings)
            self._status_var.set(result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user in a dialog, not swallowed
            messagebox.showerror("Test email failed", str(exc))
            self._status_var.set("Test email failed -- see error dialog.")


class DashboardTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, status_var: tk.StringVar) -> None:
        super().__init__(parent, padding=12)
        self._status_var = status_var
        self._buttons: list[ttk.Button] = []

        controls = ttk.Frame(self)
        controls.pack(fill="x")

        ttk.Label(controls, text="Strategy:").pack(side="left")
        self._strategy_var = tk.StringVar(value="balanced")
        strategy_menu = ttk.Combobox(
            controls,
            textvariable=self._strategy_var,
            values=["conservative", "balanced", "aggressive", "risk_adjusted"],
            state="readonly",
            width=14,
        )
        strategy_menu.pack(side="left", padx=(4, 16))

        self._add_button(controls, "Health Check", lambda s: actions.health_check(s))
        self._add_button(controls, "Analyse Squad", lambda s: actions.analyse_squad(s))
        self._add_button(
            controls, "Recommend Transfers",
            lambda s: actions.recommend_transfers_action(s, self._strategy_var.get()),  # type: ignore[arg-type]
        )
        self._add_button(
            controls, "Optimise Lineup",
            lambda s: actions.optimise_lineup_action(s, self._strategy_var.get()),  # type: ignore[arg-type]
        )
        self._add_button(
            controls, "Run Weekly Plan",
            lambda s: actions.run_weekly_plan_action(s, self._strategy_var.get()),  # type: ignore[arg-type]
        )
        ttk.Button(controls, text="Open Reports Folder", command=self._open_reports).pack(
            side="left", padx=(16, 0)
        )

        self._output = scrolledtext.ScrolledText(self, wrap="word", height=30, font=("Consolas", 10))
        self._output.pack(fill="both", expand=True, pady=(12, 0))
        self._output.configure(state="disabled")

    def _add_button(self, parent: tk.Widget, text: str, action: Callable[[Settings], str]) -> None:
        button = ttk.Button(parent, text=text, command=lambda: self._run(text, action))
        button.pack(side="left", padx=(0, 6))
        self._buttons.append(button)

    def _set_output(self, text: str) -> None:
        self._output.configure(state="normal")
        self._output.delete("1.0", "end")
        self._output.insert("1.0", text)
        self._output.configure(state="disabled")

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self._buttons:
            button.configure(state=state)

    def _run(self, label: str, action: Callable[[Settings], str]) -> None:
        try:
            settings = get_app_settings()
        except ConfigError as exc:
            messagebox.showerror(
                "Settings needed", f"Fix your settings first (see the Settings tab):\n\n{exc}"
            )
            return

        self._status_var.set(f"Running {label}...")
        self._set_buttons_enabled(False)
        self._set_output(f"Running {label}...\n")

        def worker() -> None:
            try:
                result = action(settings)
                self.after(0, lambda: self._on_done(label, result, None))
            except Exception as exc:  # noqa: BLE001 - reported to the user, not swallowed
                # `exc` is auto-deleted by Python when this except block exits, so it
                # must be turned into a plain string now -- a lambda referencing `exc`
                # directly would raise NameError once `self.after` actually runs it.
                message = str(exc)
                self.after(0, lambda: self._on_done(label, None, message))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, label: str, result: str | None, error: str | None) -> None:
        self._set_buttons_enabled(True)
        if error is not None:
            self._status_var.set(f"{label} failed.")
            self._set_output(f"{label} failed:\n\n{error}")
        else:
            self._status_var.set(f"{label} finished.")
            self._set_output(result or "")

    def _open_reports(self) -> None:
        DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        _open_folder(DEFAULT_REPORTS_DIR)


class App(ttk.Frame):
    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root)
        self.pack(fill="both", expand=True)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        status_var = tk.StringVar(value="Ready.")
        notebook.add(DashboardTab(notebook, status_var), text="Dashboard")
        notebook.add(SettingsTab(notebook, status_var), text="Settings")

        status_bar = ttk.Label(self, textvariable=status_var, anchor="w", relief="sunken", padding=(6, 2))
        status_bar.pack(fill="x", side="bottom")


def gui_main() -> None:
    configure_logging("INFO")
    root = tk.Tk()
    root.title("FPL Automate")
    root.geometry("980x720")
    App(root)

    if os.environ.get("FPL_AUTOMATE_GUI_SELFTEST"):
        # CI smoke test: confirm the window builds and the mainloop runs at
        # all on the real target platform, then close automatically.
        root.after(1500, root.destroy)

    root.mainloop()


if __name__ == "__main__":
    gui_main()
