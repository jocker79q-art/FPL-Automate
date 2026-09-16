# Architecture

## Layers and where they live

| Layer | Module | Responsibility |
|---|---|---|
| Data ingestion | `data/fpl_client.py`, `data/parsers.py` | Public FPL API calls (retry/rate-limit/cache); raw JSON -> typed models |
| Validation | `validation/checks.py` | Refuse to proceed on missing/stale/contradictory data |
| Storage | `storage/db.py`, `storage/models.py` | SQLite persistence (stdlib `sqlite3`, no ORM); shared typed domain models |
| Feature engineering | `features/engineering.py` | Form shrinkage, fixture windows, minutes reliability, value |
| Projection (baseline) | `projections/baseline_model.py` | Interpretable expected/floor/ceiling points model |
| Projection (ML) | `projections/ml/` (`historical.py`, `features.py`, `model.py`, `backtest.py`, `live.py`) | Two-stage hurdle ML model: training data ingestion, leak-free feature engineering, the model itself, walk-forward backtest vs. the baseline, and live inference -- see the **Model** section in `README.md` |
| Risk | `risk/classification.py`, `risk/portfolio.py` | Safe/balanced/risky classification from a projection's own floor/ceiling band; mean-variance risk-adjusted scoring (incl. the correct quadratic captaincy-variance formula) used by `optimization/lineup.py`'s `risk_adjusted` strategy -- see the **Risk** section in `README.md` |
| Squad state | `squad/state.py`, `squad/valuation.py` | Current squad, bank, free transfers, chips, sell values -- all from public entry endpoints |
| Optimisation | `optimization/lineup.py` | ILP (PuLP) for starting XI / bench / captaincy under 4 strategies (3 single-number substitutions + `risk_adjusted`'s genuine mean-variance objective) |
| Transfers | `transfers/engine.py` | Roll vs. 1/2-transfer search, hit-aware, budget/club-limit constrained |
| Reporting | `reporting/report.py` | Assembles the weekly Markdown/JSON report |
| Notifications | `notifications/` | Pluggable `Notifier` interface; SMTP email implementation |
| Execution | `execution/base.py` | Inert by design -- see its module docstring |
| Orchestration | `workflow.py` | Wires the above into `run_weekly_plan()`, including `reconcile_outcomes()` (production score-tracking) |
| CLI | `cli.py` | Typer commands, one per pipeline stage or the full run |
| GUI | `gui/app.py`, `gui/actions.py`, `dotenv_editor.py` | Settings tab (writes .env) + Dashboard tab, over the same core logic as the CLI |
| Shared runtime | `runtime.py`, `win_console.py` | Exe-relative path resolution and Windows console encoding, used by both frontends |
| Packaging | `packaging/*.spec`, `.github/workflows/build-windows-exe.yml` | PyInstaller builds (Windows .exe, CLI + GUI) -- see "Packaging as Windows .exe files" below |

## Why these technology choices

- **Python**, not TypeScript: mature constrained-optimisation (PuLP),
  numerical/data tooling, and the existing FPL community tooling ecosystem
  are overwhelmingly Python. TypeScript has no equivalent to PuLP/OR-tools
  with the same maturity.
- **stdlib `sqlite3`, not SQLAlchemy/Postgres**: the schema is a handful of
  snapshot tables plus a decision log for one user -- an ORM and a server
  process would add operational weight (a service to run, migrations to
  manage) without adding capability at this scale. SQLite is a single file,
  trivially backed up, and sufficient for concurrent-write patterns this
  project actually has (one writer, run sequentially).
- **GitHub Actions, not a hosted server**: the project needs no credentials
  beyond a public team ID and (optionally) SMTP creds, so there is no
  secret-management reason to run it anywhere persistent. A scheduled
  workflow in the user's own repo is free and needs no infrastructure.
- **PuLP (CBC solver)** for the lineup ILP: the problem (choose 11 of 15
  subject to linear formation constraints) is small and exactly the kind of
  problem MILP solvers are built for; a greedy heuristic would risk missing
  the true optimum for marginal formation trade-offs.

## Data flow for `run-weekly-plan`

See `workflow.py::run_weekly_plan` for the literal code; in prose:

1. Fetch `bootstrap-static` + `fixtures` (public, no login). Validate.
2. Parse into typed `Player`/`Team`/`Fixture`/`Gameweek` models; persist
   snapshots to SQLite.
3. Determine the last-finished gameweek (whose picks represent your current
   squad) and the next gameweek (whose deadline we're planning for).
3a. `reconcile_outcomes()`: fill in real results for any past decision or
    logged projection whose gameweek has since finished (see the **Model**
    section in `README.md`, "Production score-tracking"). Best-effort and
    never fatal -- a row it can't resolve yet (e.g. the current-season
    archive hasn't caught up) is simply retried on the next run.
4. Resolve squad state: picks, bank, free transfers (simulated from your
   public transfer/points history), chip usage -- from `entry/{id}/...`
   public endpoints.
5. Compute features + projections for **every** player in the game at
   three horizons (1/3/5 gameweeks): the ML model wherever it covers a
   player (`PROJECTION_MODEL=ml`, the default), the hand-built baseline
   otherwise. Both are cheap because they only need season-aggregate
   fields already present in `bootstrap-static` (baseline) or the
   already-cached current-season historical archive (ML, see
   `projections/ml/live.py`) -- no per-player extra API calls are needed
   (see the module docstring in `features/engineering.py` for why this
   matters for API politeness). This week's owned-squad projections are
   also logged to SQLite here, for a future run's `reconcile_outcomes()`
   to check against real results.
6. Search transfer scenarios (roll; best single transfers; a greedy second
   transfer), each evaluated by actually re-running the lineup optimiser on
   the resulting squad -- so "gain" reflects real best-XI impact, not a
   naive player-vs-player points comparison.
7. Optimise the starting XI/bench/captaincy for the resulting squad under
   all three strategies (conservative/balanced/aggressive).
8. Build and save the report (Markdown + JSON); log the decision to SQLite;
   send an email if configured.

## Testing strategy

- **Unit tests** exercise each layer in isolation with hand-built
  `Player`/`Team`/`Fixture` fixtures (`tests/conftest.py`), so behaviour is
  verified independent of live FPL data shape drift.
- **HTTP-level tests** (`tests/test_fpl_client.py`) use the `responses`
  library to simulate the real API's success/error/retry behaviour without
  any network access.
- **No test ever calls the real FPL API.** This keeps the suite fast,
  deterministic, and safe to run in CI without rate-limit concerns.

## Packaging as Windows .exe files

Two PyInstaller specs -- `packaging/fpl-automate.spec` (CLI, `console=True`)
and `packaging/fpl-automate-gui.spec` (GUI, `console=False`) -- share
`packaging/pulp_binary_helper.py` for PuLP's compiled CBC solver binary and
`fpl_automate/runtime.py` for exe-relative path resolution. Several details
make this actually work rather than fail at runtime:

1. **The CBC binary must be declared as a PyInstaller `binaries` entry, not
   `datas`.** PuLP ships a real platform-specific executable under
   `pulp/solverdir/cbc/<os>/<arch>/`. PyInstaller only preserves the
   executable permission bit for `binaries` entries -- bundling it as plain
   `datas` extracts a file that looks right but fails at solve-time with
   `PulpSolverError: PULP_CBC_CMD: Not Available (check permissions on
   ...)`. This was caught by an actual local build-and-run smoke test before
   ever pushing to CI, not assumed.
2. **`cli.py`'s `main()` (not the Typer `app` object) is the CLI's real
   entry point.** It resolves `.env`/`data/`/`reports/` relative to the
   exe's own folder rather than the caller's current directory (a
   double-clicked exe's cwd isn't reliable -- see `runtime.py`, shared with
   the GUI), auto-creates a starter `.env` from the bundled `.env.example`
   on first run, and pauses before closing (only when stdin is an actual
   terminal -- `sys.stdin.isatty()` -- so it never hangs a non-interactive
   CI run) so a double-clicked console window doesn't just flash and vanish.
3. **A `console=False` build has no console attached**, so `sys.stdout`/
   `sys.stderr` can be `None` at startup -- any print or logging
   `StreamHandler` touching them crashes immediately. `gui/app.py` replaces
   both with an in-memory `io.StringIO` before any other import, as the
   very first thing the module does.
4. **The GUI's Settings tab writes .env through `dotenv_editor.py`**, which
   rewrites only the specific keys the form manages and leaves every
   comment, blank line, and unrelated key untouched -- and calls
   `config.clear_settings_cache()` afterwards, since `get_settings()` is
   `functools.cache`d and would otherwise keep serving the pre-save values
   for the rest of the process.

PyInstaller does not cross-compile, so the actual Windows build only happens
on a `windows-latest` GitHub Actions runner
(`.github/workflows/build-windows-exe.yml`), which also smoke-tests both
built exes for real before they're ever handed to a user: the CLI against
the *real* FPL API and the *real* Windows CBC solver (`health-check`,
`analyse-squad`, `optimise-lineup`, `recommend-transfers`,
`run-weekly-plan`), and the GUI by actually launching its window
(gated behind `FPL_AUTOMATE_GUI_SELFTEST=1`, which schedules an automatic
close ~1.5s after startup so CI doesn't hang waiting for a click) --
possible because that runner, unlike some sandboxed dev environments, has
both normal outbound internet access and a real desktop session.
