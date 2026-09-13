# Architecture

## Layers and where they live

| Layer | Module | Responsibility |
|---|---|---|
| Data ingestion | `data/fpl_client.py`, `data/parsers.py` | Public FPL API calls (retry/rate-limit/cache); raw JSON -> typed models |
| Validation | `validation/checks.py` | Refuse to proceed on missing/stale/contradictory data |
| Storage | `storage/db.py`, `storage/models.py` | SQLite persistence (stdlib `sqlite3`, no ORM); shared typed domain models |
| Feature engineering | `features/engineering.py` | Form shrinkage, fixture windows, minutes reliability, value |
| Projection | `projections/baseline_model.py` | Interpretable expected/floor/ceiling points model |
| Squad state | `squad/state.py`, `squad/valuation.py` | Current squad, bank, free transfers, chips, sell values -- all from public entry endpoints |
| Optimisation | `optimization/lineup.py` | ILP (PuLP) for starting XI / bench / captaincy under 3 strategies |
| Transfers | `transfers/engine.py` | Roll vs. 1/2-transfer search, hit-aware, budget/club-limit constrained |
| Reporting | `reporting/report.py` | Assembles the weekly Markdown/JSON report |
| Notifications | `notifications/` | Pluggable `Notifier` interface; SMTP email implementation |
| Execution | `execution/base.py` | Inert by design -- see its module docstring |
| Orchestration | `workflow.py` | Wires the above into `run_weekly_plan()` |
| CLI | `cli.py` | Typer commands, one per pipeline stage or the full run |
| Packaging | `packaging/fpl-automate.spec`, `.github/workflows/build-windows-exe.yml` | PyInstaller build (Windows .exe) -- see "Packaging as a Windows .exe" below |

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
4. Resolve squad state: picks, bank, free transfers (simulated from your
   public transfer/points history), chip usage -- from `entry/{id}/...`
   public endpoints.
5. Compute features + baseline projections for **every** player in the
   game at three horizons (1/3/5 gameweeks). This is cheap because the
   baseline model only needs season-aggregate fields already present in
   `bootstrap-static` -- no per-player extra API calls are needed for the
   MVP model (see the module docstring in `features/engineering.py` for why
   this matters for API politeness).
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

## Packaging as a Windows .exe

`packaging/fpl-automate.spec` is a PyInstaller spec that bundles `cli.py`
(via the `main()` wrapper, not `app()` directly) plus PuLP's compiled CBC
solver binary into a single-file executable. Two details make this actually
work rather than fail at runtime:

1. **The CBC binary must be declared as a PyInstaller `binaries` entry, not
   `datas`.** PuLP ships a real platform-specific executable under
   `pulp/solverdir/cbc/<os>/<arch>/`. PyInstaller only preserves the
   executable permission bit for `binaries` entries -- bundling it as plain
   `datas` extracts a file that looks right but fails at solve-time with
   `PulpSolverError: PULP_CBC_CMD: Not Available (check permissions on
   ...)`. This was caught by an actual local build-and-run smoke test before
   ever pushing to CI, not assumed.
2. **`cli.py`'s `main()` (not the Typer `app` object) is the real entry
   point.** It resolves `.env`/`data/`/`reports/` relative to the exe's own
   folder rather than the caller's current directory (a double-clicked
   exe's cwd isn't reliable), auto-creates a starter `.env` from the bundled
   `.env.example` on first run, and pauses before closing (only when stdin
   is an actual terminal -- `sys.stdin.isatty()` -- so it never hangs a
   non-interactive CI run) so a double-clicked console window doesn't just
   flash and vanish.

PyInstaller does not cross-compile, so the actual Windows build only happens
on a `windows-latest` GitHub Actions runner
(`.github/workflows/build-windows-exe.yml`), which also smoke-tests the
built `.exe` against the *real* FPL API and the *real* Windows CBC solver
(`health-check`, `analyse-squad`, `optimise-lineup`) before it's ever handed
to a user, since that runner -- unlike some sandboxed dev environments --
has normal outbound internet access.
