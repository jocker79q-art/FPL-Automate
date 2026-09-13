# FPL Automate

A quantitative decision-support system for Fantasy Premier League -- and, longer
term, a reusable platform for other structured decisions (bursaries,
scholarships, funding applications). **It is recommendation-only.** It never
logs into your FPL account, never makes a transfer, and never changes your
squad. It reads public FPL data, produces a report, and leaves every decision
to you.

Team tracked by default: `9242093`. There are no guarantees of any point
total -- see "What this is not" below.

## Safety model (read this first)

- **No login, ever, in this version.** Every FPL endpoint this project calls
  is public (the same data anyone can view on the FPL website without
  signing in): player stats, fixtures, and any manager's picks/history/
  transfers. There is no code anywhere in this repo that authenticates to
  fantasy.premierleague.com.
- **Recommendation-only.** `ENABLE_AUTO_EXECUTION` exists in `.env` for
  future auditability but does nothing today --
  `src/fpl_automate/execution/base.py` always refuses, even if you set it to
  `true`. Building real execution would require handling your FPL login,
  which is a materially bigger trust/security surface and is deliberately
  out of scope until a future, explicitly-requested phase.
- **Emergency stop.** Set `EMERGENCY_STOP=true` in `.env` (or as a repo
  variable for the scheduled workflow) and the system refuses to do anything
  beyond nothing at all.
- **Points-hit discipline.** `MAX_TRANSFER_RISK` (default `4`) caps how big a
  hit the system will ever mark "recommended." Every scenario above that
  threshold is still shown, but flagged and never auto-approved.
- **Fails safe on bad data.** If FPL's data looks incomplete, stale, or
  self-contradictory, the pipeline raises an error and stops rather than
  guessing (see `validation/checks.py`).
- **No secrets committed.** `.env` is git-ignored; `.env.example` documents
  every variable; SMTP credentials are validated at startup and never
  logged (a logging filter also redacts anything that looks like a
  credential, as a second layer).

## What this is not

- It does **not** guarantee 80 points a gameweek, or any score. It reports
  an *expected* value, a *floor* (pessimistic case), a *ceiling* (optimistic
  case), and a *confidence* score -- treat all four together, not the
  expected value alone.
- It does **not** trade or touch money -- Fantasy Premier League is a free
  game.
- The projection model (`projections/baseline_model.py`) is an interpretable,
  hand-built baseline, **not yet backtested** (see Roadmap). Trust its
  *relative* ranking of similar players more than its absolute numbers until
  a backtest phase validates it.

## Architecture

```
fetch (FPL public API) -> validate -> parse into typed models
    -> feature engineering -> baseline expected-points model (1/3/5 GW)
    -> squad state (bank, free transfers, chips) [public entry history]
    -> transfer engine (roll vs. 1/2 transfers, hit-aware)
    -> lineup optimiser (ILP: formation + captaincy, 3 strategies)
    -> weekly report (Markdown + JSON) -> optional email notification
    -> decision saved to SQLite for later outcome tracking
```

Each stage is an independent module under `src/fpl_automate/` and is
independently testable (see `tests/`). See `docs/ARCHITECTURE.md` for a
file-by-file breakdown and the assumptions baked into each layer.

## Setup

Requires Python 3.11+.

```bash
git clone <this repo>
cd FPL-Automate
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env: at minimum confirm FPL_TEAM_ID (default 9242093 is already set)
```

### Environment variables

All variables are documented inline in `.env.example`. Summary:

| Variable | Required? | Purpose |
|---|---|---|
| `FPL_TEAM_ID` | Yes | Your team ID (public; from your FPL team URL) |
| `FPL_MINI_LEAGUE_ID` | No | For future league-rank tracking |
| `EMERGENCY_STOP` | No (default false) | Master kill switch |
| `ENABLE_AUTO_EXECUTION` | No (default false) | Reserved; currently a no-op even if true |
| `MAX_TRANSFER_RISK` | No (default 4) | Max points-hit eligible for auto-recommendation |
| `DATABASE_URL` | No | SQLite path |
| `EMAIL_NOTIFICATIONS_ENABLED` + `SMTP_*` / `EMAIL_*` | No | Email alerts |

## Running it

```bash
fpl-automate health-check          # config, database, FPL API reachability
fpl-automate fetch-data            # pull + persist current player/fixture data
fpl-automate validate-data         # fetch + run data-quality checks only
fpl-automate analyse-squad         # your current squad, bank, free transfers, chips
fpl-automate recommend-transfers   # ranked transfer scenarios (roll vs. 1/2 transfers)
fpl-automate optimise-lineup       # best XI/bench/captaincy for your CURRENT squad
fpl-automate run-weekly-plan       # full pipeline -> writes reports/gwN_<timestamp>.md/.json
fpl-automate show-history          # past decisions + recorded outcomes
```

Add `--strategy conservative|balanced|aggressive` to
`recommend-transfers`/`optimise-lineup`/`run-weekly-plan` (default
`balanced`).

Not yet implemented (see `docs/ROADMAP.md`): `backtest`, `bursary-search`,
`bursary-score`, `generate-checklist` -- these exist as CLI stubs that say so
rather than silently doing nothing.

## Scheduling

`.github/workflows/weekly-plan.yml` runs `run-weekly-plan` automatically
(twice daily, plus on-demand via the Actions tab "Run workflow" button) and
commits the resulting report + decision-history database back to the repo,
so your weekly plans and their eventual outcomes are visible in git history.

To enable it:
1. Push this repo to GitHub (or use the one it's already in).
2. Repository Settings -> Secrets and variables -> Actions:
   - **Variables**: `FPL_TEAM_ID` (e.g. `9242093`), optionally
     `FPL_MINI_LEAGUE_ID`, `MAX_TRANSFER_RISK`, `EMERGENCY_STOP`,
     `EMAIL_NOTIFICATIONS_ENABLED`, `SMTP_HOST`, `SMTP_PORT`.
   - **Secrets** (only if email notifications are on): `SMTP_USERNAME`,
     `SMTP_PASSWORD` (an app password, never your real account password),
     `EMAIL_FROM`, `EMAIL_TO`.
3. That's it -- no FPL login secret is needed anywhere, because nothing in
   this project logs in.

Prefer to run it yourself instead? Skip the above and just run
`fpl-automate run-weekly-plan` locally before each deadline, or wire it into
your own cron/Task Scheduler.

## Deployment options

- **GitHub Actions (recommended, set up above).** Free, no server to
  maintain, report history lives in your repo.
- **Your own machine.** `pip install -e .` + cron/Task Scheduler calling
  `fpl-automate run-weekly-plan`.
- **Docker.** Not included yet (see Roadmap) -- the project has no
  system-level dependencies beyond Python + the packages in
  `pyproject.toml`, so a minimal `python:3.11-slim` image with `pip install
  -e .` is all a Dockerfile would need.

## Security guidance

- Never commit `.env`. It's git-ignored; double-check `git status` before
  committing if you ever hand-edit `.gitignore`.
- For Gmail SMTP, use an **App Password** (requires 2FA enabled on the
  Google account), never your real account password.
- The only "identity" this project needs is your public FPL team ID -- not
  a secret, but personal, hence it lives in `.env`/repo variables rather
  than being hard-coded.
- If you ever build the (currently unimplemented) execution layer, treat
  your FPL login as a high-value secret and re-read
  `src/fpl_automate/execution/base.py`'s module docstring first.

## Testing

```bash
pytest              # unit + integration tests (mocked FPL responses, no network)
ruff check src tests
mypy
```

64 tests as of this writing, covering: the FPL client's retry/cache/error
handling (via `responses`-mocked HTTP), the data-validation gate, feature
engineering (form shrinkage, fixture windows, minutes reliability), the
baseline projection model (blank/double gameweeks, injury handling,
captaincy), the lineup ILP optimiser (formation constraints, strategy-driven
captaincy), the transfer engine (budget/club-limit enforcement, hit
thresholds), free-transfer ledger simulation, sell-value/price-tax logic,
and SQLite persistence.

## Known limitations

- **Not backtested yet.** The baseline model's coefficients are
  hand-set and documented (see the docstring in
  `projections/baseline_model.py`), not fitted or validated against history.
- **Sell value can be underestimated** for players still in your very first
  squad who've never been re-bought (no purchase-price record exists via the
  public API for those) -- the fallback assumes no profit, which understates
  your real budget but never overstates it.
- **Free-transfer accounting assumes current (2024/25+) rules** (bank up to
  5, wildcard/free hit don't touch the ledger) and treats each chip as
  single-use for the season, matching your stated situation (wildcard
  already used) -- see the docstring in `squad/state.py`.
- **Fixture-difficulty windows use an average**, not fixture-by-fixture
  values, across a multi-gameweek horizon.
- **The two-transfer search is greedy**, not exhaustive: it can miss a
  jointly-optimal pair that isn't optimal individually.
- **This sandboxed development session could not reach
  fantasy.premierleague.com** (blocked by this environment's own network
  policy) -- the full pipeline is validated with 64 mocked-data tests, but
  you should run `fpl-automate health-check` yourself as the first real
  connectivity check.

## Roadmap

See `docs/ROADMAP.md` for the full phased plan: backtesting/walk-forward
validation, richer ML-based projections (benchmarked against this
baseline, not replacing it silently), a dashboard, and the bursary/
scholarship quantitative-decision module.
