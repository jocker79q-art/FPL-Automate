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
- The interpretable, hand-built baseline (`projections/baseline_model.py`)
  is now backtested (see **Model** below): a walk-forward comparison found
  its floor/ceiling bands capture the real range of outcomes far less often
  than their own stated confidence implies, and an ML model beats it on raw
  accuracy. By default this project now uses that ML model in production
  (see `PROJECTION_MODEL`), falling back to the baseline per player only
  where the ML model doesn't cover them -- never a silent substitution,
  since every projection's rationale states which one produced it.

## Architecture

```
fetch (FPL public API) -> validate -> parse into typed models
    -> [reconcile past decisions/projections against real results, see Model]
    -> feature engineering -> expected-points model (1/3/5 GW):
           ML model (trained + backtested, see Model) wherever it covers a
           player, hand-built baseline otherwise
    -> squad state (bank, free transfers, chips) [public entry history]
    -> transfer engine (roll vs. 1/2 transfers, hit-aware)
    -> lineup optimiser (ILP: formation + captaincy, 3 strategies)
    -> weekly report (Markdown + JSON) -> optional email notification
    -> decision + this week's projections logged to SQLite for outcome tracking
```

Each stage is an independent module under `src/fpl_automate/` and is
independently testable (see `tests/`). See `docs/ARCHITECTURE.md` for a
file-by-file breakdown and the assumptions baked into each layer.

## Model

Two projection models exist side by side, per `docs/ROADMAP.md`'s explicit
rule: a learned model may only be added *alongside* the hand-built baseline,
validated by a real backtest, never as a silent replacement.

- **Baseline** (`projections/baseline_model.py`): interpretable, hand-set
  coefficients -- every number in a projection is traceable to a documented
  assumption in that file.
- **ML model** (`projections/ml/`): a two-stage "hurdle" model per position
  (GK/DEF/MID/FWD) -- a classifier for P(plays this gameweek at all), and a
  regressor for E[points | plays] trained only on rows where the player
  played, combined as `prediction = P(plays) x E[points | plays]`. This is
  the correct decomposition under the law of total expectation for FPL's
  scoring (a player who doesn't play always scores exactly 0), and the
  standard fix for zero-inflated data like this (a large spike of exact
  zeros from an unrelated cause -- rotation/injury -- rather than "played
  and had a bad game"). Trained on `HistGradientBoosting{Classifier,Regressor}`
  over rolling 3/5/10-gameweek form + underlying stats (xG, xA, ICT, bps,
  ...) + team attack/defence strength, all leak-free (rolling stats use
  `.shift(1)` before the window, so a gameweek's features never include
  that gameweek's own result -- see `tests/test_ml_features.py`).

**Walk-forward backtest** (`fpl-automate backtest`, or the weekly
`train-model.yml` workflow): the ML model trains on every season except the
most recently *completed* one and is evaluated only on that held-out
season; the baseline is scored on the exact same held-out rows by
reconstructing, for every (player, gameweek), a point-in-time `Player`
built only from that player's strictly-prior season-to-date stats, then
calling the real, unmodified `baseline_model.project_player` on it -- not a
re-derived approximation. Full methodology and current numbers:
[reports/model_backtest.md](reports/model_backtest.md). It also checks
calibration directly (docs/ROADMAP.md Phase 2 asks: "are floor/ceiling
bands actually capturing the real range of outcomes?") -- on the current
backtest, the baseline's bands captured the real outcome far less than its
own stated confidence would suggest, a genuine finding this backtest exists
specifically to surface, not something to paper over.

**Which model runs live:** `PROJECTION_MODEL` in `.env` (default `ml`) --
the ML model wherever it covers a player (falls back to the baseline per
player otherwise, e.g. a brand-new team the model has no history for), or
`baseline` to force the hand-built model only. Every `PlayerProjection`'s
rationale states which one actually produced it, so this is never a silent
switch either way. Live inference reuses the *current* season's own
archive on the same public dataset the historical training data comes from
(`ml/live.py`) -- since that archive updates gameweek by gameweek as real
results land, it doubles as free per-gameweek rolling history for
in-progress predictions, without hundreds of extra FPL API calls per run.

**Production score-tracking**, distinct from the offline backtest above:
every `run-weekly-plan` logs the owned squad's projections, and
automatically reconciles past ones against real results once their
gameweek finishes (`workflow.reconcile_outcomes`, run at the start of every
`run-weekly-plan`). `fpl-automate show-model-performance` shows the live
MAE/bias this has actually produced in production, gameweek by gameweek --
the running, real-world complement to the offline backtest.

## Risk

Every projection already carries `expected_points`, `floor_points`,
`ceiling_points`, and `confidence` -- `risk/` turns that into two things:
a plain-language classification, and a genuine risk-adjusted optimizer.

**Safe / balanced / risky classification** (`risk/classification.py`):
every squad player (and every transfer candidate) is labelled by
*coefficient of variation* -- the projection's implied standard deviation
(derived from its floor/ceiling band, treated as an ~80% interval; exact
by construction for the ML model, an approximation for the baseline --
see the module docstring) divided by its expected points, so a
low-scoring erratic player and a high-scoring erratic player are compared
on the same relative scale rather than raw point spread. Qualitative
signals (`injury_or_availability_doubt`, `rotation_risk`,
`blank_gameweek`, `small_sample`) can only push a classification *up* in
risk, never down -- a numerically tight spread on a player with a live
injury doubt still comes out "risky". Shown in `analyse-squad`, every
suggested transfer, and the weekly report's "Squad risk profile" section.

**Mean-variance risk-adjusted optimization** (`risk/portfolio.py`,
`optimization/lineup.py`'s `risk_adjusted` strategy): a genuine
Markowitz-style objective, not just another single-number substitution
like the other three strategies (which just swap in floor/expected/
ceiling as the thing to maximise). The objective is
`expected_points - risk_aversion * variance` per player, with
`risk_aversion` a user-set dial (`RISK_AVERSION` in `.env`, default `1.0`
-- `0` is mathematically identical to `balanced`; higher values
increasingly favour lower-variance players and captaincy picks over
higher-mean but more volatile ones, verified in
`tests/test_optimization.py`). Captaincy gets its own, mathematically
correct treatment: doubling a player's points quadruples their variance
(`Var(2X) = 4·Var(X)`), so the marginal value of captaining player *c* is
`mu_c - 3·risk_aversion·sigma_c²`, not simply double their normal-starter
score -- a naive "just double it" implementation would under-penalise
volatile captaincy picks by a factor of 3, and this optimizer can (and,
verified against real data, does) pick a different captain than
`balanced` even when the two strategies choose an identical starting XI.

Documented, honest simplification (see `risk/portfolio.py`'s docstring):
player variances are treated as independent, so a squad's total variance
is just the sum of its players' variances -- real players are correlated
(two from the same match share outcome risk), and a covariance-aware
version would need a covariance matrix this project doesn't yet estimate
from historical results (real future work, not attempted here). At very
high `risk_aversion` the quadratic captaincy penalty can concentrate the
optimizer on an oddly "boring" pick (e.g. a nailed but low-ceiling
goalkeeper) over a much higher-scoring attacker -- correct minimum-variance
behaviour at that extreme, not a bug, but a sign the dial is set higher
than most managers would actually want; the default (`1.0`) does not do
this on real data (verified in the checks above).

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
| `PROJECTION_MODEL` | No (default `ml`) | `ml` (trained model, baseline fallback per player) or `baseline` (hand-built model only) -- see **Model** |
| `RISK_AVERSION` | No (default `1.0`) | Mean-variance risk dial for the `risk_adjusted` strategy, `0` = same as `balanced` -- see **Risk** |
| `DATABASE_URL` | No | SQLite path |
| `EMAIL_NOTIFICATIONS_ENABLED` + `SMTP_*` / `EMAIL_*` | No | Email alerts |

## Running it

```bash
fpl-automate health-check          # config, database, FPL API reachability
fpl-automate fetch-data            # pull + persist current player/fixture data
fpl-automate validate-data         # fetch + run data-quality checks only
fpl-automate analyse-squad         # your current squad, bank, free transfers, chips, risk profile
fpl-automate recommend-transfers   # ranked transfer scenarios (roll vs. 1/2 transfers), risk-tagged
fpl-automate optimise-lineup       # best XI/bench/captaincy for your CURRENT squad
fpl-automate run-weekly-plan       # full pipeline -> writes reports/gwN_<timestamp>.md/.json
fpl-automate show-history          # past decisions + recorded outcomes
fpl-automate show-model-performance  # live projection accuracy (MAE/bias) from reconciled results
fpl-automate fetch-historical-data  # downloads/refreshes cached multi-season training data
fpl-automate backtest               # trains + walk-forward backtests the ML model -- see Model
```

Add `--strategy conservative|balanced|aggressive|risk_adjusted` to
`recommend-transfers`/`optimise-lineup`/`run-weekly-plan` (default
`balanced`). `--risk-aversion` (on `recommend-transfers`/`optimise-lineup`
only; `run-weekly-plan` uses `RISK_AVERSION` from `.env`) overrides the
dial for `risk_adjusted` -- see **Risk**.

Not yet implemented (see `docs/ROADMAP.md`): `bursary-search`,
`bursary-score`, `generate-checklist` -- these exist as CLI stubs that say so
rather than silently doing nothing.

## Windows executables (.exe)

If you don't want to install Python, `.github/workflows/build-windows-exe.yml`
builds **two** standalone executables on a real Windows machine (PyInstaller
can't cross-compile, so this has to run on GitHub's `windows-latest` runners,
not in a Linux dev environment), each bundling the CBC solver PuLP needs for
lineup optimisation, so nothing else needs installing:

- **`fpl-automate-gui.exe`** -- a desktop app with a **Settings tab** (a form
  for your team ID, email/SMTP details, etc. -- saves to a config file for
  you, you never open or edit that file yourself) and a **Dashboard tab**
  (buttons for each action, results shown inline). Start here if you'd
  rather not touch a terminal or text file at all.
- **`fpl-automate.exe`** -- the command-line version, for running specific
  commands, scripting, or scheduling.

**Getting them:** open the repo's GitHub Actions tab -> "Build Windows
Executables" -> "Run workflow". When it finishes, download the
`fpl-automate-gui-windows` and/or `fpl-automate-windows` artifacts and unzip
them. (Push a tag like `v0.1.0` instead of running it manually and it'll
also attach both exes to a GitHub Release, which gives you a permanent
download link instead of an Actions artifact that expires.)

**Using the GUI:** put `fpl-automate-gui.exe` in its own folder (it creates
`.env`, `data/`, and `reports/` next to itself -- keep it out of `Downloads`
clutter). Double-click it. On the **Settings** tab, fill in your FPL Team ID
(and, if you want alerts, tick "Enable email notifications" and fill in your
SMTP details -- there's a "Send Test Email" button to check it works) and
click **Save Settings**. Switch to the **Dashboard** tab, pick a strategy,
and click any action (Health Check, Analyse Squad, Recommend Transfers,
Optimise Lineup, Run Weekly Plan) -- the result appears in the panel below.
"Open Reports Folder" jumps straight to the saved report files.

**Using the CLI exe:** put `fpl-automate.exe` in its own folder the same
way. Double-click it once: it creates a starter `.env` and tells you to
edit it (or just use the GUI instead, which manages the same file for you),
then stops. Edit the file, then run it again -- it runs whatever you last
configured, prints the result, and waits for Enter before closing. For
specific commands (`analyse-squad`, `recommend-transfers --strategy
aggressive`, etc.), open a terminal (cmd or PowerShell), `cd` into the
folder, and run `fpl-automate.exe <command>` the same way you would the
Python CLI -- every command in "Running it" above works identically.

Same safety model as everywhere else in this project, for both: they only
read public FPL data and write a report. Neither logs into FPL, and neither
touches your squad -- "Send Test Email" only ever sends a test email, never
a transfer.

## Scheduling

`.github/workflows/weekly-plan.yml` runs `run-weekly-plan` automatically
(twice daily, plus on-demand via the Actions tab "Run workflow" button) and
commits the resulting report + decision-history database back to the repo,
so your weekly plans and their eventual outcomes are visible in git history.

`.github/workflows/train-model.yml` is separate and runs weekly (plus
on-demand): it fetches historical data, retrains the ML model, walk-forward
backtests it, and commits `models/*.joblib` + `reports/model_backtest.*` --
deliberately not part of `weekly-plan.yml` itself, so a retrain doesn't
happen on every single twice-daily run (see **Model** above).

To enable weekly-plan.yml:
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
- **Windows .exe, no Python needed.** See "Windows executables (.exe)" above --
  built by `.github/workflows/build-windows-exe.yml` on a real Windows
  runner, since PyInstaller can't cross-compile from Linux/macOS.
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

147 tests as of this writing, covering: the FPL client's retry/cache/error
handling (via `responses`-mocked HTTP), the data-validation gate, feature
engineering (form shrinkage, fixture windows, minutes reliability), the
baseline projection model (blank/double gameweeks, injury handling,
captaincy), the lineup ILP optimiser (formation constraints, strategy-driven
captaincy, and the risk-adjusted strategy's captaincy math specifically),
the transfer engine (budget/club-limit enforcement, hit thresholds, risk
tagging), free-transfer ledger simulation, sell-value/price-tax logic,
SQLite persistence, the risk classification/mean-variance layer (tier
overrides from qualitative risk flags, the quadratic captaincy-variance
formula verified against a naive "just double it" implementation), report
rendering, and the ML model layer specifically: leak-free rolling
features (a gameweek's own result never leaks into its own features, and a
season boundary resets rolling history even when FPL recycles element IDs),
the hurdle model's classifier/regressor split, the walk-forward backtest
(including a regression test that would have caught a real bug found during
development -- a parameter name shadowing an imported function silently
skipped saving trained models), live inference's fixture/strength handling
and live-availability override, and automatic outcome reconciliation.

## Known limitations

- **The baseline's floor/ceiling bands are poorly calibrated.** The walk-
  forward backtest found actual outcomes land inside the baseline's stated
  band far less often than its own confidence implies (see
  `reports/model_backtest.md`) -- a real, surfaced finding, not a hidden
  flaw. Its hand-set coefficients otherwise remain documented in
  `projections/baseline_model.py`.
- **The ML model has no real-time injury/team-news signal of its own**,
  same structural gap as the baseline without it -- it only ever sees
  historical minutes/starts. `ml/live.py` closes this specifically for the
  *immediate* next gameweek using FPL's own live status/chance-of-playing
  fields (the same signal the baseline already used); it does not extend to
  the 3/5-gameweek horizon, since current injury status says little that
  far out.
- **Live ML inference depends on vaastav's current-season archive being
  up to date.** It updates gameweek by gameweek as real results land, but
  can occasionally lag the live game by a few results -- if so, rolling
  features for the newest gameweeks are thinner than they'll be once the
  archive catches up (the model still runs, just with less current-season
  history to roll over; see `games_played_so_far` and the `small_sample`
  risk flag). Outcome reconciliation (`show-model-performance`) has the
  same dependency and simply retries later if a gameweek's actual result
  isn't in the archive yet.
- **Per-gameweek historical availability/injury status and set-piece
  order aren't in the archived training data**, so the backtest's
  point-in-time baseline reconstruction treats every historical player as
  fully available -- a conservative bias in the baseline's favour in that
  comparison, not the ML model's (see `ml/backtest.py`'s docstring).
- **Sell value can be underestimated** for players still in your very first
  squad who've never been re-bought (no purchase-price record exists via the
  public API for those) -- the fallback assumes no profit, which understates
  your real budget but never overstates it.
- **Free-transfer accounting assumes current (2024/25+) rules** (bank up to
  5, wildcard/free hit don't touch the ledger) and treats each chip as
  single-use for the season, matching your stated situation (wildcard
  already used) -- see the docstring in `squad/state.py`.
- **The baseline's fixture-difficulty windows use an average**, not
  fixture-by-fixture values, across a multi-gameweek horizon (the ML model
  doesn't share this limitation -- it walks each gameweek in the horizon
  individually with that gameweek's real fixture, see **Model**).
- **The two-transfer search is greedy**, not exhaustive: it can miss a
  jointly-optimal pair that isn't optimal individually.
- **The mean-variance optimizer assumes independent player variance**
  (no covariance matrix) -- see **Risk** above; a correlated (same-match)
  version is real future work, not attempted here.
- **This sandboxed development session could not reach
  fantasy.premierleague.com** (blocked by this environment's own network
  policy) -- the full pipeline is validated with 147 tests against
  mocked/fabricated data (including the ML model and the risk-adjusted
  optimizer both exercised against real, fetched historical/current-season
  data directly, bypassing only the live FPL API call itself), but you
  should run `fpl-automate health-check` yourself as the first real
  connectivity check.

## Roadmap

See `docs/ROADMAP.md` for the full phased plan. Backtesting/walk-forward
validation and an ML model benchmarked against the baseline (Phase 2) are
now done -- see **Model** above. Remaining: a dashboard, and the bursary/
scholarship quantitative-decision module.
