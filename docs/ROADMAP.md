# Roadmap

Phase 1 (this delivery) is the MVP: reliable data collection, squad
analysis, hit-aware transfer recommendations, lineup optimisation, weekly
reports, and dry-run safety by construction (no execution code exists at
all yet).

## Phase 2 -- Backtesting & model validation (done)

- ~~Walk-forward backtest: for each past gameweek, compute what the model
  *would* have projected using only data available before that gameweek,
  compare to actual points. No future information may leak into a
  historical prediction.~~ Done: `projections/ml/backtest.py`
  (`fpl-automate backtest`, run weekly by `train-model.yml`). The baseline
  is scored on the exact same held-out rows via point-in-time-reconstructed
  players and the real, unmodified `baseline_model.project_player` -- not a
  re-derived approximation. Current numbers: `reports/model_backtest.md`.
- ~~Track calibration: are 70%-confidence projections right about 70% of the
  time? Are floor/ceiling bands actually capturing the real range of
  outcomes?~~ Done -- and the honest answer for the baseline, on the
  current backtest, is "not very well" (see `reports/model_backtest.md`'s
  calibration section): actual outcomes land inside its stated
  floor/ceiling band far less often than its own confidence would imply.
  A genuine finding, surfaced rather than tuned away.
- Tune the baseline model's documented coefficients (e.g. the 55/45
  empirical/component blend, the fixture-difficulty coefficient) against
  backtest results, keeping every coefficient documented and explainable.
  **Not done** -- the backtest infrastructure to do this now exists, but no
  coefficient has actually been retuned; the baseline's numbers are still
  its original hand-set values.
- ~~Only after a baseline backtest exists: evaluate whether a learned model
  (gradient boosting on the same interpretable features) beats the
  baseline on held-out gameweeks. If so, ship it *alongside* the baseline
  with an A/B comparison, not as a silent replacement.~~ Done:
  `projections/ml/` is a two-stage hurdle model (`HistGradientBoosting`
  classifier + regressor), beats the baseline on the current backtest (see
  `reports/model_backtest.md`), and is wired in as the *default* live
  model (`PROJECTION_MODEL=ml`) with automatic per-player fallback to the
  baseline -- never a silent substitution, since every projection's
  rationale states which model produced it. Production score-tracking
  (`workflow.reconcile_outcomes`, `fpl-automate show-model-performance`) is
  the live, ongoing complement to this offline backtest.

## Phase 2.5 -- Risk classification & mean-variance optimization (done, not originally scoped)

Not part of the original phased plan above -- added because a projection
model producing `expected_points` alone tells you what to expect, not how
much to trust it, and the roadmap's own calibration finding (Phase 2) made
that gap concrete rather than theoretical.

- ~~Classify every player (squad and transfer candidates) as safe/
  balanced/risky.~~ Done: `risk/classification.py`, using coefficient of
  variation from the projection's own floor/ceiling band, with qualitative
  risk flags (injury doubt, rotation risk, blank gameweek, small sample)
  as a hard floor that can only push the classification *up*, never down.
  Shown in `analyse-squad`, every suggested transfer, and the weekly
  report's "Squad risk profile" section.
- ~~A genuine risk-adjusted optimizer, not just another single-number
  substitution like conservative/balanced/aggressive.~~ Done:
  `risk/portfolio.py` + `optimization/lineup.py`'s `risk_adjusted`
  strategy -- a real mean-variance (Markowitz-style) objective,
  user-tunable via `RISK_AVERSION`, with a mathematically correct
  (quadratic, not linear) variance treatment for the captaincy multiplier
  specifically -- verified against a naive "just double it" alternative
  in `tests/test_risk_portfolio.py`.
- ~~Player variance is treated as independent (no covariance matrix), so
  correlated risk between players in the same match isn't modelled.~~
  **Partially done**: `risk/covariance.py` now estimates real same-team
  and same-fixture-opponent correlation from backtest residuals (a
  Ledoit-Wolf-style shrinkage-to-zero estimator over a pooled,
  two-parameter structure -- see the module docstring for why a full
  per-player-pair matrix isn't attempted), and `portfolio_variance`
  computes a squad's *true* variance with it, surfaced in the weekly
  report. **Still not done**: the ILP lineup *optimizer* still selects
  players using the independent-variance objective -- a genuinely
  covariance-aware selection needs a quadratic objective, which PuLP's
  linear ILP solver doesn't support. The reported risk figure is honest
  and covariance-aware; the selection algorithm is not yet.
- ~~Does the risk-adjusted strategy's theoretical claim (lower
  risk_aversion volatility) actually hold against real outcomes, not just
  its own objective function?~~ Done: `risk/validation.py` runs the real
  optimizer on synthetic squads across the held-out backtest season,
  scored by what players *actually* scored (`reports/risk_validation.md`).
  On the current backtest: yes, realized variance decreases monotonically
  as `risk_aversion` increases, at a real measured cost in mean points --
  reported honestly either way, not smoothed over if a future backtest
  disagrees.
- ~~Real per-player predictive uncertainty, not a single global
  floor/ceiling offset per position.~~ Done: `ml/model.py`'s quantile
  regressors (10th/90th percentile of E[points|plays]) plus
  `mixture_floor_ceiling` folding P(plays) uncertainty back in --
  measurably better calibrated than the flat-offset band it replaced (90%
  empirical coverage vs. an 80% target, see `reports/model_backtest.md`).
- ~~Transfer recommendations only ever say *which* player, never *when*.~~
  Done, for single-transfer scenarios: `transfers/planning.py` compares
  acting now against a short delay using real per-gameweek (not
  horizon-summed) projections, so a genuine fixture swing can change the
  recommended timing. **Not done**: multi-transfer sequences, and chip
  (Wildcard/Free Hit/Bench Boost/Triple Captain) timing -- both real
  future work.

## Phase 3 -- Notifications, richer late-news detection, dashboard

- Detect *changes* between consecutive runs (a status flip, a price
  change, a new news string) and alert specifically on those, not just
  "here is this week's report."
- A read-only web dashboard (squad view, fixture calendar, projected
  points, transfer/captaincy comparison, risk indicators, historical
  decisions/outcomes) -- likely a small static site reading the
  already-committed JSON reports, to avoid standing up a server.
- Optional additional notification channels (Discord webhook, Telegram) as
  alternatives to email.

## Phase 4 -- Quantitative applications module (bursaries/scholarships)

Reuses the shared quantitative primitives this project already has
(constrained optimisation, expected-value/risk scoring, scenario
comparison, audit logging) for a structurally similar but distinct
decision problem:

- Structured opportunity import (deadline, eligibility criteria, award
  value, required documents).
- Transparent eligibility scoring and ranking by fit/value/deadline
  urgency/estimated probability of success.
- Checklist generation and application-status tracking; duplicate/conflict
  detection across applications.
- Hard rule carried over unchanged from the FPL module's own safety model:
  **never fabricate information, never invent achievements/grades/income,
  never auto-submit anything** -- drafts and checklists only, always
  requiring explicit human review.

## Phase 5 -- (Only if explicitly requested) Execution layer

Not started, and not implied by anything built so far. Would require:
designing FPL login/session handling with the same rigor as a
credential-bearing integration deserves, a dedicated security review, and
your explicit go-ahead given the materially larger trust surface versus
everything else in this project (which needs zero credentials today).
