# Roadmap

Phase 1 (this delivery) is the MVP: reliable data collection, squad
analysis, hit-aware transfer recommendations, lineup optimisation, weekly
reports, and dry-run safety by construction (no execution code exists at
all yet).

## Phase 2 -- Backtesting & model validation

- Walk-forward backtest: for each past gameweek, compute what the model
  *would* have projected using only data available before that gameweek,
  compare to actual points. No future information may leak into a
  historical prediction.
- Track calibration: are 70%-confidence projections right about 70% of the
  time? Are floor/ceiling bands actually capturing the real range of
  outcomes?
- Tune the baseline model's documented coefficients (e.g. the 55/45
  empirical/component blend, the fixture-difficulty coefficient) against
  backtest results, keeping every coefficient documented and explainable.
- Only after a baseline backtest exists: evaluate whether a learned model
  (gradient boosting on the same interpretable features) beats the
  baseline on held-out gameweeks. If so, ship it *alongside* the baseline
  with an A/B comparison, not as a silent replacement.

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
