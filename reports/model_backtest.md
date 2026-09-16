# Projection models: walk-forward backtest report

Generated: 2026-09-16T19:42:21.938450+00:00
ML model trained on: 2021-22, 2022-23, 2023-24, 2024-25
Backtested on (held out entirely from ML training): 2025-26

MAE = mean absolute error in points per player-gameweek (lower is better).
The hand-coded baseline has no training data to hold out -- it is scored
on the same held-out season using only point-in-time reconstructed
player state (see this module's docstring for exactly how).

## Overall

| Model | MAE | RMSE | n |
|---|---|---|---|
| ML model (two-stage hurdle) | 0.9805 | 1.9574 | 29757 |
| Hand-coded baseline | 1.2524 | 2.1268 | 29757 |
| FPL's own official xP | 1.0701 | 2.3723 | 29757 |
| Naive rolling-5-GW form | 1.0477 | 2.1275 | 29757 |

The ML model beats the hand-coded baseline by +21.7% MAE on this backtest (+8.4% vs FPL's own xP).

## By position

| Position | ML MAE | Baseline MAE | xP MAE | ML vs baseline | n (train / test) |
|---|---|---|---|---|---|
| GK | 0.6114 | 0.8463 | 0.7257 | +27.8% | 11882 / 3427 |
| DEF | 1.1112 | 1.3895 | 1.172 | +20.0% | 36560 / 9733 |
| MID | 0.9496 | 1.2301 | 1.0608 | +22.8% | 47034 / 13310 |
| FWD | 1.1029 | 1.3606 | 1.1645 | +18.9% | 13383 / 3287 |

## ML model calibration: does the quantile-regression band capture reality?

Floor/ceiling come from per-player quantile regression (10th/90th percentile of E[points|plays], `ml/model.py`'s `predict_quantiles`) combined with the hurdle model's own P(plays) via `mixture_floor_ceiling` -- a real per-player band, not a single global offset applied to every player at a position.

Overall: actual points fell inside the ML model's [floor, ceiling] band **90%** of the time (target: ~80%, since this is nominally an 80% central interval).

By P(plays) used for that gameweek (bucketed the same way as the
baseline's confidence below, for direct comparison):

| P(plays) range | n | Band coverage | Mean P(plays) |
|---|---|---|---|
| 0.0-0.2 | 14628 | 97% | 0.04 |
| 0.2-0.4 | 1767 | 88% | 0.28 |
| 0.4-0.6 | 2226 | 85% | 0.51 |
| 0.6-0.8 | 3816 | 84% | 0.71 |
| 0.8-1.0 | 7320 | 79% | 0.91 |

## Baseline calibration: does the floor/ceiling band capture reality?

Overall: actual points fell inside the baseline's [floor, ceiling] band **8%** of the time.

By the baseline's own stated confidence (docs/ROADMAP.md asks: are higher-
confidence projections actually more reliable?):

| Confidence range | n | Band coverage | Mean confidence |
|---|---|---|---|
| 0.4-0.6 | 14834 | 1% | 0.58 |
| 0.6-0.8 | 1638 | 22% | 0.78 |
| 0.8-1.0 | 13285 | 15% | 0.93 |

## Fixture correlation: are same-team/same-match players actually correlated?

`risk/portfolio.py`'s mean-variance optimizer documents an independent-variance simplification (a squad's variance is just the sum of its players' variances). This measures whether that's actually a good approximation, using real backtest residuals -- see `risk/covariance.py` for the full derivation (Ledoit-Wolf-style shrinkage toward 0).

| Relationship | Shrunk correlation (rho) | Gameweek samples |
|---|---|---|
| Same team | +0.030 | 38 |
| Opposing teams, same fixture | -0.035 | 38 |

Same-team correlation of +0.030 means the independence assumption understates real squad variance whenever a lineup concentrates picks from one team (which most managers do, e.g. 2-3 defenders from one in-form defence): `risk/covariance.py`'s `portfolio_variance` accounts for this; `risk/portfolio.py`'s per-player scoring still does not (see docs/ROADMAP.md's known limitations).

## Methodology

- Chronological holdout: the ML model never sees any 2025-26 data during training.
- Two-stage hurdle ML model per position (classifier for P(plays) x regressor for E[points|plays]) -- see `ml/model.py`.
- The baseline is the actual, unmodified `projections.baseline_model.project_player` function, run on point-in-time-reconstructed player state for every test-season row -- not a re-derived approximation.
- Known simplification: per-gameweek historical availability/injury status and set-piece order are not present in the archived dataset, so every reconstructed player is treated as fully available -- this can only help the baseline's accuracy, never the ML model's, so it is a conservative bias if anything.
- Known simplification, shared with the baseline model's own documented assumptions: team strength ratings are a season-level snapshot, not as they stood before each individual gameweek.
