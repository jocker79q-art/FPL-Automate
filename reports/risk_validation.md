# Risk system validation: does risk_adjusted actually reduce realized variance?

Backtested on: 2025-26 (the same held-out season `ml/backtest.py` uses).

Methodology: for each held-out gameweek, builds a synthetic 15-man squad (top-2 GK / top-5 DEF / top-5 MID / top-3 FWD by that gameweek's ML projection) and runs the real, unmodified `optimization.lineup.optimize_lineup` on it under each strategy -- the *same* squad and projections every time, so only the selection objective varies. Scored by what those players **actually** scored that gameweek (captain doubled), not by the projection -- the only honest test of whether the risk system does what it claims. See `risk/validation.py`'s module docstring for the documented simplification (a synthetic top-N pool, not a real continuous squad under transfer/budget constraints).

| Strategy | Mean realized points | Std dev (realized) | Gameweeks |
|---|---|---|---|
| balanced | 54.24 | 13.41 | 37 |
| risk_adjusted (aversion=0.0) | 54.24 | 13.41 | 37 |
| risk_adjusted (aversion=0.5) | 52.57 | 12.92 | 37 |
| risk_adjusted (aversion=1.0) | 50.95 | 12.1 | 37 |
| risk_adjusted (aversion=2.0) | 50.35 | 11.64 | 37 |
| risk_adjusted (aversion=5.0) | 50.03 | 11.57 | 37 |

## Verdict

Realized std dev decreases monotonically as risk_aversion increases across the tested levels -- matching the theoretical claim on this backtest.

- Lowest realized variance: **risk_adjusted (aversion=5.0)** (std 11.57, mean 50.03).
- Balanced (no risk penalty): std 13.41, mean 54.24.
- Mean realized points at the lowest-variance risk_adjusted setting vs. balanced: -4.21 (the real cost, or lack of one, of choosing lower variance on this backtest).
