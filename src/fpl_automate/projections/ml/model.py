"""Two-stage "hurdle" ML projection model: a classifier for P(plays this
gameweek at all), and a regressor for E[points | plays], combined as
`prediction = P(plays) x E[points | plays]`.

Why a hurdle model and not one regressor over everything: FPL points data
is zero-inflated -- a large spike of exact-zero rows caused by rotation
or injury, a phenomenon *unrelated* to the scoring distribution for
players who did play. A single regressor asked to fit both the
play/no-play split and the scoring distribution at once does a mediocre
job of both. The standard fix is a hurdle model, which is what this is:

  1. A classifier predicts P(plays this gameweek at all).
  2. A regressor predicts E[points | plays], trained *only* on rows
     where the player played, so rotation/injury zeros (a completely
     different phenomenon from "played but scored 0") don't drag the
     learned scoring distribution toward zero.
  3. Combined as `prediction = P(plays) x E[points | plays]` -- the
     correct decomposition under the law of total expectation, since
     E[points | didn't play] is always exactly 0 in FPL's scoring.

Floor/ceiling uncertainty is handled the same two-stage way: two more
regressors predict the 10th/90th percentile of E[points | plays] (real
quantile regression, `loss="quantile"`, conditioned on that player's own
features -- not a single global band applied to everyone at a position),
and `mixture_floor_ceiling` folds the P(plays) uncertainty back in on top
of that, since a player who might not play at all has a very different
low end to their outcome distribution than one who reliably starts.

See `ml/backtest.py` for the walk-forward evaluation of this
architecture against `projections/baseline_model.py`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from fpl_automate.projections.ml.features import PLAYED_COL, TARGET_COL, feature_columns
from fpl_automate.storage.models import Position

POSITIONS = ["GK", "DEF", "MID", "FWD"]

# vaastav's historical data (and this module's own POSITIONS list) uses
# short position codes that differ from Position.short's FPL-website-style
# labels ("GKP" not "GK") -- this is the one place that mismatch is
# resolved, so nothing downstream has to know about it.
POSITION_TO_CODE: dict[Position, str] = {
    Position.GOALKEEPER: "GK",
    Position.DEFENDER: "DEF",
    Position.MIDFIELDER: "MID",
    Position.FORWARD: "FWD",
}

# The 10th/90th percentiles of E[points | plays], not of total points --
# see `mixture_floor_ceiling` below for how P(plays) uncertainty is
# folded back in on top of these.
LOW_QUANTILE = 0.1
HIGH_QUANTILE = 0.9


def _make_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=42,
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
    )


def _make_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
    )


def _make_quantile_regressor(quantile: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        random_state=42,
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
    )


class PositionModel:
    """One position's fitted classifier + mean regressor + two quantile
    regressors (10th/90th percentile of E[points | plays]), plus the exact
    feature-column order they were trained on (persisted alongside the
    models so a later feature-list change can't silently misalign
    columns at prediction time).

    The quantile regressors give each *player* their own predicted spread
    (a nailed-on high-minutes forward's band is narrower than a rotation
    risk's, because the model conditions on that player's own features)
    -- a real improvement over a single global (residual p10/p90)
    position-wide offset applied uniformly to everyone."""

    def __init__(
        self,
        classifier: HistGradientBoostingClassifier,
        regressor: HistGradientBoostingRegressor,
        regressor_low: HistGradientBoostingRegressor,
        regressor_high: HistGradientBoostingRegressor,
        feature_cols: list[str],
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.regressor_low = regressor_low
        self.regressor_high = regressor_high
        self.feature_cols = feature_cols

    def predict(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Returns (p_play, points_given_played) for each row of X."""
        cols = X[self.feature_cols]
        p_play = self.classifier.predict_proba(cols)[:, 1]
        points_given_played = self.regressor.predict(cols)
        return pd.Series(p_play, index=X.index), pd.Series(points_given_played, index=X.index)

    def predict_points(self, X: pd.DataFrame) -> pd.Series:
        p_play, points_given_played = self.predict(X)
        return p_play * points_given_played

    def predict_quantiles(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Returns (points_given_played_p10, points_given_played_p90) --
        this player's own predicted low/high scoring outcomes conditional
        on playing, not yet combined with P(plays). See
        `mixture_floor_ceiling` for that combination."""
        cols = X[self.feature_cols]
        low = self.regressor_low.predict(cols)
        high = self.regressor_high.predict(cols)
        # Quantile regressors are fit independently, so crossing (low >
        # high) is possible in principle on out-of-distribution rows;
        # enforce the ordering rather than surface a nonsensical band.
        low, high = np.minimum(low, high), np.maximum(low, high)
        return pd.Series(low, index=X.index), pd.Series(high, index=X.index)


def mixture_floor_ceiling(p_play: float, pgp_low: float, pgp_high: float) -> tuple[float, float]:
    """The 10th/90th percentile of *total* points (not just points given
    played) under the hurdle model's own two-part mixture: with
    probability (1 - p_play) the player doesn't play and scores exactly
    0; with probability p_play they play and score from the distribution
    `predict_quantiles` describes.

    Floor is exact (not an approximation) whenever p_play < 0.9: the
    not-played point-mass at 0 alone already accounts for at least 10% of
    the outcome distribution, so 0 *is* the true 10th percentile
    regardless of what the played-distribution looks like. Only once
    p_play >= 0.9 does the 10th percentile fall inside the played
    distribution -- at the conditional level (0.1 - (1-p_play)) / p_play,
    which converges to 0.1 (i.e. exactly `pgp_low`) as p_play -> 1.
    `p_play * pgp_low` is used as a practical stand-in for that shifted
    level in the p_play >= 0.9 regime -- exact in the limit, a slight
    under-estimate of the true floor otherwise (conservative, not
    optimistic).

    Ceiling is the mirror case: exactly 0 whenever p_play <= 0.1 (the
    not-played mass alone already exceeds the 90th percentile threshold),
    and `p_play * pgp_high` otherwise -- again exact only as p_play -> 1,
    an approximation (in the same conservative direction) elsewhere."""
    floor = 0.0 if p_play < 0.9 else max(0.0, p_play * pgp_low)
    ceiling = 0.0 if p_play <= 0.1 else max(0.0, p_play * pgp_high)
    return floor, max(floor, ceiling)


def fit_position_model(train_df: pd.DataFrame) -> PositionModel:
    """Fits one position's hurdle model on already-feature-engineered rows."""
    cols = feature_columns()
    X_train = train_df[cols]
    y_train = train_df[TARGET_COL]
    played_train = (train_df[PLAYED_COL] > 0).astype(int)

    classifier = _make_classifier()
    classifier.fit(X_train, played_train)

    played_mask = played_train.to_numpy(dtype=bool)
    X_played, y_played = X_train[played_mask], y_train[played_mask]

    regressor = _make_regressor()
    regressor.fit(X_played, y_played)

    regressor_low = _make_quantile_regressor(LOW_QUANTILE)
    regressor_low.fit(X_played, y_played)

    regressor_high = _make_quantile_regressor(HIGH_QUANTILE)
    regressor_high.fit(X_played, y_played)

    return PositionModel(classifier, regressor, regressor_low, regressor_high, cols)


def save_models(models: dict[str, PositionModel], models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for position, model in models.items():
        joblib.dump(
            {
                "classifier": model.classifier,
                "regressor": model.regressor,
                "regressor_low": model.regressor_low,
                "regressor_high": model.regressor_high,
                "feature_columns": model.feature_cols,
            },
            models_dir / f"{position}.joblib",
        )


def load_models(models_dir: Path) -> dict[str, PositionModel]:
    """Loads whatever per-position .joblib bundles are present. Never
    raises for a missing directory/file -- callers should treat an empty
    (or partial) result as "fall back to the baseline model for the
    missing position(s)", not as an error: a fresh checkout before
    `train-model` has ever run is a normal, expected state."""
    models: dict[str, PositionModel] = {}
    if not models_dir.exists():
        return models
    for position in POSITIONS:
        path = models_dir / f"{position}.joblib"
        if not path.exists():
            continue
        bundle = joblib.load(path)
        models[position] = PositionModel(
            bundle["classifier"],
            bundle["regressor"],
            bundle["regressor_low"],
            bundle["regressor_high"],
            bundle["feature_columns"],
        )
    return models
