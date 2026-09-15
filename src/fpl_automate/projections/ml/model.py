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

See `ml/backtest.py` for the walk-forward evaluation of this
architecture against `projections/baseline_model.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
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


class PositionModel:
    """One position's fitted classifier + regressor, plus the exact
    feature-column order they were trained on (persisted alongside the
    models so a later feature-list change can't silently misalign
    columns at prediction time)."""

    def __init__(
        self,
        classifier: HistGradientBoostingClassifier,
        regressor: HistGradientBoostingRegressor,
        feature_cols: list[str],
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
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


def fit_position_model(train_df: pd.DataFrame) -> PositionModel:
    """Fits one position's hurdle model on already-feature-engineered rows."""
    cols = feature_columns()
    X_train = train_df[cols]
    y_train = train_df[TARGET_COL]
    played_train = (train_df[PLAYED_COL] > 0).astype(int)

    classifier = _make_classifier()
    classifier.fit(X_train, played_train)

    regressor = _make_regressor()
    played_mask = played_train.to_numpy(dtype=bool)
    regressor.fit(X_train[played_mask], y_train[played_mask])

    return PositionModel(classifier, regressor, cols)


def save_models(models: dict[str, PositionModel], models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for position, model in models.items():
        joblib.dump(
            {
                "classifier": model.classifier,
                "regressor": model.regressor,
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
            bundle["classifier"], bundle["regressor"], bundle["feature_columns"]
        )
    return models


CALIBRATION_FILE = "ml_calibration.json"


def save_calibration(calibration: dict[str, dict[str, float]], models_dir: Path) -> None:
    """Per-position {p10, p90} residual quantiles from the walk-forward
    backtest (`ml/backtest.py`), used at inference time to turn a point
    prediction into an empirical floor/ceiling band -- real observed
    error spread, not an arbitrary heuristic percentage."""
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / CALIBRATION_FILE).write_text(json.dumps(calibration, indent=2), encoding="utf-8")


def load_calibration(models_dir: Path) -> dict[str, dict[str, float]]:
    path = models_dir / CALIBRATION_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
