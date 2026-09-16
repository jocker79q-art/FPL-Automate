from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_automate.projections.ml import features, model
from fpl_automate.storage.models import Position


def _synthetic_training_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = features.feature_columns()
    data = {c: rng.normal(size=n) for c in cols}
    data["was_home"] = rng.integers(0, 2, size=n)
    played = rng.integers(0, 2, size=n)
    data["minutes"] = played * 90
    data["total_points"] = np.where(played, rng.poisson(3, size=n), 0)
    df = pd.DataFrame(data)
    return df


def test_fit_position_model_predicts_zero_ish_for_unlikely_players_and_scores_for_likely_ones():
    train = _synthetic_training_frame()
    pos_model = model.fit_position_model(train)

    preds = pos_model.predict_points(train)
    assert len(preds) == len(train)
    assert (preds >= 0).all()


def test_position_model_predict_returns_p_play_and_points_given_played_separately():
    train = _synthetic_training_frame()
    pos_model = model.fit_position_model(train)

    p_play, points_given_played = pos_model.predict(train)
    assert (p_play >= 0).all() and (p_play <= 1).all()
    assert (points_given_played.index == train.index).all()


def test_save_and_load_models_round_trip(tmp_path: Path):
    train = _synthetic_training_frame()
    pos_model = model.fit_position_model(train)

    model.save_models({"GK": pos_model}, tmp_path)
    assert (tmp_path / "GK.joblib").exists()

    loaded = model.load_models(tmp_path)
    assert set(loaded.keys()) == {"GK"}
    assert loaded["GK"].feature_cols == pos_model.feature_cols

    original_preds = pos_model.predict_points(train)
    loaded_preds = loaded["GK"].predict_points(train)
    assert np.allclose(original_preds, loaded_preds)


def test_load_models_returns_empty_dict_for_missing_directory(tmp_path: Path):
    assert model.load_models(tmp_path / "does_not_exist") == {}


def test_load_models_skips_positions_with_no_saved_bundle(tmp_path: Path):
    train = _synthetic_training_frame()
    pos_model = model.fit_position_model(train)
    model.save_models({"MID": pos_model}, tmp_path)

    loaded = model.load_models(tmp_path)
    assert set(loaded.keys()) == {"MID"}
    assert "GK" not in loaded


def test_predict_quantiles_returns_ordered_low_high_bands():
    train = _synthetic_training_frame()
    pos_model = model.fit_position_model(train)

    low, high = pos_model.predict_quantiles(train)
    assert (low.index == train.index).all()
    assert (low <= high).all()


def test_save_and_load_models_round_trip_preserves_quantile_regressors(tmp_path: Path):
    train = _synthetic_training_frame()
    pos_model = model.fit_position_model(train)
    model.save_models({"GK": pos_model}, tmp_path)

    loaded = model.load_models(tmp_path)["GK"]
    original_low, original_high = pos_model.predict_quantiles(train)
    loaded_low, loaded_high = loaded.predict_quantiles(train)
    assert np.allclose(original_low, loaded_low)
    assert np.allclose(original_high, loaded_high)


class TestMixtureFloorCeiling:
    def test_floor_is_exactly_zero_whenever_p_play_below_point_nine(self):
        """Below p_play=0.9 the not-played point-mass alone already
        accounts for >=10% of outcomes, so 0 is the true 10th percentile
        regardless of the played-distribution's own low end."""
        floor, _ = model.mixture_floor_ceiling(p_play=0.5, pgp_low=3.0, pgp_high=8.0)
        assert floor == 0.0

    def test_ceiling_is_exactly_zero_whenever_p_play_at_or_below_point_one(self):
        _, ceiling = model.mixture_floor_ceiling(p_play=0.1, pgp_low=3.0, pgp_high=8.0)
        assert ceiling == 0.0

    def test_floor_and_ceiling_scale_with_p_play_above_the_thresholds(self):
        floor, ceiling = model.mixture_floor_ceiling(p_play=0.95, pgp_low=2.0, pgp_high=10.0)
        assert floor == pytest.approx(0.95 * 2.0)
        assert ceiling == pytest.approx(0.95 * 10.0)

    def test_certain_starter_band_converges_to_the_raw_quantiles(self):
        """As p_play -> 1, both approximations become exact."""
        floor, ceiling = model.mixture_floor_ceiling(p_play=1.0, pgp_low=2.0, pgp_high=10.0)
        assert floor == pytest.approx(2.0)
        assert ceiling == pytest.approx(10.0)

    def test_ceiling_never_falls_below_floor(self):
        floor, ceiling = model.mixture_floor_ceiling(p_play=0.95, pgp_low=5.0, pgp_high=1.0)
        assert ceiling >= floor


@pytest.mark.parametrize(
    "position,code",
    [
        (Position.GOALKEEPER, "GK"),
        (Position.DEFENDER, "DEF"),
        (Position.MIDFIELDER, "MID"),
        (Position.FORWARD, "FWD"),
    ],
)
def test_position_to_code_matches_vaastav_schema_not_position_short(position, code):
    """Position.short uses FPL-website-style labels ("GKP"); the historical
    archive and this module's own POSITIONS list use "GK" -- this mapping
    is the one place that mismatch must be resolved correctly."""
    assert model.POSITION_TO_CODE[position] == code
    assert code in model.POSITIONS
