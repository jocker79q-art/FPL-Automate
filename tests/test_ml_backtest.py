from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fpl_automate.projections.ml import backtest


def test_metrics_computes_mae_rmse_n():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 6.0])
    m = backtest._metrics(y_true, y_pred)
    assert m["n"] == 4
    assert m["mae"] == pytest.approx(0.5)


def test_pct_better_positive_means_model_wins():
    assert backtest._pct_better(baseline_mae=2.0, model_mae=1.0) == 50.0
    assert backtest._pct_better(baseline_mae=1.0, model_mae=2.0) == -100.0


def test_pct_better_handles_zero_baseline_without_dividing_by_zero():
    assert backtest._pct_better(baseline_mae=0.0, model_mae=1.0) == 0.0


def test_calibration_report_overall_coverage_and_buckets():
    actual = np.array([5.0, 5.0, 5.0, 5.0])
    floor = np.array([4.0, 4.0, 6.0, 4.0])
    ceiling = np.array([6.0, 6.0, 8.0, 6.0])
    confidence = np.array([0.9, 0.9, 0.5, 0.3])

    report = backtest._calibration_report(actual, floor, ceiling, confidence)
    # 3 of 4 rows have actual (5) inside [floor, ceiling]; the third row's
    # band is [6, 8], which does not contain 5.
    assert report["overall_band_coverage"] == pytest.approx(0.75)
    buckets_by_range = {b["confidence_range"]: b for b in report["by_confidence_bucket"]}
    assert buckets_by_range["0.8-1.0"]["band_coverage"] == pytest.approx(1.0)
    assert buckets_by_range["0.4-0.6"]["band_coverage"] == pytest.approx(0.0)


def test_render_report_markdown_states_ml_win_when_ml_actually_wins():
    results = _fake_results(ml_mae=0.8, baseline_mae=1.2)
    md = backtest.render_report_markdown(results)
    assert "beats the hand-coded baseline" in md
    assert "still beats the ML model" not in md


def test_render_report_markdown_states_baseline_win_when_baseline_actually_wins():
    results = _fake_results(ml_mae=1.5, baseline_mae=1.0)
    md = backtest.render_report_markdown(results)
    assert "still beats the ML model" in md
    assert "beats the hand-coded baseline by" not in md


def _fake_results(ml_mae: float, baseline_mae: float) -> dict:
    return {
        "generated_at": "2026-01-01T00:00:00Z",
        "seasons_trained_on": ["2022-23", "2023-24"],
        "test_season": "2024-25",
        "feature_columns": [],
        "positions": {
            "MID": {
                "ml_model": {"mae": ml_mae, "rmse": ml_mae * 1.5, "n": 100},
                "baseline_model": {"mae": baseline_mae, "rmse": baseline_mae * 1.5, "n": 100},
                "fpl_xp": {"mae": 1.1, "rmse": 1.6, "n": 100},
                "naive_form": {"mae": 1.2, "rmse": 1.7, "n": 100},
                "n_train": 400,
            }
        },
        "overall": {
            "ml_model": {"mae": ml_mae, "rmse": ml_mae * 1.5, "n": 100},
            "baseline_model": {"mae": baseline_mae, "rmse": baseline_mae * 1.5, "n": 100},
            "fpl_xp": {"mae": 1.1, "rmse": 1.6, "n": 100},
            "naive_form": {"mae": 1.2, "rmse": 1.7, "n": 100},
        },
        "ml_calibration": {"MID": {"residual_p10": -1.0, "residual_p90": 1.0}},
        "calibration": {"overall_band_coverage": 0.5, "by_confidence_bucket": []},
    }


# --- full pipeline integration test ----------------------------------------


_TEAMS_CSV_HEADER = (
    "id,name,short_name,strength_overall_home,strength_overall_away,"
    "strength_attack_home,strength_attack_away,strength_defence_home,strength_defence_away\n"
)


def _write_season(season_dir: Path, players: list[int], gameweeks: list[int]) -> None:
    season_dir.mkdir(parents=True)

    (season_dir / "teams.csv").write_text(
        _TEAMS_CSV_HEADER
        + "10,Alpha,ALP,1100,1100,1300,1200,1200,1100\n"
        + "20,Beta,BET,1100,1100,1250,1150,1150,1050\n"
    )
    fixture_rows = "\n".join(
        f"{gw},{gw},10,20,2,3,{'True' if gw < max(gameweeks) else 'False'}" for gw in gameweeks
    )
    (season_dir / "fixtures.csv").write_text(
        "id,event,team_h,team_a,team_h_difficulty,team_a_difficulty,finished\n" + fixture_rows + "\n"
    )

    header = (
        "name,element,GW,position,team,opponent_team,was_home,value,total_points,minutes,starts,"
        "ict_index,influence,creativity,threat,bps,expected_goals,expected_assists,"
        "expected_goal_involvements,expected_goals_conceded,goals_scored,assists,clean_sheets,"
        "saves,bonus,goals_conceded,xP\n"
    )
    lines = [header]
    rng = np.random.default_rng(hash(str(season_dir)) % (2**32))
    for element in players:
        for gw in gameweeks:
            played = rng.integers(0, 2)
            minutes = 90 if played else 0
            points = int(rng.poisson(3)) if played else 0
            lines.append(
                f"Player {element},{element},{gw},MID,Alpha,20,True,55,{points},{minutes},{played},"
                f"10.0,10.0,10.0,10.0,20,0.5,0.3,0.8,1.0,0,0,0,0,1,0,3.0\n"
            )
    (season_dir / "merged_gw.csv").write_text("".join(lines))


def test_run_backtest_trains_scores_and_saves_models_and_calibration(tmp_path: Path):
    hist_dir = tmp_path / "hist"
    _write_season(hist_dir / "2023-24", players=[1, 2, 3, 4, 5], gameweeks=[1, 2, 3, 4, 5, 6])
    _write_season(hist_dir / "2024-25", players=[1, 2, 3, 4, 5], gameweeks=[1, 2, 3, 4, 5, 6])

    models_dir = tmp_path / "models"
    results = backtest.run_backtest(
        hist_dir,
        seasons=["2023-24", "2024-25"],
        test_season="2024-25",
        save_models=True,
        models_dir=models_dir,
    )

    assert "MID" in results["positions"]
    assert results["overall"]["ml_model"]["n"] > 0
    assert 0.0 <= results["calibration"]["overall_band_coverage"] <= 1.0

    # Regression guard for a real bug caught during development: the
    # `save_models` bool parameter once shadowed the imported `save_models`
    # function, so models were silently never written despite save_models=True.
    assert (models_dir / "MID.joblib").exists()
    assert (models_dir / "ml_calibration.json").exists()


def test_run_backtest_and_report_writes_report_files(tmp_path: Path):
    hist_dir = tmp_path / "hist"
    _write_season(hist_dir / "2023-24", players=[1, 2, 3, 4, 5], gameweeks=[1, 2, 3, 4, 5, 6])
    _write_season(hist_dir / "2024-25", players=[1, 2, 3, 4, 5], gameweeks=[1, 2, 3, 4, 5, 6])

    reports_dir = tmp_path / "reports"
    results = backtest.run_backtest_and_report(
        hist_dir,
        reports_dir,
        seasons=["2023-24", "2024-25"],
        test_season="2024-25",
        save_models=False,
    )

    assert (reports_dir / "model_backtest.md").exists()
    assert (reports_dir / "model_backtest.json").exists()
    md = (reports_dir / "model_backtest.md").read_text()
    assert "Projection models: walk-forward backtest report" in md
    assert str(results["overall"]["ml_model"]["mae"]) in md


def test_run_backtest_rejects_test_season_not_in_seasons(tmp_path: Path):
    with pytest.raises(ValueError):
        backtest.run_backtest(tmp_path, seasons=["2023-24"], test_season="2024-25")
