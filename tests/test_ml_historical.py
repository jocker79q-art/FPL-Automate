from __future__ import annotations

from pathlib import Path

import responses

from fpl_automate.projections.ml import historical


def _mock_season(season: str, missing: set[str] | None = None) -> None:
    missing = missing or set()
    for remote_path, local_name in historical.SEASON_FILES.items():
        url = f"{historical.HISTORICAL_ARCHIVE_BASE}/{season}/{remote_path}"
        if local_name in missing:
            responses.add(responses.GET, url, status=404)
        else:
            responses.add(responses.GET, url, body="col_a,col_b\n1,2\n", status=200)


@responses.activate
def test_fetch_season_downloads_and_caches_all_files(tmp_path: Path):
    _mock_season("2024-25")
    results = historical.fetch_season("2024-25", tmp_path)

    assert results == {name: True for name in historical.SEASON_FILES.values()}
    for local_name in historical.SEASON_FILES.values():
        assert (tmp_path / "2024-25" / local_name).exists()


@responses.activate
def test_fetch_season_flags_missing_files_without_raising(tmp_path: Path):
    _mock_season("2024-25", missing={"merged_gw.csv"})
    results = historical.fetch_season("2024-25", tmp_path)

    assert results["merged_gw.csv"] is False
    assert not (tmp_path / "2024-25" / "merged_gw.csv").exists()
    assert results["teams.csv"] is True


@responses.activate
def test_fetch_all_seasons_covers_every_requested_season(tmp_path: Path):
    _mock_season("2023-24")
    _mock_season("2024-25")
    results = historical.fetch_all_seasons(tmp_path, seasons=["2023-24", "2024-25"])

    assert set(results.keys()) == {"2023-24", "2024-25"}
    assert (tmp_path / "2023-24" / "teams.csv").exists()
    assert (tmp_path / "2024-25" / "teams.csv").exists()


def test_load_merged_gw_concatenates_seasons_and_tags_each_row(tmp_path: Path):
    for season, element in [("2023-24", 1), ("2024-25", 2)]:
        season_dir = tmp_path / season
        season_dir.mkdir(parents=True)
        (season_dir / "merged_gw.csv").write_text(f"element,GW,total_points\n{element},1,5\n")

    df = historical.load_merged_gw(tmp_path, seasons=["2023-24", "2024-25"])

    assert len(df) == 2
    assert set(df["season"]) == {"2023-24", "2024-25"}
    assert list(df["element"]) == [1, 2]


def test_load_merged_gw_skips_seasons_with_no_cached_file(tmp_path: Path):
    season_dir = tmp_path / "2024-25"
    season_dir.mkdir(parents=True)
    (season_dir / "merged_gw.csv").write_text("element,GW,total_points\n1,1,5\n")

    df = historical.load_merged_gw(tmp_path, seasons=["2023-24", "2024-25"])

    assert len(df) == 1
    assert df.iloc[0]["season"] == "2024-25"


def test_load_merged_gw_returns_empty_frame_when_nothing_cached(tmp_path: Path):
    df = historical.load_merged_gw(tmp_path, seasons=["2024-25"])
    assert df.empty
