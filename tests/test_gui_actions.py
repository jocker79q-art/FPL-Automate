"""Tests for gui/actions.py -- deliberately tkinter-free (per that module's
own docstring), so these run in any environment including one without a
Tk display.

Focus: `recommend_transfers_action`/`optimise_lineup_action` must thread
`settings.projection_model` and `settings.risk_aversion` through to
`compute_projections`/`recommend_transfers`/`optimize_lineup` -- a real
regression this module previously had (the CLI equivalents did this
correctly; the GUI actions silently ignored both settings and always used
the baseline "ml"/1.0 defaults regardless of what a user configured)."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fpl_automate.config import Settings
from fpl_automate.gui import actions
from fpl_automate.storage.models import ChipPlay, Gameweek, Position, SquadPick, SquadState
from fpl_automate.workflow import FetchedData
from tests.conftest import make_player, make_team

# A real FPL-shaped 15-man squad (2 GKP / 5 DEF / 5 MID / 3 FWD) --
# optimize_lineup requires exactly this shape, so every test shares it.
SQUAD_POSITIONS = (
    [Position.GOALKEEPER] * 2 + [Position.DEFENDER] * 5 + [Position.MIDFIELDER] * 5 + [Position.FORWARD] * 3
)
SQUAD_PLAYER_IDS = list(range(1, 16))


def _settings(**overrides) -> Settings:
    defaults = {
        "fpl_team_id": 123,
        "database_url": "sqlite:///:memory:",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _gameweek(id_: int, *, finished=False, is_current=False, is_next=False) -> Gameweek:
    return Gameweek(
        id=id_, name=f"GW{id_}", deadline_time=datetime(2026, 1, id_, tzinfo=UTC),
        finished=finished, is_current=is_current, is_next=is_next,
    )


def _fetched_data(squad_player_ids: list[int]) -> FetchedData:
    team = make_team(1)
    players = [
        make_player(pid, team_id=1, position=pos) for pid, pos in zip(squad_player_ids, SQUAD_POSITIONS, strict=True)
    ]
    gameweeks = [_gameweek(5, finished=True, is_current=True), _gameweek(6, is_next=True)]
    return FetchedData(teams=[team], players=players, gameweeks=gameweeks, fixtures=[])


def _squad_state(player_ids: list[int]) -> SquadState:
    picks = [
        SquadPick(element_id=pid, squad_position=i + 1, multiplier=1, is_captain=i == 0, is_vice_captain=i == 1)
        for i, pid in enumerate(player_ids)
    ]
    return SquadState(
        team_id=123, as_of_event=5, picks=picks, bank_tenths=10, squad_value_tenths=450,
        free_transfers_available=1, chips_used=[ChipPlay(name="wildcard", event=3)],
        wildcard_available=False, free_hit_available=True, bench_boost_available=True,
        triple_captain_available=True,
    )


class _FakeClient:
    def get_entry_transfers(self, team_id):
        return []


@pytest.fixture(autouse=True)
def _patch_common(monkeypatch):
    monkeypatch.setattr(actions, "build_client", lambda settings, cache_dir: _FakeClient())
    monkeypatch.setattr(actions, "build_db", lambda settings: object())
    monkeypatch.setattr(actions, "fetch_and_validate", lambda client, db: _fetched_data(SQUAD_PLAYER_IDS))
    monkeypatch.setattr(
        actions, "resolve_squad_state", lambda client, team_id, gameweeks: _squad_state(SQUAD_PLAYER_IDS)
    )


class TestRecommendTransfersActionThreadsSettings:
    def test_passes_projection_model_to_compute_projections(self, monkeypatch):
        captured = {}

        def fake_compute_projections(data, from_event, projection_model="ml", **kwargs):
            captured["projection_model"] = projection_model
            return {1: {}, 3: {}, 5: {}}

        def fake_recommend_transfers(**kwargs):
            captured["scenario_kwargs"] = kwargs
            return []

        monkeypatch.setattr(actions, "compute_projections", fake_compute_projections)
        monkeypatch.setattr(actions, "recommend_transfers", fake_recommend_transfers)

        actions.recommend_transfers_action(_settings(projection_model="baseline"), "balanced")

        assert captured["projection_model"] == "baseline"

    def test_passes_risk_aversion_to_recommend_transfers(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(actions, "compute_projections", lambda *a, **k: {1: {}, 3: {}, 5: {}})
        monkeypatch.setattr(
            actions, "recommend_transfers", lambda **kwargs: captured.update(kwargs) or []
        )

        actions.recommend_transfers_action(_settings(risk_aversion=3.5), "risk_adjusted")

        assert captured["risk_aversion"] == 3.5


class TestOptimiseLineupActionThreadsSettings:
    def test_passes_projection_model_to_compute_projections(self, monkeypatch):
        captured = {}

        def fake_compute_projections(data, from_event, horizons=(1, 3, 5), projection_model="ml", **kwargs):
            captured["projection_model"] = projection_model
            return {1: {p.id: _tight_proj(p.id) for p in data.players}}

        monkeypatch.setattr(actions, "compute_projections", fake_compute_projections)

        actions.optimise_lineup_action(_settings(projection_model="baseline"), "balanced")

        assert captured["projection_model"] == "baseline"

    def test_passes_risk_aversion_to_optimize_lineup(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            actions,
            "compute_projections",
            lambda data, from_event, horizons=(1, 3, 5), **k: {1: {p.id: _tight_proj(p.id) for p in data.players}},
        )
        real_optimize_lineup = actions.optimize_lineup

        def fake_optimize_lineup(squad, projections, strategy, risk_aversion=1.0):
            captured["risk_aversion"] = risk_aversion
            return real_optimize_lineup(squad, projections, strategy, risk_aversion)

        monkeypatch.setattr(actions, "optimize_lineup", fake_optimize_lineup)

        actions.optimise_lineup_action(_settings(risk_aversion=2.5), "risk_adjusted")

        assert captured["risk_aversion"] == 2.5


def _tight_proj(player_id: int):
    from fpl_automate.storage.models import PlayerProjection

    return PlayerProjection(
        player_id=player_id, gameweek=6, expected_points=5.0, floor_points=4.0, ceiling_points=6.0, confidence=0.8
    )
