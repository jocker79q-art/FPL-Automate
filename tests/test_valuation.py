from __future__ import annotations

from fpl_automate.squad.valuation import build_purchase_price_index, estimate_sell_value_tenths
from tests.conftest import make_player


def test_no_purchase_record_falls_back_to_current_price():
    player = make_player(1, now_cost_tenths=80)
    value, is_fallback = estimate_sell_value_tenths(player, {})
    assert value == 80
    assert is_fallback is True


def test_profit_is_halved_and_rounded_down():
    player = make_player(1, now_cost_tenths=90)  # bought at 80, now 90 -> profit 10 -> bank 5
    value, is_fallback = estimate_sell_value_tenths(player, {1: 80})
    assert value == 85
    assert is_fallback is False


def test_odd_profit_rounds_down():
    player = make_player(1, now_cost_tenths=85)  # profit 5 -> //2 == 2 -> sell 82
    value, _ = estimate_sell_value_tenths(player, {1: 80})
    assert value == 82


def test_loss_is_not_taxed():
    player = make_player(1, now_cost_tenths=70)  # bought at 80, fell to 70
    value, _ = estimate_sell_value_tenths(player, {1: 80})
    assert value == 70


def test_purchase_price_index_keeps_latest_transfer():
    transfers = [
        {"element_in": 1, "element_in_cost": 70, "event": 3},
        {"element_in": 1, "element_in_cost": 90, "event": 10},  # re-bought later at higher price
    ]
    index = build_purchase_price_index(transfers, team_id=999)
    assert index[1] == 90
