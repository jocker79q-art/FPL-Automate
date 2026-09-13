"""Sell-value estimation: what you'd actually receive for selling a squad player.

FPL rule: if a player's price has risen since you bought them, you only
bank HALF the profit (rounded down to the nearest £0.1m); if it's fallen,
you take the full loss. This means "budget available" is not simply
"sum of now_cost for players you'd sell" -- overestimating it would risk
recommending a transfer you can't actually afford.

Limitation: we can only know your purchase price for a player if your
transfer history (a public endpoint) shows you buying them. Anyone still
in your very first (GW1) squad, who you've never re-bought after selling,
has no such record -- for those we conservatively assume no profit
(sell value = current price), which can *underestimate* your real budget
but will never *overestimate* it and suggest an unaffordable transfer.
"""
from __future__ import annotations

from fpl_automate.storage.models import Player


def build_purchase_price_index(transfers: list[dict], team_id: int) -> dict[int, int]:
    """Maps player_id -> most recent known purchase price (tenths of £m) for this manager."""
    index: dict[int, tuple[int, int]] = {}  # player_id -> (event, price) keeping the latest event
    for t in transfers:
        pid = t.get("element_in")
        event = t.get("event", 0)
        price = t.get("element_in_cost")
        if pid is None or price is None:
            continue
        prev = index.get(pid)
        if prev is None or event >= prev[0]:
            index[pid] = (event, price)
    return {pid: price for pid, (_event, price) in index.items()}


def estimate_sell_value_tenths(
    player: Player, purchase_price_index: dict[int, int]
) -> tuple[int, bool]:
    """Returns (sell_value_tenths, is_estimated_no_profit_fallback)."""
    purchase_price = purchase_price_index.get(player.id)
    if purchase_price is None:
        return player.now_cost_tenths, True

    profit = player.now_cost_tenths - purchase_price
    if profit <= 0:
        return player.now_cost_tenths, False
    sell_value = purchase_price + profit // 2
    return sell_value, False
