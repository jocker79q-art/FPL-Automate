"""Transfer recommendation engine: roll vs. one/two transfers, hit-aware.

Design summary
---------------
1. Establish a baseline: the best achievable lineup (per strategy) from the
   CURRENT squad, at 1/3/5-gameweek horizons.
2. Identify "sell candidates" in the current squad: players with weak
   medium-term projections, risk flags, or poor upcoming fixtures.
3. For each sell candidate, search realistic "buy candidates": same
   position, affordable within bank + this player's estimated sell value,
   not already owned, respecting the max-3-players-per-club rule.
4. For every (sell, buy) pair, build the hypothetical resulting squad and
   re-run the lineup optimiser on it, so the "gain" from a transfer
   reflects its actual effect on your best XI -- not just a raw
   points-per-player comparison.
5. A second transfer is searched *greedily* on top of the best single
   transfer (not an exhaustive search over all transfer pairs), to keep
   the search space tractable. This is a documented simplification: it can
   miss a jointly-optimal pair of transfers that isn't optimal
   individually.
6. "Roll" (0 transfers) is always included as a baseline scenario. Rolling
   has real option value -- an unused free transfer becomes 2 next week
   (up to a cap of 5) -- so a transfer is only recommended when its net
   gain (after any points hit) clears `min_net_gain_to_recommend`, a
   configurable margin, not merely when it's positive by any amount.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel

from fpl_automate.optimization.lineup import Strategy, optimize_lineup
from fpl_automate.squad.valuation import build_purchase_price_index, estimate_sell_value_tenths
from fpl_automate.storage.models import Player, PlayerProjection, SquadState

logger = logging.getLogger(__name__)

MAX_PLAYERS_PER_CLUB = 3
CANDIDATE_POOL_SIZE = 12  # buy candidates considered per sell candidate, per horizon
WORST_N_TO_CONSIDER_SELLING = 4
POINTS_PER_HIT = 4
DEFAULT_MIN_NET_GAIN_TO_RECOMMEND = 2.0  # net 5-GW points, after any hit


class TransferMove(BaseModel):
    sell_player_id: int
    buy_player_id: int
    sell_value_tenths: int
    buy_price_tenths: int


class TransferScenario(BaseModel):
    label: str
    moves: list[TransferMove]
    free_transfers_used: int
    paid_transfers: int
    points_hit: int
    resulting_bank_tenths: int
    baseline_ep_1gw: float
    new_ep_1gw: float
    baseline_ep_5gw: float
    new_ep_5gw: float
    ep_gain_1gw: float
    ep_gain_5gw: float
    net_ep_gain_5gw: float  # ep_gain_5gw - points_hit
    rationale: str
    risk_notes: list[str] = []
    recommended: bool = False


class TransferEngineError(RuntimeError):
    pass


def _club_counts(squad: list[Player]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for p in squad:
        counts[p.team_id] = counts.get(p.team_id, 0) + 1
    return counts


def _identify_sell_candidates(
    squad: list[Player], projections_5gw: dict[int, PlayerProjection]
) -> list[Player]:
    def weakness(p: Player) -> float:
        proj = projections_5gw[p.id]
        score = proj.expected_points
        if "injury_or_availability_doubt" in proj.risk_flags:
            score -= 3
        if "rotation_risk" in proj.risk_flags:
            score -= 2
        if "blank_gameweek" in proj.risk_flags:
            score -= 5
        return score

    ranked = sorted(squad, key=weakness)
    return ranked[:WORST_N_TO_CONSIDER_SELLING]


def _find_buy_candidates(
    position: int,
    max_affordable_tenths: int,
    exclude_ids: set[int],
    club_counts: dict[int, int],
    excluded_club_id: int | None,
    all_players: list[Player],
    projections_5gw: dict[int, PlayerProjection],
) -> list[Player]:
    def club_ok(p: Player) -> bool:
        current = club_counts.get(p.team_id, 0)
        # if we're replacing a player from the same club, that club's slot frees up
        if excluded_club_id is not None and p.team_id == excluded_club_id:
            current -= 1
        return current < MAX_PLAYERS_PER_CLUB

    candidates = [
        p
        for p in all_players
        if p.position == position
        and p.id not in exclude_ids
        and p.now_cost_tenths <= max_affordable_tenths
        and p.availability.status != "u"  # never suggest buying an unavailable/left-league player
        and club_ok(p)
        and p.id in projections_5gw
    ]
    candidates.sort(key=lambda p: projections_5gw[p.id].expected_points, reverse=True)
    return candidates[:CANDIDATE_POOL_SIZE]


def _lineup_ep(
    squad: list[Player], projections: dict[int, PlayerProjection], strategy: Strategy
) -> float:
    return optimize_lineup(squad, projections, strategy).total_expected_points


def recommend_transfers(
    squad: list[Player],
    squad_state: SquadState,
    transfers_history: list[dict],
    all_players: list[Player],
    projections_1gw: dict[int, PlayerProjection],
    projections_3gw: dict[int, PlayerProjection],
    projections_5gw: dict[int, PlayerProjection],
    strategy: Strategy = "balanced",
    max_transfer_risk: int = 4,
    min_net_gain_to_recommend: float = DEFAULT_MIN_NET_GAIN_TO_RECOMMEND,
) -> list[TransferScenario]:
    if len(squad) != 15:
        raise TransferEngineError(f"Expected a 15-man squad, got {len(squad)}")

    purchase_index = build_purchase_price_index(transfers_history, squad_state.team_id)
    squad_ids = {p.id for p in squad}
    by_id = {p.id: p for p in all_players}

    baseline_ep_1 = _lineup_ep(squad, projections_1gw, strategy)
    baseline_ep_5 = _lineup_ep(squad, projections_5gw, strategy)

    scenarios: list[TransferScenario] = []
    scenarios.append(
        TransferScenario(
            label="Roll transfer (make no changes)",
            moves=[],
            free_transfers_used=0,
            paid_transfers=0,
            points_hit=0,
            resulting_bank_tenths=squad_state.bank_tenths,
            baseline_ep_1gw=round(baseline_ep_1, 2),
            new_ep_1gw=round(baseline_ep_1, 2),
            baseline_ep_5gw=round(baseline_ep_5, 2),
            new_ep_5gw=round(baseline_ep_5, 2),
            ep_gain_1gw=0.0,
            ep_gain_5gw=0.0,
            net_ep_gain_5gw=0.0,
            rationale=(
                f"Keeps {min(squad_state.free_transfers_available + 1, 5)} free transfer(s) "
                "available next gameweek. Always the safe default when no transfer clears "
                "the minimum recommended net gain."
            ),
            recommended=False,  # decided after comparing against alternatives below
        )
    )

    sell_candidates = _identify_sell_candidates(squad, projections_5gw)
    club_counts = _club_counts(squad)

    single_transfer_scenarios: list[TransferScenario] = []
    for sell in sell_candidates:
        sell_value, is_fallback = estimate_sell_value_tenths(sell, purchase_index)
        max_affordable = sell_value + squad_state.bank_tenths
        buy_candidates = _find_buy_candidates(
            position=int(sell.position),
            max_affordable_tenths=max_affordable,
            exclude_ids=squad_ids,
            club_counts=club_counts,
            excluded_club_id=sell.team_id,
            all_players=all_players,
            projections_5gw=projections_5gw,
        )
        for buy in buy_candidates:
            new_squad = [p for p in squad if p.id != sell.id] + [buy]
            try:
                new_ep_1 = _lineup_ep(new_squad, projections_1gw, strategy)
                new_ep_5 = _lineup_ep(new_squad, projections_5gw, strategy)
            except Exception as exc:  # noqa: BLE001 - a single bad candidate shouldn't kill the search
                logger.debug("Skipping candidate swap %s->%s: %s", sell.id, buy.id, exc)
                continue

            for use_free in (True, False):
                paid = 0 if (use_free and squad_state.free_transfers_available >= 1) else 1
                hit = paid * POINTS_PER_HIT
                if not use_free and squad_state.free_transfers_available >= 1:
                    continue  # no reason to pay for a transfer you have free
                gain_5 = round(new_ep_5 - baseline_ep_5, 2)
                risk_notes = []
                if is_fallback:
                    risk_notes.append(
                        f"Sell value for {sell.web_name} assumed = current price "
                        "(no purchase-price record found; real sell value may be higher)."
                    )
                for flag in projections_5gw[buy.id].risk_flags:
                    risk_notes.append(f"{buy.web_name}: {flag}")

                scenario = TransferScenario(
                    label=f"{sell.web_name} -> {buy.web_name}"
                    + (" (free)" if hit == 0 else f" (-{hit} pts)"),
                    moves=[
                        TransferMove(
                            sell_player_id=sell.id,
                            buy_player_id=buy.id,
                            sell_value_tenths=sell_value,
                            buy_price_tenths=buy.now_cost_tenths,
                        )
                    ],
                    free_transfers_used=1 if hit == 0 else 0,
                    paid_transfers=paid,
                    points_hit=hit,
                    resulting_bank_tenths=max_affordable - buy.now_cost_tenths,
                    baseline_ep_1gw=round(baseline_ep_1, 2),
                    new_ep_1gw=round(new_ep_1, 2),
                    baseline_ep_5gw=round(baseline_ep_5, 2),
                    new_ep_5gw=round(new_ep_5, 2),
                    ep_gain_1gw=round(new_ep_1 - baseline_ep_1, 2),
                    ep_gain_5gw=gain_5,
                    net_ep_gain_5gw=round(gain_5 - hit, 2),
                    rationale=(
                        f"Projected {sell.web_name} {projections_5gw[sell.id].expected_points:.1f} "
                        f"vs {buy.web_name} {projections_5gw[buy.id].expected_points:.1f} pts "
                        f"over the next 5 GWs (best-XI impact, not raw player comparison)."
                    ),
                    risk_notes=risk_notes,
                )
                single_transfer_scenarios.append(scenario)

    single_transfer_scenarios.sort(key=lambda s: s.net_ep_gain_5gw, reverse=True)
    top_singles = single_transfer_scenarios[:5]
    scenarios.extend(top_singles)

    if top_singles and squad_state.free_transfers_available >= 2:
        best_first = top_singles[0]
        intermediate_squad = [p for p in squad if p.id != best_first.moves[0].sell_player_id] + [
            by_id[best_first.moves[0].buy_player_id]
        ]
        second_sell_candidates = _identify_sell_candidates(intermediate_squad, projections_5gw)
        second_bank = best_first.resulting_bank_tenths
        second_club_counts = _club_counts(intermediate_squad)
        best_second: TransferScenario | None = None
        for sell2 in second_sell_candidates:
            if sell2.id == best_first.moves[0].buy_player_id:
                continue
            sell_value2, is_fallback2 = estimate_sell_value_tenths(sell2, purchase_index)
            max_affordable2 = sell_value2 + second_bank
            buy_candidates2 = _find_buy_candidates(
                position=int(sell2.position),
                max_affordable_tenths=max_affordable2,
                exclude_ids={p.id for p in intermediate_squad},
                club_counts=second_club_counts,
                excluded_club_id=sell2.team_id,
                all_players=all_players,
                projections_5gw=projections_5gw,
            )
            first_sell_name = by_id[best_first.moves[0].sell_player_id].web_name
            first_buy_name = by_id[best_first.moves[0].buy_player_id].web_name
            for buy2 in buy_candidates2[:5]:
                final_squad = [p for p in intermediate_squad if p.id != sell2.id] + [buy2]
                new_ep_1 = _lineup_ep(final_squad, projections_1gw, strategy)
                new_ep_5 = _lineup_ep(final_squad, projections_5gw, strategy)
                hit = 0 if squad_state.free_transfers_available >= 2 else POINTS_PER_HIT
                gain_5 = round(new_ep_5 - baseline_ep_5, 2)
                candidate_scenario = TransferScenario(
                    label=(
                        f"{first_sell_name} -> {first_buy_name}, "
                        f"{sell2.web_name} -> {buy2.web_name}"
                        + (" (both free)" if hit == 0 else f" (-{hit} pts)")
                    ),
                    moves=[
                        best_first.moves[0],
                        TransferMove(
                            sell_player_id=sell2.id,
                            buy_player_id=buy2.id,
                            sell_value_tenths=sell_value2,
                            buy_price_tenths=buy2.now_cost_tenths,
                        ),
                    ],
                    free_transfers_used=2 if hit == 0 else 0,
                    paid_transfers=0 if hit == 0 else 2,
                    points_hit=hit,
                    resulting_bank_tenths=max_affordable2 - buy2.now_cost_tenths,
                    baseline_ep_1gw=round(baseline_ep_1, 2),
                    new_ep_1gw=round(new_ep_1, 2),
                    baseline_ep_5gw=round(baseline_ep_5, 2),
                    new_ep_5gw=round(new_ep_5, 2),
                    ep_gain_1gw=round(new_ep_1 - baseline_ep_1, 2),
                    ep_gain_5gw=gain_5,
                    net_ep_gain_5gw=round(gain_5 - hit, 2),
                    rationale="Greedy 2nd transfer on top of the best single transfer found.",
                    risk_notes=(["Sell value estimated with no-profit fallback."] if is_fallback2 else []),
                )
                if best_second is None or candidate_scenario.net_ep_gain_5gw > best_second.net_ep_gain_5gw:
                    best_second = candidate_scenario
        if best_second is not None:
            scenarios.append(best_second)

    for s in scenarios:
        if s.points_hit > max_transfer_risk:
            s.recommended = False
            s.risk_notes.append(
                f"Points hit ({s.points_hit}) exceeds MAX_TRANSFER_RISK ({max_transfer_risk}); "
                "not eligible for auto-approval regardless of projected gain."
            )
            continue
        if s.net_ep_gain_5gw >= min_net_gain_to_recommend:
            s.recommended = True

    # Ensure "roll" is marked recommended if nothing else clears the bar.
    if not any(s.recommended for s in scenarios):
        scenarios[0].recommended = True
        scenarios[0].rationale += " No transfer cleared the minimum recommended net gain threshold."

    scenarios.sort(key=lambda s: s.net_ep_gain_5gw, reverse=True)
    return scenarios
