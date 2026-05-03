"""Adapter: dickreuter scraper table state → our GameState.

The scraper's `table` object has ad-hoc attributes set across many methods.
This module provides a clean translation to our typed GameState.

Card format conversion:
  - dickreuter: list of strings like ['As', 'Kh', '2d']
  - ours: list of ints 0-51 via Card encoding
"""

from __future__ import annotations

from typing import Any

from .engine.action import Action, ActionType
from .engine.card import Card
from .engine.game_state import GameState, Street


def card_str_to_int(card_str: str) -> int:
    """Convert dickreuter card string ('As', 'Kh', etc.) to our int encoding."""
    if not card_str or len(card_str) != 2:
        raise ValueError(f"Invalid card string: {card_str!r}")
    return Card.from_str(card_str).to_int()


def cards_to_ints(cards: list[str]) -> list[int]:
    return [card_str_to_int(c) for c in cards if c]


def gamestage_to_street(stage: str) -> Street:
    mapping = {
        'PreFlop': Street.PREFLOP,
        'Flop': Street.FLOP,
        'Turn': Street.TURN,
        'River': Street.RIVER,
    }
    return mapping.get(stage, Street.PREFLOP)


def to_chips(amount: float | str | None, bb_value: float = 0.02) -> int:
    """Convert dollar amount to chip units (1 BB = bb_value dollars).

    Default: 1 BB = $0.02 (micro stakes). Adjust per stake level.
    Returns chips in units where 1 BB = 100 chips (matches our defaults).
    """
    if amount is None or amount == '':
        return 0
    try:
        dollars = float(amount)
    except (ValueError, TypeError):
        return 0
    # 1 BB = bb_value dollars. We use 1 BB = 2 chips (matching default sb=1, bb=2).
    # So chips_per_dollar = 2 / bb_value.
    return max(0, int(dollars * (2.0 / bb_value)))


def to_game_state(
    table: Any,
    strategy_config: dict[str, Any] | None = None,
    bb_dollar_value: float = 0.02,
) -> GameState:
    """Translate dickreuter table object to our GameState.

    Args:
        table: dickreuter scraper table object (TableScreenBased instance)
        strategy_config: optional dict with 'bigBlind', 'smallBlind' overrides
        bb_dollar_value: dollar value of 1 BB on the platform (default $0.02)
    """
    config = strategy_config or {}

    # Cards: hole + board
    hole_strs = getattr(table, 'mycards', []) or []
    board_strs = getattr(table, 'cardsOnTable', []) or []
    hole_ints = cards_to_ints(hole_strs)
    board_ints = cards_to_ints(board_strs)

    # Street
    stage = getattr(table, 'gameStage', 'PreFlop')
    street = gamestage_to_street(stage)

    # Blinds
    bb_dollars = float(config.get('bigBlind', getattr(table, 'bigBlind', bb_dollar_value)))
    sb_dollars = float(config.get('smallBlind', getattr(table, 'smallBlind', bb_dollar_value / 2)))
    big_blind = to_chips(bb_dollars, bb_dollar_value)
    small_blind = to_chips(sb_dollars, bb_dollar_value)

    # Players
    other_players = getattr(table, 'other_players', []) or []
    num_other = len(other_players)
    num_players = 1 + num_other

    # Stacks: our seat is index 0, others follow
    my_stack = to_chips(getattr(table, 'myFunds', 100 * bb_dollar_value), bb_dollar_value)
    if my_stack == 0:
        my_stack = big_blind * 100  # default to 100 BB
    stacks = [my_stack]
    for op in other_players:
        stacks.append(to_chips(op.get('funds', 0), bb_dollar_value))

    # Folded: derive from status (status 0 = active, anything else may indicate folded)
    folded = [False]
    for op in other_players:
        status = op.get('status', 1)
        folded.append(status != 1)

    all_in = [False] * num_players

    # Current bets in this round
    pot = to_chips(getattr(table, 'totalPotValue', 0), bb_dollar_value)
    round_pot = to_chips(getattr(table, 'round_pot_value', 0), bb_dollar_value)

    # Our current bet derived from minCall (amount we still need)
    min_call_dollars = float(getattr(table, 'minCall', 0) or 0)
    current_call_value = float(getattr(table, 'currentCallValue', 0) or 0)
    to_call_chips = to_chips(min_call_dollars, bb_dollar_value)

    current_bets = [0] * num_players
    # Fill in opponent bets from their pot contribution this round
    for i, op in enumerate(other_players):
        op_pot = op.get('pot', 0)
        if op_pot in ('', None):
            current_bets[i + 1] = 0
        else:
            current_bets[i + 1] = to_chips(op_pot, bb_dollar_value)

    # If we owe call, our current bet is max - to_call
    max_other_bet = max(current_bets[1:]) if num_players > 1 else 0
    if to_call_chips > 0:
        current_bets[0] = max(0, max_other_bet - to_call_chips)
    else:
        current_bets[0] = max_other_bet

    # Button position (heads-up assumption: dealer = SB)
    button = int(getattr(table, 'dealer_position', 0))
    button = max(0, min(button, num_players - 1))

    return GameState(
        num_players=num_players,
        stacks=stacks,
        pot=max(pot, round_pot),
        board=board_ints,
        hole_cards=[hole_ints] + [[] for _ in range(num_other)],
        street=street,
        current_player=0,
        button=button,
        small_blind=small_blind,
        big_blind=big_blind,
        current_bets=current_bets,
        action_history=[],
        folded=folded,
        all_in=all_in,
    )


def action_to_dickreuter(action: Action) -> str:
    """Convert our Action to dickreuter's decision string for mouse.mouse_action."""
    mapping = {
        ActionType.FOLD: 'Fold',
        ActionType.CHECK: 'Check',
        ActionType.CALL: 'Call',
        ActionType.BET: 'Bet half pot',
        ActionType.RAISE: 'Bet pot',
        ActionType.ALL_IN: 'BetPlus',
    }
    return mapping.get(action.type, 'Fold')
