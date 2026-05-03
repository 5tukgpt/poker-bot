from __future__ import annotations

from phevaluator import evaluate_cards

from .card import Card, RANK_SYMBOLS, SUIT_SYMBOLS


def _int_to_phe(card_int: int) -> str:
    c = Card.from_int(card_int)
    return RANK_SYMBOLS[c.rank] + SUIT_SYMBOLS[c.suit]


def evaluate_hand(hole: list[int], board: list[int]) -> int:
    """Lower rank = better hand. Range: 1 (royal flush) to 7462 (worst high card)."""
    cards = [_int_to_phe(c) for c in hole + board]
    return evaluate_cards(*cards)


def best_hand_rank(hole: list[int], board: list[int]) -> int:
    """Evaluate best 5-card hand from hole + board. Works for 5, 6, or 7 total cards."""
    return evaluate_hand(hole, board)


def determine_winners(hands: list[list[int]], board: list[int]) -> list[int]:
    """Return indices of players with the best hand (ties possible)."""
    ranks = [evaluate_hand(h, board) for h in hands]
    best = min(ranks)
    return [i for i, r in enumerate(ranks) if r == best]
