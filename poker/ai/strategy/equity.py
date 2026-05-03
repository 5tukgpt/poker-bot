from __future__ import annotations

import random

from ..engine.evaluator import evaluate_hand


def monte_carlo_equity(
    hole_cards: list[int],
    board: list[int],
    num_opponents: int = 1,
    num_simulations: int = 1000,
) -> float:
    """Estimate win probability via Monte Carlo simulation."""
    known = set(hole_cards + board)
    remaining = [c for c in range(52) if c not in known]
    wins = 0
    ties = 0

    for _ in range(num_simulations):
        deck = remaining.copy()
        random.shuffle(deck)
        idx = 0

        sim_board = board.copy()
        while len(sim_board) < 5:
            sim_board.append(deck[idx])
            idx += 1

        opp_hands: list[list[int]] = []
        for _ in range(num_opponents):
            opp_hands.append([deck[idx], deck[idx + 1]])
            idx += 2

        my_rank = evaluate_hand(hole_cards, sim_board)
        opp_ranks = [evaluate_hand(h, sim_board) for h in opp_hands]
        best_opp = min(opp_ranks) if opp_ranks else 7463

        if my_rank < best_opp:
            wins += 1
        elif my_rank == best_opp:
            ties += 1

    return (wins + ties * 0.5) / num_simulations
