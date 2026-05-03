from __future__ import annotations

from ..engine.action import Action, ActionType
from ..engine.card import Card, RANK_SYMBOLS
from ..engine.game_state import GameState, Street
from .base import BaseStrategy
from .equity import monte_carlo_equity

PREMIUM_HANDS = {
    'AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo',
}
STRONG_HANDS = {
    'TT', '99', 'AQs', 'AQo', 'AJs', 'KQs',
}
PLAYABLE_HANDS = {
    '88', '77', '66', 'ATs', 'ATo', 'KJs', 'KJo', 'QJs',
    'JTs', 'T9s', '98s', '87s', 'A9s', 'A8s', 'KTs',
}


def _hand_category(hole_ints: list[int]) -> str:
    c1, c2 = Card.from_int(hole_ints[0]), Card.from_int(hole_ints[1])
    r1, r2 = max(c1.rank, c2.rank), min(c1.rank, c2.rank)
    s1, s2 = RANK_SYMBOLS[r1], RANK_SYMBOLS[r2]
    suited = 's' if c1.suit == c2.suit else 'o'

    if r1 == r2:
        return s1 + s2
    return s1 + s2 + suited


class HeuristicStrategy(BaseStrategy):
    """Rule-based strategy using preflop hand categories + postflop equity."""

    def __init__(self, aggression: float = 1.0, num_simulations: int = 500) -> None:
        self.aggression = aggression
        self.num_simulations = num_simulations

    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action:
        player = state.current_player
        hole = state.hole_cards[player]

        if state.street == Street.PREFLOP:
            return self._preflop_action(state, legal_actions, hole)
        return self._postflop_action(state, legal_actions, hole)

    def _preflop_action(
        self, state: GameState, legal: list[ActionType], hole: list[int],
    ) -> Action:
        player = state.current_player
        cat = _hand_category(hole)

        max_bet = max(state.current_bets) if state.current_bets else 0
        my_bet = state.current_bets[player] if state.current_bets else 0
        to_call = max_bet - my_bet

        if cat in PREMIUM_HANDS:
            raise_amount = int(max(max_bet * 3, state.big_blind * 3) * self.aggression)
            if ActionType.RAISE in legal:
                return Action(ActionType.RAISE, raise_amount, player)
            if ActionType.BET in legal:
                return Action(ActionType.BET, raise_amount, player)
            if ActionType.CALL in legal:
                return Action(ActionType.CALL, to_call, player)
            return Action(ActionType.CHECK, 0, player)

        if cat in STRONG_HANDS:
            raise_amount = int(max(max_bet * 2.5, state.big_blind * 2.5) * self.aggression)
            if to_call <= state.big_blind * 4:
                if ActionType.RAISE in legal:
                    return Action(ActionType.RAISE, raise_amount, player)
                if ActionType.BET in legal:
                    return Action(ActionType.BET, raise_amount, player)
            if ActionType.CALL in legal:
                return Action(ActionType.CALL, to_call, player)
            return Action(ActionType.CHECK, 0, player)

        if cat in PLAYABLE_HANDS:
            if to_call <= state.big_blind * 3:
                if ActionType.CALL in legal:
                    return Action(ActionType.CALL, to_call, player)
            if ActionType.CHECK in legal:
                return Action(ActionType.CHECK, 0, player)
            return Action(ActionType.FOLD, 0, player)

        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK, 0, player)
        return Action(ActionType.FOLD, 0, player)

    def _postflop_action(
        self, state: GameState, legal: list[ActionType], hole: list[int],
    ) -> Action:
        player = state.current_player
        num_opponents = len(state.players_in_hand) - 1
        equity = monte_carlo_equity(
            hole, state.board, num_opponents, self.num_simulations,
        )

        max_bet = max(state.current_bets) if state.current_bets else 0
        my_bet = state.current_bets[player] if state.current_bets else 0
        to_call = max_bet - my_bet
        pot = state.pot

        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

        if equity > 0.7 * self.aggression:
            bet_size = int(pot * 0.75 * self.aggression)
            if ActionType.RAISE in legal:
                return Action(ActionType.RAISE, bet_size, player)
            if ActionType.BET in legal:
                return Action(ActionType.BET, bet_size, player)
            if ActionType.CALL in legal:
                return Action(ActionType.CALL, to_call, player)

        if equity > pot_odds + 0.05:
            if to_call > 0 and ActionType.CALL in legal:
                return Action(ActionType.CALL, to_call, player)
            bet_size = int(pot * 0.5 * self.aggression)
            if ActionType.BET in legal and equity > 0.5:
                return Action(ActionType.BET, bet_size, player)
            if ActionType.CHECK in legal:
                return Action(ActionType.CHECK, 0, player)

        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK, 0, player)
        return Action(ActionType.FOLD, 0, player)
