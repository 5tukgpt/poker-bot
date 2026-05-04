"""GTO chart strategy: hard-coded heads-up NLHE preflop ranges + equity-based postflop.

Preflop ranges are simplified approximations of published GTO solutions for
heads-up NLHE at ~100bb depth. Far from perfect but vastly better than
hand-categorized heuristics.

Sources informing these ranges:
- Heads-up NL preflop solutions (SB opens ~70%, BB defends ~50%)
- Standard 3-bet/4-bet ranges from solver outputs

Postflop: uses Monte Carlo equity vs pot odds (same as HeuristicStrategy).
"""

from __future__ import annotations

from ..engine.action import Action, ActionType
from ..engine.card import Card, RANK_SYMBOLS
from ..engine.game_state import GameState, Street
from .base import BaseStrategy
from .equity import monte_carlo_equity


def hand_class(hole: list[int]) -> str:
    """Convert hole cards to canonical class like 'AA', 'AKs', 'AKo'."""
    c1, c2 = Card.from_int(hole[0]), Card.from_int(hole[1])
    r1, r2 = max(c1.rank, c2.rank), min(c1.rank, c2.rank)
    s1, s2 = RANK_SYMBOLS[r1], RANK_SYMBOLS[r2]
    if r1 == r2:
        return s1 + s2
    return s1 + s2 + ('s' if c1.suit == c2.suit else 'o')


def _expand(*items: str) -> set[str]:
    """Expand range notation: '22+' -> all pairs 22-AA, 'A2s+' -> A2s through AKs."""
    result: set[str] = set()
    rank_order = '23456789TJQKA'

    for item in items:
        if item.endswith('+'):
            base = item[:-1]
            if len(base) == 2 and base[0] == base[1]:  # pair like '22+'
                start_rank = base[0]
                start_idx = rank_order.index(start_rank)
                for i in range(start_idx, len(rank_order)):
                    r = rank_order[i]
                    result.add(r + r)
            elif len(base) == 3:  # like 'A2s+', 'KTo+'
                high, low, suit = base[0], base[1], base[2]
                high_idx = rank_order.index(high)
                low_idx = rank_order.index(low)
                for i in range(low_idx, high_idx):
                    r = rank_order[i]
                    result.add(high + r + suit)
            else:
                result.add(item)
        else:
            result.add(item)
    return result


# === RANGES ===

# SB (button) opening range — ~70% of hands in heads-up NL
SB_OPEN_RAISE = _expand(
    '22+',                                          # all pairs
    'A2s+', 'A2o+',                                 # all Ax
    'K2s+', 'K2o+',                                 # all Kx
    'Q2s+', 'Q4o+',                                 # most Qx
    'J3s+', 'J7o+',                                 # most Jx
    'T6s+', 'T8o+',                                 # T-something
    '95s+', '97o+',
    '85s+', '87o',
    '74s+',
    '64s+',
    '53s+',
    '43s', '42s',
    '32s',
)

# SB hands too weak to open (basically just folds)
SB_FOLD = {'72o', '73o', '74o', '75o', '76o', '62o', '63o', '64o', '65o',
           '52o', '53o', '54o', '42o', '43o', '32o',
           '82o', '83o', '84o', '85o', '86o', '87o',
           '92o', '93o', '94o', '95o', '96o',
           'T2o', 'T3o', 'T4o', 'T5o', 'T6o', 'T7o',
           'J2o', 'J3o', 'J4o', 'J5o', 'J6o',
           'Q2o', 'Q3o',
           '63s', '53s', '52s', '43s', '42s', '32s'}  # some marginal suited (debatable)

# Note: ranges overlap — SB_FOLD takes precedence over SB_OPEN_RAISE for collision

# SB hands strong enough to limp instead of raise (very rare, mostly traps)
SB_LIMP: set[str] = set()  # In modern HU GTO, SB rarely limps — almost always raise or fold

# BB defending vs SB minraise
BB_3BET = _expand(
    '99+',                                  # 99, TT, JJ, QQ, KK, AA
    'AQs+', 'AKo',
    'A5s', 'A4s',                          # some bluff 3-bets
    'KQs',
)
BB_CALL_VS_OPEN = _expand(
    '22+',                                  # underpairs call
    'A2s+',                                 # weak Ax suited
    'A2o+',
    'K2s+',
    'K9o+',
    'Q4s+', 'QTo+',
    'J7s+', 'JTo',
    'T7s+',
    '96s+',
    '85s+',
    '74s+',
    '64s+',
    '54s', '53s',
)

# SB facing BB 3-bet — 4-bet jam with premiums, call with rest
SB_4BET_JAM = _expand('QQ+', 'AKs', 'AKo')
SB_CALL_VS_3BET = _expand(
    'TT', '99', '88', '77', '66',          # mid pairs call
    'AQs', 'AJs',                          # decent broadways
    'KQs', 'KJs',
    'A5s', 'A4s',                          # bluff catchers
)

# BB facing SB 4-bet jam — call with strongest, fold rest
BB_CALL_VS_4BET = _expand('JJ+', 'AKs', 'AKo')


class GTOChartStrategy(BaseStrategy):
    """Heads-up NLHE strategy using hard-coded preflop GTO chart + equity postflop.

    Recognizes 4 preflop scenarios:
      1. SB opening (no bet faced)
      2. BB facing open
      3. SB facing 3-bet
      4. BB facing 4-bet
    Falls back to equity-based decisions postflop or in unknown scenarios.
    """

    def __init__(self, postflop_sims: int = 200, aggression: float = 1.0) -> None:
        self.postflop_sims = postflop_sims
        self.aggression = aggression

    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action:
        if state.street == Street.PREFLOP:
            return self._preflop(state, legal_actions)
        return self._postflop(state, legal_actions)

    def _preflop(self, state: GameState, legal: list[ActionType]) -> Action:
        p = state.current_player
        hole = state.hole_cards[p]
        hc = hand_class(hole)

        max_bet = max(state.current_bets) if state.current_bets else 0
        my_bet = state.current_bets[p] if state.current_bets else 0
        to_call = max_bet - my_bet
        bb = state.big_blind

        # Detect scenario from action history + position
        # In heads-up: button = SB acts first preflop
        is_button = (p == state.button)
        prior_actions = [a for a in state.action_history if a.player_idx == p]
        opp_raises = sum(1 for a in state.action_history
                         if a.player_idx != p and a.type in (ActionType.BET, ActionType.RAISE))

        # Scenario 1: SB opening (no bet faced beyond blinds)
        if is_button and to_call <= bb and len(prior_actions) == 0:
            return self._sb_open(state, legal, hc, p)

        # Scenario 2: BB facing SB raise
        if not is_button and opp_raises >= 1 and len(prior_actions) == 0:
            return self._bb_vs_open(state, legal, hc, p, to_call)

        # Scenario 3: SB facing BB 3-bet
        if is_button and opp_raises >= 1 and len(prior_actions) >= 1:
            return self._sb_vs_3bet(state, legal, hc, p, to_call)

        # Scenario 4: BB facing SB 4-bet
        if not is_button and opp_raises >= 2 and len(prior_actions) >= 1:
            return self._bb_vs_4bet(state, legal, hc, p, to_call)

        # Fallback: equity-based decision
        return self._equity_decision(state, legal, hole, p, to_call)

    def _sb_open(self, state, legal, hc, p) -> Action:
        if hc in SB_FOLD:
            if ActionType.CHECK in legal:
                return Action(ActionType.CHECK, 0, p)
            return Action(ActionType.FOLD, 0, p)
        if hc in SB_OPEN_RAISE:
            raise_size = int(state.big_blind * 2.5 * self.aggression)
            if ActionType.RAISE in legal:
                return Action(ActionType.RAISE, raise_size, p)
            if ActionType.BET in legal:
                return Action(ActionType.BET, raise_size, p)
            if ActionType.CALL in legal:
                return Action(ActionType.CALL, state.big_blind, p)
        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK, 0, p)
        return Action(ActionType.FOLD, 0, p)

    def _bb_vs_open(self, state, legal, hc, p, to_call) -> Action:
        if hc in BB_3BET:
            raise_size = int(to_call * 3 * self.aggression)
            if ActionType.RAISE in legal:
                return Action(ActionType.RAISE, raise_size, p)
            if ActionType.BET in legal:
                return Action(ActionType.BET, raise_size, p)
        if hc in BB_CALL_VS_OPEN:
            if ActionType.CALL in legal:
                return Action(ActionType.CALL, to_call, p)
        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK, 0, p)
        return Action(ActionType.FOLD, 0, p)

    def _sb_vs_3bet(self, state, legal, hc, p, to_call) -> Action:
        if hc in SB_4BET_JAM:
            if ActionType.ALL_IN in legal:
                return Action(ActionType.ALL_IN, state.stacks[p], p)
            if ActionType.RAISE in legal:
                return Action(ActionType.RAISE, state.stacks[p], p)
        if hc in SB_CALL_VS_3BET:
            if ActionType.CALL in legal:
                return Action(ActionType.CALL, to_call, p)
        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK, 0, p)
        return Action(ActionType.FOLD, 0, p)

    def _bb_vs_4bet(self, state, legal, hc, p, to_call) -> Action:
        if hc in BB_CALL_VS_4BET:
            # If facing all-in, call
            if ActionType.CALL in legal:
                return Action(ActionType.CALL, to_call, p)
            if ActionType.ALL_IN in legal:
                return Action(ActionType.ALL_IN, state.stacks[p], p)
        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK, 0, p)
        return Action(ActionType.FOLD, 0, p)

    def _equity_decision(self, state, legal, hole, p, to_call) -> Action:
        """Fallback when scenario isn't recognized."""
        num_opp = max(1, len(state.players_in_hand) - 1)
        equity = monte_carlo_equity(hole, state.board, num_opp, self.postflop_sims)
        pot = state.pot
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

        if equity > 0.65:
            bet = int(pot * 0.7 * self.aggression)
            if ActionType.RAISE in legal:
                return Action(ActionType.RAISE, bet, p)
            if ActionType.BET in legal:
                return Action(ActionType.BET, bet, p)
            if ActionType.CALL in legal:
                return Action(ActionType.CALL, to_call, p)
        if equity > pot_odds + 0.05:
            if to_call > 0 and ActionType.CALL in legal:
                return Action(ActionType.CALL, to_call, p)
            if ActionType.CHECK in legal:
                return Action(ActionType.CHECK, 0, p)
        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK, 0, p)
        return Action(ActionType.FOLD, 0, p)

    def _postflop(self, state, legal) -> Action:
        p = state.current_player
        hole = state.hole_cards[p]
        max_bet = max(state.current_bets) if state.current_bets else 0
        my_bet = state.current_bets[p] if state.current_bets else 0
        to_call = max_bet - my_bet
        return self._equity_decision(state, legal, hole, p, to_call)
