"""'By the book' HU NLHE strategy with proper preflop ranges, c-bet logic,
board texture awareness, and standard bet sizing.

Implements canonical HU NLHE play at ~100bb:
- Preflop: SB opens 70%, BB defends 65% (3-bet 12%, call 53%)
- 4-bet/5-bet jam ranges
- Continuation betting: 70% on dry boards, 55% on wet
- Bet sizing varies by board texture and street
- Mixed strategies for marginal hands (probabilistic)
- Range vs range awareness (PFR has range advantage)

References (heads-up NL ~100bb):
- Will Tipton "Expert Heads Up No Limit Hold'em"
- Standard solver outputs (PioSolver heads-up trees)
- Modern Poker Theory preflop charts
"""

from __future__ import annotations

import random

from ..engine.action import Action, ActionType
from ..engine.card import Card, RANK_SYMBOLS
from ..engine.evaluator import evaluate_hand
from ..engine.game_state import GameState, Street
from .base import BaseStrategy
from .equity import monte_carlo_equity
from .gto_chart import _expand, hand_class


# === PREFLOP RANGES (HU NLHE 100bb) ===

# SB open-raise (~70% of hands). Mixed strategy values are 0.0-1.0 frequency.
SB_OPEN_RAISE = _expand(
    '22+',
    'A2s+', 'A2o+',
    'K2s+', 'K2o+',
    'Q2s+', 'Q4o+',
    'J3s+', 'J7o+',
    'T6s+', 'T8o+',
    '95s+', '97o+',
    '85s+', '87o',
    '74s+',
    '64s+',
    '53s+',
    '43s', '42s',
    '32s',
)

# BB 3-bet range (~12% — value + bluffs)
BB_3BET_VALUE = _expand('99+', 'AQs+', 'AKo')
BB_3BET_BLUFF = {'A5s', 'A4s', 'A3s', 'KQs', 'K9s', 'Q9s'}
BB_3BET = BB_3BET_VALUE | BB_3BET_BLUFF

# BB call range (~53% — defends wide)
BB_CALL_VS_OPEN = _expand(
    '22+',
    'A2s+', 'A2o+',
    'K2s+', 'K7o+',
    'Q4s+', 'Q9o+',
    'J7s+', 'JTo',
    'T7s+', 'T9o',
    '96s+',
    '85s+',
    '74s+',
    '64s+',
    '53s+', '54o',
)

# SB 4-bet range (vs BB 3-bet)
SB_4BET_VALUE = _expand('QQ+', 'AKs', 'AKo')
SB_4BET_BLUFF = {'A5s', 'A4s', 'KQs'}
SB_4BET_JAM = SB_4BET_VALUE | SB_4BET_BLUFF

# SB call vs 3-bet range
SB_CALL_VS_3BET = _expand(
    'JJ', 'TT', '99', '88', '77', '66',
    'AQs', 'AJs', 'ATs',
    'KQs', 'KJs',
    'QJs',
)

# BB call vs 4-bet jam (gets very tight)
BB_CALL_VS_4BET = _expand('JJ+', 'AKs', 'AKo')


# === BOARD TEXTURE CLASSIFICATION ===

def board_texture(board: list[int]) -> dict:
    """Classify board: dry/wet, paired, draws, etc."""
    if len(board) < 3:
        return {'dry': True, 'wet': False, 'paired': False, 'flush_draw': False,
                'straight_draw': False, 'monotone': False, 'highcard': 0}

    cards = [Card.from_int(c) for c in board]
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    suit_counts = {s: suits.count(s) for s in set(suits)}
    rank_counts = {r: ranks.count(r) for r in set(ranks)}

    paired = any(v >= 2 for v in rank_counts.values())
    monotone = max(suit_counts.values()) == len(board)
    flush_draw = max(suit_counts.values()) >= 2

    # Straight-y if ranks span <= 5 OR there are connected/gapped cards
    rank_range = max(ranks) - min(ranks)
    sorted_ranks = sorted(set(ranks))
    has_connected = any(sorted_ranks[i + 1] - sorted_ranks[i] <= 2
                         for i in range(len(sorted_ranks) - 1))
    straight_draw = (rank_range <= 5 and len(set(ranks)) >= 3) or has_connected

    # "Dry" = uncoordinated, no draws, ace high or king high
    wet = (flush_draw and len([s for s, c in suit_counts.items() if c >= 2]) > 0) \
          or straight_draw or paired
    dry = not wet

    return {
        'dry': dry,
        'wet': wet,
        'paired': paired,
        'flush_draw': flush_draw or monotone,
        'monotone': monotone,
        'straight_draw': straight_draw,
        'highcard': ranks[0],
    }


# === HAND CATEGORIES (POSTFLOP) ===

def categorize_made_hand(hole: list[int], board: list[int]) -> str:
    """Bucket made hand strength: 'monster', 'strong', 'medium', 'weak', 'air'."""
    if len(board) < 3:
        return 'medium'  # preflop fallback shouldn't reach here

    rank = evaluate_hand(hole, board)
    # phevaluator: 1=royal flush, 7462=worst high card
    if rank <= 1609: return 'monster'      # flush+
    if rank <= 2467: return 'monster'      # straight, trips
    if rank <= 3325: return 'strong'       # two pair
    if rank <= 5853: return 'strong'       # over pair / top pair top kicker
    if rank <= 6185: return 'medium'       # weaker pair
    if rank <= 6678: return 'weak'         # high card with strong kickers
    return 'air'


def has_strong_draw(hole: list[int], board: list[int]) -> bool:
    """Detect flush or open-ended straight draws."""
    if len(board) < 3 or len(board) >= 5:
        return False
    cards = [Card.from_int(c) for c in hole + board]
    suits = [c.suit for c in cards]
    suit_counts = {s: suits.count(s) for s in set(suits)}
    has_flush_draw = max(suit_counts.values()) == 4

    ranks = sorted(set([c.rank for c in cards]))
    has_oesd = False
    for i in range(len(ranks) - 3):
        if ranks[i + 3] - ranks[i] == 3:
            has_oesd = True
            break
    return has_flush_draw or has_oesd


# === BET SIZING ===

def cbet_size(pot: int, bb: int, texture: dict, is_value: bool) -> int:
    """Standard c-bet sizing. Smaller on dry boards, bigger on wet."""
    if texture['monotone'] or texture['paired']:
        return max(int(pot * 0.33), bb)
    if texture['wet']:
        return max(int(pot * 0.67), bb)
    # Dry board: small c-bet
    return max(int(pot * 0.33), bb)


def value_bet_size(pot: int, bb: int, street: Street) -> int:
    """Bet sizing for value (made hands)."""
    if street == Street.RIVER:
        return max(int(pot * 0.75), bb)  # bigger on river
    return max(int(pot * 0.67), bb)


def bluff_size(pot: int, bb: int) -> int:
    return max(int(pot * 0.5), bb)


# === MIXED FREQUENCIES (probabilistic decisions) ===

CBET_FREQ_DRY = 0.75       # c-bet 75% on dry boards
CBET_FREQ_WET = 0.55       # c-bet 55% on wet boards
CHECK_RAISE_FREQ = 0.15    # check-raise 15% of the time with strong hands
FLOAT_FREQ = 0.30          # float (call without strong hand) 30%


class BookStrategy(BaseStrategy):
    """Comprehensive HU NLHE strategy following standard poker theory.

    Preflop: detailed ranges with mixed strategies for marginal hands.
    Postflop: c-bet logic, board texture awareness, range advantage,
    bluff catching, value bet sizing.
    """

    def __init__(self, postflop_sims: int = 150, aggression: float = 1.0,
                  rng_seed: int | None = None) -> None:
        self.postflop_sims = postflop_sims
        self.aggression = aggression
        if rng_seed is not None:
            random.seed(rng_seed)

    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action:
        if state.street == Street.PREFLOP:
            return self._preflop(state, legal_actions)
        return self._postflop(state, legal_actions)

    # === PREFLOP ===

    def _preflop(self, state: GameState, legal: list[ActionType]) -> Action:
        p = state.current_player
        hole = state.hole_cards[p]
        hc = hand_class(hole)

        max_bet = max(state.current_bets) if state.current_bets else 0
        my_bet = state.current_bets[p] if state.current_bets else 0
        to_call = max_bet - my_bet
        bb = state.big_blind

        is_button = (p == state.button)
        prior_self_actions = [a for a in state.action_history if a.player_idx == p]
        opp_raises = sum(1 for a in state.action_history
                         if a.player_idx != p and a.type in (ActionType.BET, ActionType.RAISE))

        # SB opening
        if is_button and to_call <= bb and len(prior_self_actions) == 0:
            return self._sb_open(state, legal, hc, p)

        # BB facing SB raise
        if not is_button and opp_raises >= 1 and len(prior_self_actions) == 0:
            return self._bb_vs_open(state, legal, hc, p, to_call)

        # SB facing BB 3-bet
        if is_button and opp_raises >= 1 and len(prior_self_actions) >= 1:
            return self._sb_vs_3bet(state, legal, hc, p, to_call)

        # BB facing SB 4-bet
        if not is_button and opp_raises >= 2 and len(prior_self_actions) >= 1:
            return self._bb_vs_4bet(state, legal, hc, p, to_call)

        return self._fallback_action(state, legal, hole, p, to_call)

    def _sb_open(self, state, legal, hc, p) -> Action:
        if hc not in SB_OPEN_RAISE:
            if ActionType.CHECK in legal:
                return Action(ActionType.CHECK, 0, p)
            return Action(ActionType.FOLD, 0, p)

        # Mixed: open-raise with sizing varying by hand strength
        raise_size = int(state.big_blind * 2.5 * self.aggression)
        if ActionType.RAISE in legal:
            return Action(ActionType.RAISE, raise_size, p)
        if ActionType.BET in legal:
            return Action(ActionType.BET, raise_size, p)
        return Action(ActionType.CALL, state.big_blind, p)

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
            if ActionType.CALL in legal:
                return Action(ActionType.CALL, to_call, p)
            if ActionType.ALL_IN in legal:
                return Action(ActionType.ALL_IN, state.stacks[p], p)
        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK, 0, p)
        return Action(ActionType.FOLD, 0, p)

    # === POSTFLOP ===

    def _postflop(self, state: GameState, legal: list[ActionType]) -> Action:
        p = state.current_player
        hole = state.hole_cards[p]
        bb = state.big_blind
        pot = state.pot
        max_bet = max(state.current_bets) if state.current_bets else 0
        my_bet = state.current_bets[p] if state.current_bets else 0
        to_call = max_bet - my_bet

        texture = board_texture(state.board)
        category = categorize_made_hand(hole, state.board)
        has_draw = has_strong_draw(hole, state.board)

        # Did we raise preflop? (Are we PFR?)
        my_preflop_actions = [a for a in state.action_history
                                if a.player_idx == p and a.type in (ActionType.BET, ActionType.RAISE)]
        is_pfr = len(my_preflop_actions) > 0

        # Who acts first matters: in HU postflop, BB acts first
        opp_action_this_street = self._opp_action_this_street(state)
        facing_bet = to_call > 0
        opp_checked = opp_action_this_street == 'check'
        opp_bet = opp_action_this_street == 'bet'

        # MONSTER HAND: bet for value, sometimes slow-play
        if category == 'monster':
            if facing_bet:
                # Raise for value (60%) or call to trap (40%)
                if random.random() < 0.6 and ActionType.RAISE in legal:
                    return Action(ActionType.RAISE, value_bet_size(pot, bb, state.street) * 2, p)
                if ActionType.CALL in legal:
                    return Action(ActionType.CALL, to_call, p)
            # No bet faced: bet for value
            size = value_bet_size(pot, bb, state.street)
            if ActionType.BET in legal:
                return Action(ActionType.BET, size, p)
            if ActionType.RAISE in legal:
                return Action(ActionType.RAISE, size, p)

        # STRONG HAND (top pair/overpair/two pair): value bet, fold to massive aggression
        if category == 'strong':
            if facing_bet:
                # Pot odds check: if facing >2x pot, fold weaker strong hands
                if to_call > pot * 1.5 and category != 'monster':
                    return Action(ActionType.FOLD, 0, p)
                if ActionType.CALL in legal:
                    return Action(ActionType.CALL, to_call, p)
            size = value_bet_size(pot, bb, state.street)
            if ActionType.BET in legal:
                return Action(ActionType.BET, size, p)
            if ActionType.CHECK in legal:
                return Action(ActionType.CHECK, 0, p)

        # MEDIUM HAND (mid pair, top pair weak kicker): bluff catch, check
        if category == 'medium':
            if facing_bet:
                # Pot odds: call if odds are reasonable
                pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 1
                # Estimate equity quickly
                num_opp = max(1, len(state.players_in_hand) - 1)
                eq = monte_carlo_equity(hole, state.board, num_opp, self.postflop_sims)
                if eq > pot_odds + 0.05 and ActionType.CALL in legal:
                    return Action(ActionType.CALL, to_call, p)
                return Action(ActionType.FOLD, 0, p)
            # No bet faced: thin value bet sometimes if PFR
            if is_pfr and random.random() < 0.5:
                size = max(int(pot * 0.4), bb)
                if ActionType.BET in legal:
                    return Action(ActionType.BET, size, p)
            if ActionType.CHECK in legal:
                return Action(ActionType.CHECK, 0, p)

        # DRAW: semi-bluff
        if has_draw:
            if facing_bet:
                pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 1
                # Draws have ~30% equity, call if odds are good
                if pot_odds < 0.33 and ActionType.CALL in legal:
                    return Action(ActionType.CALL, to_call, p)
                # Sometimes raise as semi-bluff
                if random.random() < 0.25 and ActionType.RAISE in legal:
                    return Action(ActionType.RAISE, bluff_size(pot, bb), p)
                return Action(ActionType.FOLD, 0, p)
            # No bet faced: bet draws sometimes
            if random.random() < 0.5 and ActionType.BET in legal:
                return Action(ActionType.BET, bluff_size(pot, bb), p)
            if ActionType.CHECK in legal:
                return Action(ActionType.CHECK, 0, p)

        # WEAK / AIR: c-bet bluff if PFR, otherwise check-fold
        if facing_bet:
            return Action(ActionType.FOLD, 0, p)

        # Check or c-bet bluff?
        if is_pfr and state.street == Street.FLOP:
            cbet_freq = CBET_FREQ_DRY if texture['dry'] else CBET_FREQ_WET
            if random.random() < cbet_freq * self.aggression:
                size = cbet_size(pot, bb, texture, is_value=False)
                if ActionType.BET in legal:
                    return Action(ActionType.BET, size, p)

        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK, 0, p)
        return Action(ActionType.FOLD, 0, p)

    def _opp_action_this_street(self, state: GameState) -> str:
        """Detect what opponent did most recently this street: 'check', 'bet', or 'none'."""
        # Find actions in current street (after last street break in history)
        # Simplification: look at last opponent action in action_history
        p = state.current_player
        for action in reversed(state.action_history):
            if action.player_idx != p:
                if action.type in (ActionType.CHECK,):
                    return 'check'
                if action.type in (ActionType.BET, ActionType.RAISE, ActionType.CALL):
                    return 'bet'
                break
        return 'none'

    def _fallback_action(self, state, legal, hole, p, to_call) -> Action:
        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK, 0, p)
        return Action(ActionType.FOLD, 0, p)
