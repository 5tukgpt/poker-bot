"""Counterfactual Regret Minimization for heads-up NLHE with abstraction.

Card abstraction: 5 buckets based on hand strength (premium → trash).
Action abstraction: 4 actions (FOLD, CHECK_CALL, BET_HALF_POT, ALL_IN).
History string encodes the action sequence per street.

Algorithm: External-sampling MCCFR over heads-up NL game tree.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

import numpy as np

from ..engine.action import Action, ActionType
from ..engine.card import Card
from ..engine.evaluator import determine_winners, evaluate_hand
from ..engine.game_state import GameState, Street
from .base import BaseStrategy


CFR_ACTIONS = [
    ActionType.FOLD,
    ActionType.CHECK,
    ActionType.BET,
    ActionType.ALL_IN,
]
NUM_CFR_ACTIONS = len(CFR_ACTIONS)
NUM_BUCKETS = 5


def hand_to_bucket(hole: list[int], board: list[int]) -> int:
    """Map (hole, board) → 1 of 5 strength buckets. 0=trash, 4=premium."""
    if len(board) == 0:
        c1, c2 = Card.from_int(hole[0]), Card.from_int(hole[1])
        high = max(c1.rank, c2.rank)
        low = min(c1.rank, c2.rank)
        suited = c1.suit == c2.suit

        if c1.rank == c2.rank:  # pocket pair
            if c1.rank >= 12: return 4   # QQ+
            if c1.rank >= 9:  return 3   # 99-JJ
            if c1.rank >= 6:  return 2   # 66-88
            return 1
        if high == 14:  # ace high
            if low >= 12: return 4       # AK, AQ
            if low >= 10 or (suited and low >= 8): return 3
            if suited: return 2
            return 1
        if high >= 12 and low >= 10:    # KQ, KJ, QJ
            return 3 if suited else 2
        if suited and (high - low <= 2) and high >= 9:
            return 2
        if high >= 11:
            return 1
        return 0

    rank = evaluate_hand(hole, board)
    if rank <= 322: return 4    # straight flush+
    if rank <= 1609: return 4   # flush
    if rank <= 2467: return 3   # straight or trips
    if rank <= 3325: return 3   # two pair
    if rank <= 6185: return 2   # one pair
    if rank <= 7000: return 1
    return 0


def info_set_key(bucket: int, history: str, street: int) -> str:
    return f"s{street}b{bucket}h{history}"


class CFRTrainer:
    """External-sampling MCCFR over a heads-up NLHE abstraction."""

    def __init__(self, small_blind: int = 1, big_blind: int = 2, starting_stack: int = 200) -> None:
        self.sb = small_blind
        self.bb = big_blind
        self.starting_stack = starting_stack
        self.regret_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_CFR_ACTIONS))
        self.strategy_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(NUM_CFR_ACTIONS))

    def get_strategy(self, key: str, legal_mask: np.ndarray) -> np.ndarray:
        """Regret-matched strategy: prob ∝ max(regret, 0)."""
        regret = np.maximum(self.regret_sum[key], 0) * legal_mask
        total = regret.sum()
        if total > 0:
            return regret / total
        return legal_mask / max(legal_mask.sum(), 1)

    def get_average_strategy(self, key: str, legal_mask: np.ndarray) -> np.ndarray:
        s = self.strategy_sum[key] * legal_mask
        total = s.sum()
        if total > 0:
            return s / total
        return legal_mask / max(legal_mask.sum(), 1)

    def train(self, num_iterations: int, verbose: bool = True) -> None:
        for i in range(num_iterations):
            for player in range(2):
                deck = list(range(52))
                random.shuffle(deck)
                hole = [[deck[0], deck[1]], [deck[2], deck[3]]]
                board = deck[4:9]
                self._cfr(
                    traverser=player,
                    current=0,                      # player 0 = button/SB acts first preflop
                    hole=hole,
                    board=board,
                    revealed=0,
                    street=0,
                    history="",
                    actions_this_round=0,
                    stacks=[self.starting_stack - self.sb, self.starting_stack - self.bb],
                    bets=[self.sb, self.bb],
                    pot=self.sb + self.bb,
                    p0_reach=1.0,
                    p1_reach=1.0,
                )
            if verbose and (i + 1) % max(1, num_iterations // 20) == 0:
                print(f"  Iter {i + 1}/{num_iterations}  info_sets={len(self.regret_sum)}")

    def _legal_mask(self, current: int, bets: list[int], stacks: list[int]) -> np.ndarray:
        """Compute legal action mask."""
        mask = np.zeros(NUM_CFR_ACTIONS)
        my_bet = bets[current]
        max_bet = max(bets)
        to_call = max_bet - my_bet
        my_stack = stacks[current]

        # FOLD: only when facing a bet
        if to_call > 0:
            mask[0] = 1
        # CHECK or CALL: always (check if to_call=0, call otherwise)
        mask[1] = 1
        # BET/RAISE half-pot: only if has chips left after potential call
        if my_stack > to_call:
            mask[2] = 1
        # ALL-IN: only if has chips
        if my_stack > 0:
            mask[3] = 1

        return mask

    def _cfr(
        self,
        traverser: int,
        current: int,
        hole: list[list[int]],
        board: list[int],
        revealed: int,
        street: int,
        history: str,
        actions_this_round: int,
        stacks: list[int],
        bets: list[int],
        pot: int,
        p0_reach: float,
        p1_reach: float,
    ) -> float:
        """Returns expected payoff for traverser at this game state."""
        opponent = 1 - current
        legal_mask = self._legal_mask(current, bets, stacks)
        bucket = hand_to_bucket(hole[current], board[:revealed])
        key = info_set_key(bucket, history, street)
        strategy = self.get_strategy(key, legal_mask)

        if current != traverser:
            # Sample opponent's action; descend into single branch
            action_idx = self._sample(strategy)
            new_p0_reach = p0_reach * (strategy[action_idx] if current == 0 else 1.0)
            new_p1_reach = p1_reach * (strategy[action_idx] if current == 1 else 1.0)
            # Update opponent's strategy sum (using their reach)
            opp_reach = p0_reach if current == 0 else p1_reach
            self.strategy_sum[key] += opp_reach * strategy
            return self._next(
                action_idx, traverser, current, hole, board, revealed, street,
                history, actions_this_round, stacks, bets, pot,
                new_p0_reach, new_p1_reach,
            )

        # Traverser: evaluate all legal actions
        action_utils = np.zeros(NUM_CFR_ACTIONS)
        node_util = 0.0
        for a in range(NUM_CFR_ACTIONS):
            if legal_mask[a] == 0:
                continue
            new_p0 = p0_reach * (strategy[a] if current == 0 else 1.0)
            new_p1 = p1_reach * (strategy[a] if current == 1 else 1.0)
            util = self._next(
                a, traverser, current, hole, board, revealed, street,
                history, actions_this_round, stacks, bets, pot,
                new_p0, new_p1,
            )
            action_utils[a] = util
            node_util += strategy[a] * util

        # Update regrets weighted by opponent's reach probability
        opp_reach = p0_reach if opponent == 0 else p1_reach
        for a in range(NUM_CFR_ACTIONS):
            if legal_mask[a] == 0:
                continue
            self.regret_sum[key][a] += opp_reach * (action_utils[a] - node_util)
        # Update strategy sum weighted by traverser's own reach
        own_reach = p0_reach if traverser == 0 else p1_reach
        self.strategy_sum[key] += own_reach * strategy

        return node_util

    def _sample(self, strategy: np.ndarray) -> int:
        r = random.random()
        cum = 0.0
        for i, p in enumerate(strategy):
            cum += p
            if r < cum:
                return i
        return len(strategy) - 1

    def _next(
        self,
        action_idx: int,
        traverser: int,
        current: int,
        hole: list[list[int]],
        board: list[int],
        revealed: int,
        street: int,
        history: str,
        actions_this_round: int,
        stacks: list[int],
        bets: list[int],
        pot: int,
        p0_reach: float,
        p1_reach: float,
    ) -> float:
        """Apply action and recurse. Returns expected payoff for traverser."""
        opponent = 1 - current
        my_bet = bets[current]
        max_bet = max(bets)
        to_call = max_bet - my_bet

        new_stacks = stacks.copy()
        new_bets = bets.copy()
        new_pot = pot

        if action_idx == 0:  # FOLD
            # Opponent wins the current pot
            return self._terminal_payoff(traverser, opponent, new_pot, new_stacks)

        elif action_idx == 1:  # CHECK or CALL
            call_amount = min(to_call, new_stacks[current])
            new_stacks[current] -= call_amount
            new_bets[current] += call_amount
            new_pot += call_amount

        elif action_idx == 2:  # BET or RAISE half-pot
            # Total chips to put in: call amount + half-pot raise
            raise_size = max(int(new_pot * 0.5), self.bb)
            total_in = min(to_call + raise_size, new_stacks[current])
            new_stacks[current] -= total_in
            new_bets[current] += total_in
            new_pot += total_in

        elif action_idx == 3:  # ALL-IN
            all_in = new_stacks[current]
            new_stacks[current] = 0
            new_bets[current] += all_in
            new_pot += all_in

        new_round_count = actions_this_round + 1
        new_history = history + str(action_idx)

        # Did the round end?
        bets_matched = new_bets[0] == new_bets[1]
        someone_all_in = new_stacks[0] == 0 or new_stacks[1] == 0
        round_ends = (
            new_round_count >= 2 and (bets_matched or someone_all_in)
        )

        if not round_ends:
            # Opponent acts
            return self._cfr(
                traverser, opponent, hole, board, revealed, street,
                new_history, new_round_count,
                new_stacks, new_bets, new_pot,
                p0_reach, p1_reach,
            )

        # Round ends. Either go to showdown or advance street.
        if street == 3 or someone_all_in:
            return self._showdown(traverser, hole, board, new_pot, new_stacks)

        # Advance to next street
        next_street = street + 1
        next_revealed = 3 if next_street == 1 else (4 if next_street == 2 else 5)
        # Postflop in HU: BB (player 1) acts first
        first_to_act = 1
        new_history_with_break = new_history + "|"

        return self._cfr(
            traverser, first_to_act, hole, board, next_revealed, next_street,
            new_history_with_break, 0,
            new_stacks, [0, 0], new_pot,
            p0_reach, p1_reach,
        )

    def _terminal_payoff(self, traverser: int, winner: int, pot: int, stacks: list[int]) -> float:
        """Return traverser's net chip change (loss is negative)."""
        # Total chips traverser put in this hand:
        contributed = self.starting_stack - stacks[traverser]
        if winner == traverser:
            # They get the pot back (which includes their contribution)
            return float(pot - contributed)
        return -float(contributed)

    def _showdown(self, traverser: int, hole: list[list[int]], full_board: list[int],
                   pot: int, stacks: list[int]) -> float:
        """Compute traverser's payoff at showdown."""
        winners = determine_winners([hole[0], hole[1]], full_board)
        contributed = self.starting_stack - stacks[traverser]

        if 0 in winners and 1 in winners:
            # Tie: split pot
            return (pot / 2) - contributed
        winner = winners[0]
        if winner == traverser:
            return float(pot - contributed)
        return -float(contributed)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        # Save average strategy with legal masks already applied implicitly
        # (by using all-ones mask at save time; runtime will re-mask by legal actions)
        data = {
            k: self.get_average_strategy(k, np.ones(NUM_CFR_ACTIONS)).tolist()
            for k in self.strategy_sum
        }
        with open(path, 'w') as f:
            json.dump(data, f)


class CFRStrategy(BaseStrategy):
    """Plays from a trained CFR strategy table."""

    def __init__(self, strategy_path: str | None = None, trainer: CFRTrainer | None = None) -> None:
        if trainer is not None:
            self.strategy_table = {
                k: trainer.get_average_strategy(k, np.ones(NUM_CFR_ACTIONS))
                for k in trainer.strategy_sum
            }
        elif strategy_path is not None:
            with open(strategy_path) as f:
                raw = json.load(f)
            self.strategy_table = {k: np.array(v) for k, v in raw.items()}
        else:
            self.strategy_table = {}

    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action:
        p = state.current_player
        hole = state.hole_cards[p]
        bucket = hand_to_bucket(hole, state.board)

        # Build history from action_history split by street
        history_chars = []
        last_street_seen = None
        for act in state.action_history:
            history_chars.append(str(self._action_to_cfr_idx(act)))
        history = ''.join(history_chars)
        key = info_set_key(bucket, history, int(state.street))

        if key in self.strategy_table:
            strategy = self.strategy_table[key]
        else:
            # Default to a sensible fallback by bucket strength
            strategy = self._fallback_strategy(bucket, legal_actions)

        # Apply legal action mask
        legal_mask = np.zeros(NUM_CFR_ACTIONS)
        for i, cfr_act in enumerate(CFR_ACTIONS):
            if self._cfr_action_legal(cfr_act, legal_actions):
                legal_mask[i] = 1
        masked = strategy * legal_mask
        if masked.sum() > 0:
            masked = masked / masked.sum()
        else:
            masked = legal_mask / max(legal_mask.sum(), 1)

        action_idx = int(np.random.choice(NUM_CFR_ACTIONS, p=masked))
        return self._make_action(state, action_idx)

    def _fallback_strategy(self, bucket: int, legal: list[ActionType]) -> np.ndarray:
        """Sensible defaults when info set wasn't seen during training."""
        # Bucket 4 (premium): mostly bet/raise
        # Bucket 0 (trash): mostly fold/check
        if bucket >= 3:
            return np.array([0.0, 0.2, 0.5, 0.3])
        if bucket == 2:
            return np.array([0.05, 0.5, 0.4, 0.05])
        if bucket == 1:
            return np.array([0.3, 0.5, 0.18, 0.02])
        return np.array([0.6, 0.35, 0.04, 0.01])

    def _action_to_cfr_idx(self, action: Action) -> int:
        if action.type == ActionType.FOLD: return 0
        if action.type in (ActionType.CHECK, ActionType.CALL): return 1
        if action.type in (ActionType.BET, ActionType.RAISE): return 2
        if action.type == ActionType.ALL_IN: return 3
        return 1

    def _cfr_action_legal(self, cfr_action: ActionType, legal: list[ActionType]) -> bool:
        if cfr_action == ActionType.FOLD:
            return ActionType.FOLD in legal
        if cfr_action == ActionType.CHECK:
            return ActionType.CHECK in legal or ActionType.CALL in legal
        if cfr_action == ActionType.BET:
            return ActionType.BET in legal or ActionType.RAISE in legal
        if cfr_action == ActionType.ALL_IN:
            return ActionType.ALL_IN in legal
        return False

    def _make_action(self, state: GameState, cfr_idx: int) -> Action:
        p = state.current_player
        max_bet = max(state.current_bets) if state.current_bets else 0
        my_bet = state.current_bets[p] if state.current_bets else 0
        to_call = max_bet - my_bet
        pot = state.pot

        if cfr_idx == 0:
            return Action(ActionType.FOLD, 0, p)
        if cfr_idx == 1:
            if to_call > 0:
                return Action(ActionType.CALL, to_call, p)
            return Action(ActionType.CHECK, 0, p)
        if cfr_idx == 2:
            amount = max(int(pot * 0.5), state.big_blind)
            if to_call > 0:
                return Action(ActionType.RAISE, amount, p)
            return Action(ActionType.BET, amount, p)
        if cfr_idx == 3:
            return Action(ActionType.ALL_IN, state.stacks[p], p)
        return Action(ActionType.FOLD, 0, p)
