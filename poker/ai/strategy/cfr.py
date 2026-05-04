"""Counterfactual Regret Minimization for heads-up NLHE.

Card abstraction: k-means clustering on equity histograms (50+ buckets per street),
loaded from poker/ai/models/abstraction.json. Falls back to legacy 5-bucket
rule-based abstraction if the file is missing.

Action abstraction: 6 actions (FOLD, CHECK_CALL, BET 33%, BET 67%, BET 150%, ALL_IN).

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

# Lazy import — only loaded if abstraction file exists
_BUCKETER = None
_BUCKETER_PATH = 'poker/ai/models/abstraction.json'


CFR_ACTIONS = [
    ActionType.FOLD,
    ActionType.CHECK,         # check or call
    ActionType.BET,           # bet 33% pot (small)
    ActionType.RAISE,         # bet 67% pot (standard)
    ActionType.BET,           # bet 150% pot (overbet) — same enum but distinct CFR index
    ActionType.ALL_IN,
]
NUM_CFR_ACTIONS = len(CFR_ACTIONS)

# Bet sizing per CFR action index (as fraction of pot for indices 2, 3, 4)
BET_SIZE_FRACTIONS = {2: 0.33, 3: 0.67, 4: 1.5}


def _load_bucketer():
    """Lazily load the EquityBucketer if abstraction file exists."""
    global _BUCKETER
    if _BUCKETER is not None:
        return _BUCKETER
    if os.path.exists(_BUCKETER_PATH):
        from .abstraction import EquityBucketer
        _BUCKETER = EquityBucketer.load(_BUCKETER_PATH)
        return _BUCKETER
    return None


def _legacy_bucket(hole: list[int], board: list[int]) -> int:
    """Fallback 5-bucket rule-based abstraction if no precomputed file exists."""
    if len(board) == 0:
        c1, c2 = Card.from_int(hole[0]), Card.from_int(hole[1])
        high = max(c1.rank, c2.rank)
        low = min(c1.rank, c2.rank)
        suited = c1.suit == c2.suit
        if c1.rank == c2.rank:
            if c1.rank >= 12: return 4
            if c1.rank >= 9: return 3
            if c1.rank >= 6: return 2
            return 1
        if high == 14:
            if low >= 12: return 4
            if low >= 10 or (suited and low >= 8): return 3
            if suited: return 2
            return 1
        if high >= 12 and low >= 10:
            return 3 if suited else 2
        if suited and (high - low <= 2) and high >= 9:
            return 2
        if high >= 11:
            return 1
        return 0
    rank = evaluate_hand(hole, board)
    if rank <= 1609: return 4
    if rank <= 3325: return 3
    if rank <= 6185: return 2
    if rank <= 7000: return 1
    return 0


def hand_to_bucket(hole: list[int], board: list[int]) -> int:
    """Map (hole, board) → bucket index.

    Hybrid: k-means clustering for preflop (50 buckets, much richer than rule-based),
    rule-based for postflop (deterministic, fast, fewer info sets to converge).
    """
    if len(board) == 0:
        bucketer = _load_bucketer()
        if bucketer is not None:
            return bucketer.bucket(hole, board, street=0)
    return _legacy_bucket(hole, board)


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
                    current=0,
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

    def _legal_mask(self, current: int, bets: list[int], stacks: list[int], pot: int) -> np.ndarray:
        """Compute legal action mask for the current player."""
        mask = np.zeros(NUM_CFR_ACTIONS)
        my_bet = bets[current]
        max_bet = max(bets)
        to_call = max_bet - my_bet
        my_stack = stacks[current]

        # FOLD: only when facing a bet
        if to_call > 0:
            mask[0] = 1
        # CHECK or CALL: always
        mask[1] = 1
        # BET sizes: only if has chips left after potential call AND raise size > min raise
        if my_stack > to_call:
            for idx in (2, 3, 4):
                raise_size = max(int(pot * BET_SIZE_FRACTIONS[idx]), self.bb)
                # Only legal if raise size is strictly less than going all-in
                # (otherwise it collapses with ALL_IN action)
                if to_call + raise_size < my_stack:
                    mask[idx] = 1
        # ALL-IN: only if has chips
        if my_stack > 0:
            mask[5] = 1

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
        opponent = 1 - current
        legal_mask = self._legal_mask(current, bets, stacks, pot)
        bucket = hand_to_bucket(hole[current], board[:revealed])
        key = info_set_key(bucket, history, street)
        strategy = self.get_strategy(key, legal_mask)

        if current != traverser:
            action_idx = self._sample(strategy)
            new_p0 = p0_reach * (strategy[action_idx] if current == 0 else 1.0)
            new_p1 = p1_reach * (strategy[action_idx] if current == 1 else 1.0)
            opp_reach = p0_reach if current == 0 else p1_reach
            self.strategy_sum[key] += opp_reach * strategy
            return self._next(
                action_idx, traverser, current, hole, board, revealed, street,
                history, actions_this_round, stacks, bets, pot,
                new_p0, new_p1,
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

        # Update regrets
        opp_reach = p0_reach if opponent == 0 else p1_reach
        for a in range(NUM_CFR_ACTIONS):
            if legal_mask[a] == 0:
                continue
            self.regret_sum[key][a] += opp_reach * (action_utils[a] - node_util)
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
        opponent = 1 - current
        my_bet = bets[current]
        max_bet = max(bets)
        to_call = max_bet - my_bet

        new_stacks = stacks.copy()
        new_bets = bets.copy()
        new_pot = pot

        if action_idx == 0:  # FOLD
            return self._terminal_payoff(traverser, opponent, new_pot, new_stacks)

        elif action_idx == 1:  # CHECK / CALL
            call_amount = min(to_call, new_stacks[current])
            new_stacks[current] -= call_amount
            new_bets[current] += call_amount
            new_pot += call_amount

        elif action_idx in (2, 3, 4):  # BET sized
            raise_size = max(int(new_pot * BET_SIZE_FRACTIONS[action_idx]), self.bb)
            total_in = min(to_call + raise_size, new_stacks[current])
            new_stacks[current] -= total_in
            new_bets[current] += total_in
            new_pot += total_in

        elif action_idx == 5:  # ALL-IN
            all_in = new_stacks[current]
            new_stacks[current] = 0
            new_bets[current] += all_in
            new_pot += all_in

        new_round_count = actions_this_round + 1
        new_history = history + str(action_idx)

        bets_matched = new_bets[0] == new_bets[1]
        someone_all_in = new_stacks[0] == 0 or new_stacks[1] == 0
        round_ends = new_round_count >= 2 and (bets_matched or someone_all_in)

        if not round_ends:
            return self._cfr(
                traverser, opponent, hole, board, revealed, street,
                new_history, new_round_count,
                new_stacks, new_bets, new_pot,
                p0_reach, p1_reach,
            )

        if street == 3 or someone_all_in:
            return self._showdown(traverser, hole, board, new_pot, new_stacks)

        next_street = street + 1
        next_revealed = 3 if next_street == 1 else (4 if next_street == 2 else 5)
        first_to_act = 1
        new_history_with_break = new_history + "|"

        return self._cfr(
            traverser, first_to_act, hole, board, next_revealed, next_street,
            new_history_with_break, 0,
            new_stacks, [0, 0], new_pot,
            p0_reach, p1_reach,
        )

    def _terminal_payoff(self, traverser: int, winner: int, pot: int, stacks: list[int]) -> float:
        contributed = self.starting_stack - stacks[traverser]
        if winner == traverser:
            return float(pot - contributed)
        return -float(contributed)

    def _showdown(self, traverser: int, hole: list[list[int]], full_board: list[int],
                   pot: int, stacks: list[int]) -> float:
        winners = determine_winners([hole[0], hole[1]], full_board)
        contributed = self.starting_stack - stacks[traverser]
        if 0 in winners and 1 in winners:
            return (pot / 2) - contributed
        winner = winners[0]
        if winner == traverser:
            return float(pot - contributed)
        return -float(contributed)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
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
            # Auto-detect old 4-action models: pad to 6 actions if needed
            for k, v in list(self.strategy_table.items()):
                if len(v) == 4:
                    self.strategy_table[k] = np.array([v[0], v[1], 0, v[2], 0, v[3]])
        else:
            self.strategy_table = {}

    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action:
        p = state.current_player
        hole = state.hole_cards[p]
        bucket = hand_to_bucket(hole, state.board)

        history = ''.join(str(self._action_to_cfr_idx(a)) for a in state.action_history)
        key = info_set_key(bucket, history, int(state.street))

        if key in self.strategy_table:
            strategy = self.strategy_table[key]
        else:
            strategy = self._fallback_strategy(bucket)

        # Apply legal action mask
        legal_mask = np.zeros(NUM_CFR_ACTIONS)
        for i, cfr_act in enumerate(CFR_ACTIONS):
            if self._cfr_action_legal(i, legal_actions):
                legal_mask[i] = 1

        masked = strategy * legal_mask
        if masked.sum() > 0:
            masked = masked / masked.sum()
        else:
            masked = legal_mask / max(legal_mask.sum(), 1)

        action_idx = int(np.random.choice(NUM_CFR_ACTIONS, p=masked))
        return self._make_action(state, action_idx)

    def _fallback_strategy(self, bucket: int) -> np.ndarray:
        """Sensible defaults for unseen info sets, scaled by hand strength.

        We don't know exact bucket count when loading, so use heuristic:
        - low bucket index → likely weak (relative to total)
        - high bucket index → likely strong
        """
        # Default: uniform over non-fold actions
        return np.array([0.05, 0.4, 0.15, 0.2, 0.1, 0.1])

    def _action_to_cfr_idx(self, action: Action) -> int:
        if action.type == ActionType.FOLD: return 0
        if action.type in (ActionType.CHECK, ActionType.CALL): return 1
        if action.type in (ActionType.BET, ActionType.RAISE):
            # Could be any of 2, 3, 4 — we use middle as default
            return 3
        if action.type == ActionType.ALL_IN: return 5
        return 1

    def _cfr_action_legal(self, cfr_idx: int, legal: list[ActionType]) -> bool:
        if cfr_idx == 0:
            return ActionType.FOLD in legal
        if cfr_idx == 1:
            return ActionType.CHECK in legal or ActionType.CALL in legal
        if cfr_idx in (2, 3, 4):
            return ActionType.BET in legal or ActionType.RAISE in legal
        if cfr_idx == 5:
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
        if cfr_idx in (2, 3, 4):
            fraction = BET_SIZE_FRACTIONS[cfr_idx]
            amount = max(int(pot * fraction), state.big_blind)
            if to_call > 0:
                return Action(ActionType.RAISE, amount, p)
            return Action(ActionType.BET, amount, p)
        if cfr_idx == 5:
            return Action(ActionType.ALL_IN, state.stacks[p], p)
        return Action(ActionType.FOLD, 0, p)
