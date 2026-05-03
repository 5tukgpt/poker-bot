"""Counterfactual Regret Minimization for heads-up NLHE with abstraction.

Card abstraction: 5 buckets based on preflop equity (premium → trash).
Action abstraction: 4 actions (FOLD, CHECK_CALL, BET_HALF_POT, ALL_IN).
History string encodes the action sequence for info set lookup.
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


# Action abstraction: indices 0-3
CFR_ACTIONS = [
    ActionType.FOLD,
    ActionType.CHECK,    # also used for call when facing bet
    ActionType.BET,      # bet/raise half pot
    ActionType.ALL_IN,
]
NUM_CFR_ACTIONS = len(CFR_ACTIONS)
NUM_BUCKETS = 5


# Pre-computed preflop equity buckets (heads-up vs random hand)
# 0 = trash, 4 = premium
def hand_to_bucket(hole: list[int], board: list[int]) -> int:
    """Map (hole, board) to 1 of 5 strength buckets."""
    if len(board) == 0:
        # Preflop: use simple high card + pair logic
        c1, c2 = Card.from_int(hole[0]), Card.from_int(hole[1])
        high = max(c1.rank, c2.rank)
        low = min(c1.rank, c2.rank)
        suited = c1.suit == c2.suit

        if c1.rank == c2.rank:
            if c1.rank >= 11:  # JJ+
                return 4
            if c1.rank >= 8:   # 88-TT
                return 3
            return 2
        if high == 14:  # ace high
            if low >= 11 or (suited and low >= 9):
                return 4 if low >= 12 else 3
            return 2 if suited else 1
        if high >= 12 and low >= 10:  # KQ, KJ, QJ
            return 3 if suited else 2
        if suited and (high - low <= 2) and high >= 8:
            return 2
        if high >= 11:
            return 1
        return 0

    # Postflop: estimate equity by hand rank percentile
    rank = evaluate_hand(hole, board)
    # phevaluator ranks: 1 (royal flush) to 7462 (worst)
    # Lower = better. Map to 5 buckets.
    if rank <= 322:      # straight or better
        return 4
    if rank <= 1609:     # flush
        return 4
    if rank <= 2467:     # straight or three of a kind
        return 3
    if rank <= 3325:     # two pair
        return 3
    if rank <= 6185:     # one pair
        return 2
    if rank <= 7000:     # decent high card
        return 1
    return 0


def history_string(actions: list[int], street_breaks: list[int]) -> str:
    """Encode action history. street_breaks marks where new streets start."""
    parts = []
    breaks_set = set(street_breaks)
    for i, a in enumerate(actions):
        if i in breaks_set:
            parts.append('|')
        parts.append(str(a))
    return ''.join(parts)


def info_set_key(bucket: int, history: str, street: int) -> str:
    return f"s{street}b{bucket}h{history}"


class CFRTrainer:
    """Train a CFR strategy via external-sampling MCCFR over an abstracted game."""

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
            strategy = regret / total
        else:
            strategy = legal_mask / max(legal_mask.sum(), 1)
        return strategy

    def get_average_strategy(self, key: str, legal_mask: np.ndarray) -> np.ndarray:
        s = self.strategy_sum[key] * legal_mask
        total = s.sum()
        if total > 0:
            return s / total
        return legal_mask / max(legal_mask.sum(), 1)

    def train(self, num_iterations: int, verbose: bool = True) -> None:
        for i in range(num_iterations):
            for player in range(2):
                self._cfr_iteration(player)
            if verbose and (i + 1) % max(1, num_iterations // 20) == 0:
                print(f"  Iter {i + 1}/{num_iterations}  info_sets={len(self.regret_sum)}")

    def _cfr_iteration(self, traverser: int) -> None:
        deck = list(range(52))
        random.shuffle(deck)
        hole = [[deck[0], deck[1]], [deck[2], deck[3]]]
        # Pre-deal full board for sampling efficiency
        full_board = deck[4:9]

        self._traverse(
            traverser=traverser,
            current_player=0,  # button (SB) acts first preflop in HU
            hole=hole,
            full_board=full_board,
            board_revealed=0,
            street=0,
            actions=[],
            street_breaks=[],
            stacks=[self.starting_stack - self.sb, self.starting_stack - self.bb],
            current_bets=[self.sb, self.bb],
            pot=self.sb + self.bb,
            reach=[1.0, 1.0],
        )

    def _traverse(
        self,
        traverser: int,
        current_player: int,
        hole: list[list[int]],
        full_board: list[int],
        board_revealed: int,
        street: int,
        actions: list[int],
        street_breaks: list[int],
        stacks: list[int],
        current_bets: list[int],
        pot: int,
        reach: list[float],
    ) -> float:
        # Terminal: only one player remains
        # We track active by checking if anyone has folded (encoded as last action == 0 ending the hand)
        # For simplicity, we treat showdown when both checked through to river
        opponent = 1 - current_player

        # Determine legal actions
        max_bet = max(current_bets)
        my_bet = current_bets[current_player]
        to_call = max_bet - my_bet
        my_stack = stacks[current_player]

        legal_mask = np.zeros(NUM_CFR_ACTIONS)
        # FOLD only legal if facing a bet
        if to_call > 0:
            legal_mask[0] = 1
        # CHECK/CALL always legal if has chips or no bet to call
        legal_mask[1] = 1
        # BET half-pot if can afford
        if my_stack > 0:
            legal_mask[2] = 1
        # ALL_IN
        if my_stack > 0:
            legal_mask[3] = 1

        board = full_board[:board_revealed]
        bucket = hand_to_bucket(hole[current_player], board)
        history = history_string(actions, street_breaks)
        key = info_set_key(bucket, history, street)

        strategy = self.get_strategy(key, legal_mask)

        # If this is not the traverser, sample a single action
        if current_player != traverser:
            action_idx = self._sample_action(strategy)
            return self._take_action(
                action_idx, traverser, current_player, hole, full_board,
                board_revealed, street, actions, street_breaks,
                stacks, current_bets, pot, reach, strategy,
            )

        # Traverser: try all actions, accumulate regret
        action_utils = np.zeros(NUM_CFR_ACTIONS)
        node_util = 0.0
        for a in range(NUM_CFR_ACTIONS):
            if legal_mask[a] == 0:
                continue
            new_reach = reach.copy()
            new_reach[traverser] *= strategy[a]
            util = self._take_action(
                a, traverser, current_player, hole, full_board,
                board_revealed, street, actions.copy(), street_breaks.copy(),
                stacks.copy(), current_bets.copy(), pot, new_reach, strategy,
            )
            action_utils[a] = util
            node_util += strategy[a] * util

        # Update regrets
        opp_reach = reach[opponent]
        for a in range(NUM_CFR_ACTIONS):
            if legal_mask[a] == 0:
                continue
            self.regret_sum[key][a] += opp_reach * (action_utils[a] - node_util)

        # Update strategy sum
        self.strategy_sum[key] += reach[traverser] * strategy

        return node_util

    def _sample_action(self, strategy: np.ndarray) -> int:
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(strategy):
            cumulative += p
            if r < cumulative:
                return i
        return len(strategy) - 1

    def _take_action(
        self,
        action_idx: int,
        traverser: int,
        current_player: int,
        hole: list[list[int]],
        full_board: list[int],
        board_revealed: int,
        street: int,
        actions: list[int],
        street_breaks: list[int],
        stacks: list[int],
        current_bets: list[int],
        pot: int,
        reach: list[float],
        strategy: np.ndarray,
    ) -> float:
        opponent = 1 - current_player
        new_actions = actions + [action_idx]
        max_bet = max(current_bets)
        my_bet = current_bets[current_player]
        to_call = max_bet - my_bet

        new_stacks = stacks.copy()
        new_bets = current_bets.copy()
        new_pot = pot

        if action_idx == 0:  # FOLD
            return self._payoff(opponent, traverser, new_pot, stacks)

        if action_idx == 1:  # CHECK/CALL
            call_amount = min(to_call, new_stacks[current_player])
            new_stacks[current_player] -= call_amount
            new_bets[current_player] += call_amount
            new_pot += call_amount

        elif action_idx == 2:  # BET half-pot
            bet_amount = max(int(new_pot * 0.5), self.bb)
            bet_amount = min(bet_amount + to_call, new_stacks[current_player])
            new_stacks[current_player] -= bet_amount
            new_bets[current_player] += bet_amount
            new_pot += bet_amount

        elif action_idx == 3:  # ALL_IN
            all_in_amount = new_stacks[current_player]
            new_stacks[current_player] = 0
            new_bets[current_player] += all_in_amount
            new_pot += all_in_amount

        # Determine if street ends
        new_max = max(new_bets)
        opp_matched = new_bets[opponent] == new_max or new_stacks[opponent] == 0
        action_closes = (action_idx in (1, 2, 3) and opp_matched and len(new_actions) >= 2)

        # Bet/raise reopens action; check-call or all-in call closes
        if action_idx == 1 and to_call == 0 and len(new_actions) < 2:
            # Initial check, opponent still acts
            return self._traverse(
                traverser, opponent, hole, full_board, board_revealed, street,
                new_actions, street_breaks, new_stacks, new_bets, new_pot, reach,
            )

        if action_idx in (2, 3) or to_call > 0:
            # Bet or call after a bet; opponent must respond if not all-in
            if new_stacks[opponent] > 0 and not (action_idx == 1 and to_call == 0):
                if action_idx in (2, 3):
                    # Opponent needs to respond to the bet
                    return self._traverse(
                        traverser, opponent, hole, full_board, board_revealed, street,
                        new_actions, street_breaks, new_stacks, new_bets, new_pot, reach,
                    )

        # Street ends: advance or showdown
        if action_idx == 1 and to_call > 0:
            # Call closes the street
            pass
        elif action_idx == 1 and to_call == 0 and len(new_actions) >= 2:
            # Both checked, street ends
            pass

        # Advance to next street or showdown
        if street == 3 or new_stacks[0] == 0 or new_stacks[1] == 0:
            return self._showdown(traverser, hole, full_board, new_pot, stacks)

        new_street = street + 1
        new_revealed = 3 if new_street == 1 else (4 if new_street == 2 else 5)
        new_breaks = street_breaks + [len(new_actions)]
        new_bets_reset = [0, 0]

        # Postflop: BB acts first in HU (opponent of button)
        next_player = 1  # BB position (player 1 in our setup)
        return self._traverse(
            traverser, next_player, hole, full_board, new_revealed, new_street,
            new_actions, new_breaks, new_stacks, new_bets_reset, new_pot, reach,
        )

    def _payoff(self, winner: int, traverser: int, pot: int, original_stacks: list[int]) -> float:
        # Payoff is from traverser's perspective
        # original_stacks reflects what was contributed before this branch
        contributed = (self.starting_stack - original_stacks[traverser]) - (
            self.sb if traverser == 0 else self.bb
        )
        if winner == traverser:
            return float(pot - (self.sb if traverser == 0 else self.bb) - contributed)
        return -float(self.sb if traverser == 0 else self.bb) - contributed

    def _showdown(self, traverser: int, hole: list[list[int]], full_board: list[int],
                   pot: int, stacks: list[int]) -> float:
        winners = determine_winners([hole[0], hole[1]], full_board)
        if 0 in winners and 1 in winners:
            # Split pot
            traverser_share = pot / 2
            initial_blind = self.sb if traverser == 0 else self.bb
            contributed = self.starting_stack - stacks[traverser]
            return traverser_share - contributed
        winner = winners[0]
        return self._payoff(winner, traverser, pot, stacks)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            k: self.get_average_strategy(k, np.ones(NUM_CFR_ACTIONS)).tolist()
            for k in self.strategy_sum
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str) -> dict[str, np.ndarray]:
        with open(path) as f:
            raw = json.load(f)
        return {k: np.array(v) for k, v in raw.items()}


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
        history_actions = [self._action_to_cfr_idx(a) for a in state.action_history]
        history = ''.join(str(a) for a in history_actions)
        key = info_set_key(bucket, history, int(state.street))

        if key in self.strategy_table:
            strategy = self.strategy_table[key]
        else:
            # Default: uniform over legal actions
            strategy = np.ones(NUM_CFR_ACTIONS) / NUM_CFR_ACTIONS

        # Mask illegal actions
        legal_mask = np.zeros(NUM_CFR_ACTIONS)
        for i, cfr_act in enumerate(CFR_ACTIONS):
            if self._cfr_action_legal(cfr_act, legal_actions):
                legal_mask[i] = 1
        masked = strategy * legal_mask
        if masked.sum() > 0:
            masked /= masked.sum()
        else:
            masked = legal_mask / max(legal_mask.sum(), 1)

        action_idx = int(np.random.choice(NUM_CFR_ACTIONS, p=masked))
        return self._make_action(state, action_idx)

    def _action_to_cfr_idx(self, action: Action) -> int:
        if action.type == ActionType.FOLD:
            return 0
        if action.type in (ActionType.CHECK, ActionType.CALL):
            return 1
        if action.type in (ActionType.BET, ActionType.RAISE):
            return 2
        if action.type == ActionType.ALL_IN:
            return 3
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
