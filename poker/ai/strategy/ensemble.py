"""Ensemble strategy: weighted vote across heuristic + CFR + DQN.

Each sub-strategy proposes an action; the ensemble picks the most-confident
action. If a model isn't trained/loaded, it's skipped.
"""

from __future__ import annotations

import os
from collections import Counter

from ..engine.action import Action, ActionType
from ..engine.game_state import GameState
from .base import BaseStrategy, PokerStrategy


class EnsembleStrategy(BaseStrategy):
    """Weighted majority vote over multiple strategies.

    Default weights favor the heuristic (most reliable baseline); CFR and DQN
    add their votes when models are available.
    """

    def __init__(
        self,
        cfr_path: str | None = 'poker/ai/models/cfr_strategy.json',
        dqn_path: str | None = 'poker/ai/models/dqn_weights.npz',
        weights: dict[str, float] | None = None,
        heuristic_sims: int = 100,
    ) -> None:
        from .heuristic import HeuristicStrategy

        self.weights = weights or {'heuristic': 2.0, 'cfr': 1.0, 'dqn': 1.0}
        self.strategies: dict[str, PokerStrategy] = {
            'heuristic': HeuristicStrategy(num_simulations=heuristic_sims),
        }

        if cfr_path and os.path.exists(cfr_path):
            from .cfr import CFRStrategy
            self.strategies['cfr'] = CFRStrategy(strategy_path=cfr_path)

        if dqn_path and os.path.exists(dqn_path):
            from .dqn import DQNAgent
            agent = DQNAgent(training=False)
            agent.load(dqn_path)
            self.strategies['dqn'] = agent

    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action:
        votes: dict[ActionType, float] = Counter()
        proposed: dict[ActionType, list[Action]] = {}

        for name, strat in self.strategies.items():
            try:
                action = strat.choose_action(state, legal_actions)
                weight = self.weights.get(name, 1.0)
                votes[action.type] += weight
                proposed.setdefault(action.type, []).append(action)
            except Exception:
                continue

        if not votes:
            return Action(ActionType.FOLD, 0, state.current_player)

        best_type = max(votes, key=lambda k: votes[k])
        # Among proposers of the winning action type, pick the median amount
        candidates = sorted(proposed[best_type], key=lambda a: a.amount)
        return candidates[len(candidates) // 2]
