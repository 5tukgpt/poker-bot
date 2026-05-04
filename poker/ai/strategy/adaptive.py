"""Adaptive strategy: tracks opponent type live, picks counter-strategy.

This is the "play anyone" strategy. It observes opponent behavior over time,
classifies them as NIT/TAG/LAG/FISH/MANIAC/UNKNOWN, and picks the best
strategy from our pool to exploit them.

Default strategy mapping:
  UNKNOWN  → heuristic (safe baseline while gathering info)
  FISH     → heuristic (value bet, never bluff — they call too much)
  NIT      → book      (more bluffs/aggression — they fold too much)
  LAG      → heuristic (don't bluff back — call down with strength)
  TAG      → dqn       (most balanced vs solid players)
  MANIAC   → heuristic (call wider, don't bluff into them)
"""

from __future__ import annotations

import os

from ..engine.action import Action, ActionType
from ..engine.game_state import GameState
from .base import BaseStrategy, PokerStrategy
from .heuristic import HeuristicStrategy
from .opponent_model import OpponentStats, PlayerType


# Strategy name → strategy mapping for each opponent type.
#
# Empirically tuned via cross-product benchmark (scripts/cross_product.py)
# at 1000 hands per matchup. Best counter per in-house bot:
#   vs heuristic-style (tight value player) → DQN (+43 BB/100)
#   vs dqn-style (aggressive RL player)     → book (-7 BB/100, least bad)
#   vs gto_chart-style (TAG GTO player)     → heuristic (+36 BB/100)
#   vs book-style (LAG aggressive player)   → heuristic (+57 BB/100)
#
# Strongest overall: DQN at +19 BB/100 average. Use as default for unknown.
# Confirmed against Slumbot: NIT classification → DQN gives best result (-26 BB/100).
DEFAULT_TYPE_TO_STRATEGY = {
    PlayerType.UNKNOWN: 'dqn',         # strongest overall — safe default
    PlayerType.NIT:     'dqn',         # tight opponents — DQN exploits well
    PlayerType.TAG:     'heuristic',   # GTO-style players — heuristic surprisingly beats them
    PlayerType.LAG:     'heuristic',   # aggressive players — call down with equity
    PlayerType.FISH:    'heuristic',   # value-bet stations
    PlayerType.MANIAC:  'heuristic',   # call wider, don't bluff into them
}


class AdaptiveStrategy(BaseStrategy):
    """Plays based on observed opponent type. Tracks stats across hands.

    Usage: create ONCE per opponent. Re-use across many hands so stats accumulate.
    For tournaments where opponents change, use one AdaptiveStrategy per seat.
    """

    def __init__(
        self,
        type_to_strategy: dict[PlayerType, str] | None = None,
        opponent_name: str = "opponent",
        verbose_switching: bool = False,
    ) -> None:
        self.type_to_strategy = type_to_strategy or DEFAULT_TYPE_TO_STRATEGY
        self.opp_stats = OpponentStats(name=opponent_name)
        self.verbose_switching = verbose_switching
        self._last_observed_idx = 0  # how many history actions we've already observed
        self._last_strategy_name: str | None = None

        # Lazy-load strategies on first use to avoid startup cost
        self._strategies: dict[str, PokerStrategy] = {}

    def _get_strategy(self, name: str) -> PokerStrategy:
        if name in self._strategies:
            return self._strategies[name]
        if name == 'heuristic':
            self._strategies[name] = HeuristicStrategy(num_simulations=150)
        elif name == 'gto_chart':
            from .gto_chart import GTOChartStrategy
            self._strategies[name] = GTOChartStrategy(postflop_sims=150)
        elif name == 'book':
            from .book_strategy import BookStrategy
            self._strategies[name] = BookStrategy(postflop_sims=150)
        elif name == 'dqn':
            from .dqn import DQNAgent
            agent = DQNAgent(training=False)
            dqn_path = 'poker/ai/models/dqn_weights.npz'
            if os.path.exists(dqn_path):
                agent.load(dqn_path)
            self._strategies[name] = agent
        else:
            # Unknown name — fallback
            self._strategies[name] = HeuristicStrategy(num_simulations=150)
        return self._strategies[name]

    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action:
        # Observe any new opponent actions since last call
        self._observe_new_actions(state)

        # Classify opponent and pick strategy
        opp_type = self.opp_stats.player_type
        strategy_name = self.type_to_strategy.get(opp_type, 'heuristic')

        if self.verbose_switching and strategy_name != self._last_strategy_name:
            print(f"[Adaptive] switching to {strategy_name} (opp type: {opp_type.value}, "
                  f"hands: {self.opp_stats.hands_observed})")
            self._last_strategy_name = strategy_name

        strategy = self._get_strategy(strategy_name)
        return strategy.choose_action(state, legal_actions)

    def _observe_new_actions(self, state: GameState) -> None:
        """Process opponent actions in action_history we haven't seen yet."""
        my_player = state.current_player
        for action in state.action_history[self._last_observed_idx:]:
            if action.player_idx != my_player:
                self.opp_stats.observe_action(state, action)
        self._last_observed_idx = len(state.action_history)

    def notify_result(self, state: GameState, payoff: int) -> None:
        # Observe any final opponent actions
        self._observe_new_actions(state)
        # Commit hand to stats
        self.opp_stats.end_hand()
        # Reset per-hand tracking
        self._last_observed_idx = 0
        # Forward result to all sub-strategies (for those that learn, like DQN)
        for s in self._strategies.values():
            s.notify_result(state, payoff)

    def get_opponent_summary(self) -> str:
        return self.opp_stats.summary()
