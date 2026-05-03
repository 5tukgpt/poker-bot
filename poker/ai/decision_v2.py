"""Decision shim — drop-in replacement for poker.decisionmaker.decisionmaker.Decision.

Wraps a PokerStrategy so it presents the same interface as the original
615-line Decision class but routes through our AI engine.

Usage in main.py (replaces lines 206-207):
    # OLD:
    # d = Decision(table, history, strategy, self.game_logger)
    # d.make_decision(table, history, strategy, self.game_logger)

    # NEW:
    from poker.ai.decision_v2 import DecisionV2
    d = DecisionV2(table, strategy_engine=my_strategy)
    d.make_decision(table, history, strategy, self.game_logger)
"""

from __future__ import annotations

import logging
from typing import Any

from .adapter import action_to_dickreuter, to_game_state
from .strategy.base import PokerStrategy

log = logging.getLogger(__name__)


class DecisionV2:
    """Drop-in replacement for the original Decision class.

    Exposes the attributes main.py expects:
      - decision: str (e.g. 'Fold', 'Call', 'Bet half pot')
      - finalCallLimit: float
      - finalBetLimit: float
      - maxCallEV: float
      - outs: int
      - pot_multiple: float
    """

    def __init__(
        self,
        table: Any,
        strategy_engine: PokerStrategy | None = None,
        bb_dollar_value: float = 0.02,
    ) -> None:
        self.table = table
        self.strategy_engine = strategy_engine
        self.bb_dollar_value = bb_dollar_value

        # Attributes that main.py reads
        self.decision: str = 'Fold'
        self.finalCallLimit: float = 0.0
        self.finalBetLimit: float = 0.0
        self.maxCallEV: float = 0.0
        self.outs: int = 0
        self.pot_multiple: float = 0.0
        self.DeriveCallButtonFromBetButton = False

    def make_decision(self, table: Any, history: Any, strategy: Any, game_logger: Any) -> None:
        """Compute decision via our strategy engine."""
        if self.strategy_engine is None:
            log.error("DecisionV2: no strategy_engine set, defaulting to Fold")
            self.decision = 'Fold'
            return

        try:
            config = getattr(strategy, 'selected_strategy', {}) or {}
            state = to_game_state(table, config, self.bb_dollar_value)
        except Exception as e:
            log.error(f"DecisionV2: failed to translate state: {e}")
            self.decision = 'Fold'
            return

        legal = state.legal_actions()
        if not legal:
            self.decision = 'Fold'
            return

        try:
            action = self.strategy_engine.choose_action(state, legal)
        except Exception as e:
            log.error(f"DecisionV2: strategy raised: {e}")
            self.decision = 'Fold'
            return

        self.decision = action_to_dickreuter(action)
        self.finalCallLimit = float(action.amount) * self.bb_dollar_value / 2
        self.finalBetLimit = float(action.amount) * self.bb_dollar_value / 2

        log.info(f"DecisionV2: street={state.street.name} action={self.decision} amount={action.amount}")
