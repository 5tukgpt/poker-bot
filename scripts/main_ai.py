"""Parallel entry point demonstrating how to wire our AI strategies into the
dickreuter platform game loop.

This file shows the integration pattern WITHOUT modifying the original main.py.
When you have a working poker client + scraper config, swap your strategy of
choice into the `STRATEGY` global below and adapt main.py accordingly.

To actually run live:
  1. Configure a table scraper via the dickreuter GUI (or load a table_dict)
  2. Replace lines 206-207 of poker/main.py with:
        from poker.ai.decision_v2 import DecisionV2
        from scripts.main_ai import build_strategy
        d = DecisionV2(table, strategy_engine=build_strategy())
        d.make_decision(table, history, strategy, self.game_logger)
  3. Remove the MongoDB dependency or run a local backend
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, '.')

from poker.ai.strategy.base import PokerStrategy


def build_strategy(name: str = 'heuristic') -> PokerStrategy:
    """Factory: returns a configured PokerStrategy instance.

    Options: 'heuristic' (recommended baseline), 'cfr', 'dqn', 'ensemble'.
    """
    if name == 'heuristic':
        from poker.ai.strategy.heuristic import HeuristicStrategy
        return HeuristicStrategy(num_simulations=200)

    if name == 'cfr':
        from poker.ai.strategy.cfr import CFRStrategy
        path = 'poker/ai/models/cfr_strategy.json'
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CFR strategy not trained. Run: python poker/ai/train/train_cfr.py"
            )
        return CFRStrategy(strategy_path=path)

    if name == 'dqn':
        from poker.ai.strategy.dqn import DQNAgent
        path = 'poker/ai/models/dqn_weights.npz'
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DQN not trained. Run: python poker/ai/train/train_dqn.py"
            )
        agent = DQNAgent(training=False)
        agent.load(path)
        return agent

    if name == 'ensemble':
        from poker.ai.strategy.ensemble import EnsembleStrategy
        return EnsembleStrategy()

    raise ValueError(f"Unknown strategy: {name}")


if __name__ == '__main__':
    # Smoke test the wiring without a real platform
    from types import SimpleNamespace
    from poker.ai.adapter import to_game_state
    from poker.ai.decision_v2 import DecisionV2

    name = sys.argv[1] if len(sys.argv) > 1 else 'heuristic'

    print(f"Loading strategy: {name}")
    strategy = build_strategy(name)

    # Simulate a minimal scraper output
    fake_table = SimpleNamespace(
        mycards=['As', 'Kh'],
        cardsOnTable=[],
        gameStage='PreFlop',
        totalPotValue=0.06,
        round_pot_value=0.06,
        bigBlind=0.02,
        smallBlind=0.01,
        minCall=0.02,
        currentCallValue=0.02,
        currentBetValue=0.02,
        dealer_position=0,
        myFunds=2.0,
        other_players=[{'funds': 2.0, 'pot': 0.02, 'status': 1, 'name': 'Villain'}],
        checkButton=False,
        allInCallButton=False,
    )

    state = to_game_state(fake_table)
    print(f"GameState: street={state.street.name} pot={state.pot} stacks={state.stacks}")
    print(f"Hole cards: {[hex(c) for c in state.hole_cards[0]]}")

    decision = DecisionV2(fake_table, strategy_engine=strategy)
    fake_strategy_obj = SimpleNamespace(selected_strategy={'bigBlind': 0.02, 'smallBlind': 0.01})
    decision.make_decision(fake_table, None, fake_strategy_obj, None)
    print(f"Decision: {decision.decision}")
    print(f"  finalCallLimit: {decision.finalCallLimit}")
    print(f"  finalBetLimit: {decision.finalBetLimit}")
