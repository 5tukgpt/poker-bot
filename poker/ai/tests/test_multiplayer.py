"""Tests for multi-player (3+) game support."""

from __future__ import annotations

import signal
from contextlib import contextmanager

from poker.ai.engine.action import Action, ActionType
from poker.ai.engine.table import Table
from poker.ai.sim.arena import Arena
from poker.ai.strategy.heuristic import HeuristicStrategy


@contextmanager
def time_limit(seconds: int):
    def handler(s, f):
        raise TimeoutError(f"timed out after {seconds}s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class TestThreePlayer:
    def test_completes(self):
        strategies = [HeuristicStrategy(num_simulations=20) for _ in range(3)]
        arena = Arena(strategies, names=['A', 'B', 'C'])
        with time_limit(20):
            stats = arena.play(10, verbose=False)
        assert sum(s.total_profit for s in stats) == 0  # chips conserved

    def test_button_rotates(self):
        strategies = [HeuristicStrategy(num_simulations=20) for _ in range(3)]
        table = Table(strategies, starting_stack=200)
        for i in range(6):
            initial_button = table.button
            with time_limit(10):
                table.play_hand()
            # Button should rotate by 1
            assert table.button == (initial_button + 1) % 3


class TestSixPlayer:
    def test_completes_quickly(self):
        strategies = [HeuristicStrategy(num_simulations=20) for _ in range(6)]
        arena = Arena(strategies, names=[f'P{i}' for i in range(6)])
        with time_limit(30):
            stats = arena.play(20, verbose=False)
        assert sum(s.total_profit for s in stats) == 0


class TestAllFold:
    def test_three_player_all_fold_to_button(self):
        class AlwaysFold:
            def choose_action(self, state, legal):
                return Action(ActionType.FOLD, 0, state.current_player)
            def notify_result(self, state, payoff):
                pass

        class AlwaysCheck:
            def choose_action(self, state, legal):
                if ActionType.CHECK in legal:
                    return Action(ActionType.CHECK, 0, state.current_player)
                return Action(ActionType.FOLD, 0, state.current_player)
            def notify_result(self, state, payoff):
                pass

        # 3 always-fold players except one — pot should go to the holdout
        # Use AlwaysCheck for player 0 (button) so they don't fold
        strategies = [AlwaysCheck(), AlwaysFold(), AlwaysFold()]
        table = Table(strategies, starting_stack=200)
        with time_limit(10):
            deltas = table.play_hand()
        # Sum is 0, button (player 0) wins blinds
        assert sum(deltas) == 0
        # Some player should have positive delta
        assert max(deltas) > 0
