"""Smoke tests for strategy modules."""

import os
import tempfile

import numpy as np

from poker.ai.engine.action import Action, ActionType
from poker.ai.engine.game_state import GameState, Street
from poker.ai.strategy.cfr import CFRTrainer, CFRStrategy, hand_to_bucket
from poker.ai.strategy.dqn import DQNAgent, encode_state, NUM_ACTIONS
from poker.ai.strategy.equity import monte_carlo_equity
from poker.ai.strategy.heuristic import HeuristicStrategy


def _state(hole, board=None, current_bets=None, street=Street.PREFLOP):
    return GameState(
        num_players=2,
        stacks=[200, 200],
        pot=3,
        board=board or [],
        hole_cards=[hole, [50, 51]],
        street=street,
        current_player=0,
        button=0,
        small_blind=1,
        big_blind=2,
        current_bets=current_bets or [1, 2],
        folded=[False, False],
        all_in=[False, False],
    )


class TestEquity:
    def test_aces_high_equity(self):
        # AA vs random hand should have ~85% equity
        hole = [48, 49]  # AcAd in our int encoding (rank 14 = A, suits 0,1)
        eq = monte_carlo_equity(hole, [], num_opponents=1, num_simulations=200)
        assert eq > 0.7

    def test_trash_low_equity(self):
        # 72o is the worst hand
        hole = [20, 0]  # 7c2c
        eq = monte_carlo_equity(hole, [], num_opponents=1, num_simulations=200)
        assert eq < 0.45


class TestHeuristic:
    def test_premium_hand_raises(self):
        # AA preflop should raise
        strat = HeuristicStrategy(num_simulations=50)
        s = _state([48, 49])  # AA
        action = strat.choose_action(s, s.legal_actions())
        assert action.type in (ActionType.RAISE, ActionType.BET, ActionType.CALL)

    def test_trash_folds_to_raise(self):
        # 72o facing big raise should fold
        strat = HeuristicStrategy(num_simulations=50)
        s = _state([20, 0], current_bets=[1, 20])  # facing 20 chip raise
        action = strat.choose_action(s, s.legal_actions())
        assert action.type == ActionType.FOLD


class TestCFR:
    def test_hand_to_bucket_premium(self):
        # AA = bucket 4
        assert hand_to_bucket([48, 49], []) == 4

    def test_hand_to_bucket_trash(self):
        # 72o = bucket 0
        assert hand_to_bucket([20, 0], []) == 0

    def test_trainer_runs(self):
        trainer = CFRTrainer()
        trainer.train(10, verbose=False)
        assert len(trainer.regret_sum) > 0

    def test_strategy_plays(self):
        trainer = CFRTrainer()
        trainer.train(10, verbose=False)
        strat = CFRStrategy(trainer=trainer)
        s = _state([48, 49])
        action = strat.choose_action(s, s.legal_actions())
        assert isinstance(action, Action)
        assert action.type in s.legal_actions()

    def test_save_load(self):
        trainer = CFRTrainer()
        trainer.train(10, verbose=False)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            trainer.save(path)
            loaded = CFRStrategy(strategy_path=path)
            assert len(loaded.strategy_table) > 0
        finally:
            os.unlink(path)


class TestDQN:
    def test_encode_state_shape(self):
        s = _state([48, 49])
        encoded = encode_state(s)
        assert encoded.shape == (64,)
        assert np.all(encoded >= -3) and np.all(encoded <= 3)

    def test_agent_chooses_legal(self):
        agent = DQNAgent(training=False)
        s = _state([48, 49])
        legal = s.legal_actions()
        action = agent.choose_action(s, legal)
        assert action.type in legal

    def test_replay_buffer_grows(self):
        agent = DQNAgent(training=True)
        s = _state([48, 49])
        agent.choose_action(s, s.legal_actions())
        agent.notify_result(s, 5)
        assert len(agent.replay_buffer) == 1

    def test_train_on_batch(self):
        agent = DQNAgent(training=True, batch_size=2)
        s = _state([48, 49])
        for _ in range(5):
            agent.choose_action(s, s.legal_actions())
            agent.notify_result(s, 1)
        loss = agent.train_on_batch()
        assert isinstance(loss, float)

    def test_save_load(self):
        agent = DQNAgent(training=False)
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            agent2 = DQNAgent(training=False)
            agent2.load(path)
            s = _state([48, 49])
            encoded = encode_state(s)
            q1 = agent.q_network.forward(encoded)
            q2 = agent2.q_network.forward(encoded)
            assert np.allclose(q1, q2)
        finally:
            os.unlink(path)
