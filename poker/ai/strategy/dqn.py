"""Pure-numpy DQN agent for poker. Adapted from trading bot DQN pattern."""

from __future__ import annotations

import os
import random
from collections import deque, namedtuple

import numpy as np

from ..engine.action import Action, ActionType
from ..engine.card import Card, RANK_SYMBOLS
from ..engine.game_state import GameState, Street
from .base import BaseStrategy
from .equity import monte_carlo_equity


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


# Action space: discretized for poker
DQN_ACTIONS = [
    ActionType.FOLD,
    ActionType.CHECK,
    ActionType.CALL,
    ActionType.BET,        # bet/raise 0.5x pot
    ActionType.RAISE,      # raise 1x pot (only when facing a bet)
    ActionType.ALL_IN,
]
NUM_ACTIONS = len(DQN_ACTIONS)
STATE_DIM = 64


class NeuralNetwork:
    """Two-hidden-layer MLP with ReLU, Adam optimizer. Pure numpy."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim // 2) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim // 2)
        self.W3 = np.random.randn(hidden_dim // 2, output_dim) * np.sqrt(2.0 / (hidden_dim // 2))
        self.b3 = np.zeros(output_dim)

        # Adam state
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self._init_adam()

    def _init_adam(self) -> None:
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        self.m_W3 = np.zeros_like(self.W3)
        self.v_W3 = np.zeros_like(self.W3)
        self.m_b3 = np.zeros_like(self.b3)
        self.v_b3 = np.zeros_like(self.b3)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = np.maximum(0, self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = np.maximum(0, self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3
        return self._z3

    def backward(self, target: np.ndarray, actions: np.ndarray, lr: float) -> float:
        batch_size = self._x.shape[0]
        q_values = self._z3
        action_q = q_values[np.arange(batch_size), actions]
        td_error = action_q - target
        loss = float(np.mean(td_error ** 2))

        dz3 = np.zeros_like(q_values)
        dz3[np.arange(batch_size), actions] = 2 * td_error / batch_size

        dW3 = self._a2.T @ dz3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (self._z2 > 0)
        dW2 = self._a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self._z1 > 0)
        dW1 = self._x.T @ dz1
        db1 = dz1.sum(axis=0)

        # Gradient clipping
        for g in (dW1, db1, dW2, db2, dW3, db3):
            np.clip(g, -1.0, 1.0, out=g)

        self.t += 1
        self._adam_update('W1', dW1, lr)
        self._adam_update('b1', db1, lr)
        self._adam_update('W2', dW2, lr)
        self._adam_update('b2', db2, lr)
        self._adam_update('W3', dW3, lr)
        self._adam_update('b3', db3, lr)

        return loss

    def _adam_update(self, name: str, grad: np.ndarray, lr: float) -> None:
        m = getattr(self, f'm_{name}')
        v = getattr(self, f'v_{name}')
        param = getattr(self, name)

        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def copy_weights_from(self, other: NeuralNetwork) -> None:
        for attr in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            getattr(self, attr)[:] = getattr(other, attr)

    def save(self, path: str) -> None:
        np.savez(
            path,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3,
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            getattr(self, key)[:] = data[key]


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append(Transition(
            state.copy(), action, reward,
            next_state.copy() if next_state is not None else np.zeros_like(state),
            done,
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


def encode_state(state: GameState) -> np.ndarray:
    """Encode GameState into 64-dim feature vector. All values clipped to [-3, 3]."""
    feat = np.zeros(STATE_DIM, dtype=np.float32)
    p = state.current_player
    if p < 0:
        return feat

    hole = state.hole_cards[p] if p < len(state.hole_cards) and state.hole_cards[p] else []

    # Hand strength (4 dims: rank1, rank2, suited, equity)
    if len(hole) == 2:
        c1, c2 = Card.from_int(hole[0]), Card.from_int(hole[1])
        feat[0] = (max(c1.rank, c2.rank) - 8) / 6  # high card normalized [-1, 1]
        feat[1] = (min(c1.rank, c2.rank) - 8) / 6
        feat[2] = 1.0 if c1.suit == c2.suit else 0.0
        if state.street != Street.PREFLOP and len(state.board) >= 3:
            num_opp = max(1, len(state.players_in_hand) - 1)
            feat[3] = monte_carlo_equity(hole, state.board, num_opp, num_simulations=100) * 2 - 1
        elif c1.rank == c2.rank:
            feat[3] = (c1.rank - 8) / 6  # pair strength

    # Position one-hot (6 dims)
    if state.num_players >= 2:
        rel_pos = (p - state.button) % state.num_players
        if rel_pos < 6:
            feat[4 + rel_pos] = 1.0

    # Stack-to-pot ratio, pot odds (2 dims)
    stack = state.stacks[p] if p < len(state.stacks) else 0
    pot = max(state.pot, 1)
    feat[10] = np.clip(stack / pot, 0, 10) / 5 - 1
    max_bet = max(state.current_bets) if state.current_bets else 0
    my_bet = state.current_bets[p] if state.current_bets else 0
    to_call = max_bet - my_bet
    if to_call > 0:
        feat[11] = to_call / (pot + to_call) * 2 - 1
    else:
        feat[11] = -1.0

    # Street one-hot (4 dims)
    feat[12 + int(state.street)] = 1.0

    # Board texture (8 dims)
    if len(state.board) >= 3:
        suits = [Card.from_int(c).suit for c in state.board]
        ranks = [Card.from_int(c).rank for c in state.board]
        feat[16] = 1.0 if len(set(suits)) <= 2 else 0.0  # flush draw possible
        feat[17] = 1.0 if max(suits.count(s) for s in set(suits)) >= 3 else 0.0  # 3+ same suit
        sorted_ranks = sorted(set(ranks))
        feat[18] = 1.0 if any(sorted_ranks[i + 1] - sorted_ranks[i] <= 2 for i in range(len(sorted_ranks) - 1)) else 0.0
        feat[19] = 1.0 if len(set(ranks)) < len(ranks) else 0.0  # paired
        feat[20] = (max(ranks) - 8) / 6 if ranks else 0.0
        feat[21] = (min(ranks) - 8) / 6 if ranks else 0.0
        feat[22] = len(state.board) / 5
        feat[23] = (max(ranks) - min(ranks)) / 12 if len(ranks) > 1 else 0.0

    # Action history summary (16 dims): last 8 actions, type + amount
    recent = state.action_history[-8:]
    for i, act in enumerate(recent):
        feat[24 + i * 2] = list(ActionType).index(act.type) / len(ActionType)
        feat[25 + i * 2] = np.clip(act.amount / pot, 0, 5) / 2.5 - 1

    # Stack sizes (8 dims), normalized by big blind
    bb = max(state.big_blind, 1)
    for i in range(min(state.num_players, 8)):
        feat[40 + i] = np.clip(state.stacks[i] / bb, 0, 100) / 50 - 1

    # Bets summary (16 dims)
    for i in range(min(state.num_players, 8)):
        if i < len(state.current_bets):
            feat[48 + i] = np.clip(state.current_bets[i] / bb, 0, 50) / 25 - 1
        feat[56 + i] = 1.0 if i < len(state.folded) and state.folded[i] else 0.0

    return np.clip(feat, -3, 3)


class DQNAgent(BaseStrategy):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 64,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        replay_buffer_size: int = 50_000,
        batch_size: int = 32,
        target_update_freq: int = 500,
        training: bool = True,
    ) -> None:
        self.state_dim = state_dim
        self.q_network = NeuralNetwork(state_dim, hidden_dim, NUM_ACTIONS)
        self.target_network = NeuralNetwork(state_dim, hidden_dim, NUM_ACTIONS)
        self.target_network.copy_weights_from(self.q_network)

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.lr = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training = training

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.total_steps = 0
        self.epsilon = epsilon_start

        self._last_state: np.ndarray | None = None
        self._last_action: int | None = None

    def _epsilon(self) -> float:
        progress = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action:
        encoded = encode_state(state)
        legal_indices = [i for i, a in enumerate(DQN_ACTIONS) if a in legal_actions]

        if not legal_indices:
            return Action(ActionType.FOLD, 0, state.current_player)

        if self.training and random.random() < self._epsilon():
            action_idx = random.choice(legal_indices)
        else:
            q_values = self.q_network.forward(encoded)[0]
            masked_q = np.full(NUM_ACTIONS, -np.inf)
            for i in legal_indices:
                masked_q[i] = q_values[i]
            action_idx = int(np.argmax(masked_q))

        if self.training:
            self._last_state = encoded
            self._last_action = action_idx
            self.total_steps += 1

        return self._make_action(state, DQN_ACTIONS[action_idx])

    def _make_action(self, state: GameState, action_type: ActionType) -> Action:
        p = state.current_player
        max_bet = max(state.current_bets) if state.current_bets else 0
        my_bet = state.current_bets[p] if state.current_bets else 0
        to_call = max_bet - my_bet
        pot = state.pot

        if action_type == ActionType.BET:
            amount = max(int(pot * 0.5), state.big_blind)
        elif action_type == ActionType.RAISE:
            amount = max(int(pot * 1.0), state.big_blind * 2)
        elif action_type == ActionType.CALL:
            amount = to_call
        elif action_type == ActionType.ALL_IN:
            amount = state.stacks[p]
        else:
            amount = 0

        return Action(action_type, amount, p)

    def store_transition(self, reward: float, next_state: GameState | None, done: bool) -> None:
        if self._last_state is None or self._last_action is None:
            return
        next_encoded = encode_state(next_state) if next_state is not None else None
        self.replay_buffer.add(self._last_state, self._last_action, reward, next_encoded, done)

    def notify_result(self, state: GameState, payoff: int) -> None:
        if not self.training or self._last_state is None:
            return
        bb = max(state.big_blind, 1)
        normalized_reward = payoff / bb
        self.store_transition(normalized_reward, None, True)
        self._last_state = None
        self._last_action = None

        if len(self.replay_buffer) >= self.batch_size:
            self.train_on_batch()

    def train_on_batch(self) -> float:
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Double DQN: main net picks best action, target net evaluates it
        next_q_main = self.q_network.forward(next_states)
        best_next = np.argmax(next_q_main, axis=1)
        next_q_target = self.target_network.forward(next_states)[np.arange(self.batch_size), best_next]
        target = rewards + self.gamma * next_q_target * (1 - dones)

        self.q_network.forward(states)
        loss = self.q_network.backward(target, actions, self.lr)

        if self.total_steps % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)

        return loss

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.q_network.save(path)

    def load(self, path: str) -> None:
        self.q_network.load(path)
        self.target_network.copy_weights_from(self.q_network)
