from __future__ import annotations

from typing import Protocol

from ..engine.action import Action, ActionType
from ..engine.game_state import GameState


class PokerStrategy(Protocol):
    """Any poker strategy must implement choose_action."""

    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action:
        ...

    def notify_result(self, state: GameState, payoff: int) -> None:
        ...


class BaseStrategy:
    """Default no-op implementations for optional methods."""

    def notify_result(self, state: GameState, payoff: int) -> None:
        pass
