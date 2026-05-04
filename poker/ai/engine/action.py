from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    FOLD = 'fold'
    CHECK = 'check'
    CALL = 'call'
    BET = 'bet'
    RAISE = 'raise'
    ALL_IN = 'all_in'


@dataclass(frozen=True)
class Action:
    type: ActionType
    amount: int = 0
    player_idx: int = 0
    # Optional street index (0=preflop, 1=flop, 2=turn, 3=river).
    # Used for accurate stat tracking when actions are observed out-of-band.
    # If None, observers fall back to using state.street.
    street: int | None = None

    def __str__(self) -> str:
        if self.type in (ActionType.FOLD, ActionType.CHECK):
            return self.type.value
        return f"{self.type.value} {self.amount}"
