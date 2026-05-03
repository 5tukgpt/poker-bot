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

    def __str__(self) -> str:
        if self.type in (ActionType.FOLD, ActionType.CHECK):
            return self.type.value
        return f"{self.type.value} {self.amount}"
