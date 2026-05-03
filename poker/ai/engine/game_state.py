from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .action import Action, ActionType


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


@dataclass
class GameState:
    num_players: int
    stacks: list[int]
    pot: int
    board: list[int]
    hole_cards: list[list[int]]
    street: Street
    current_player: int
    button: int
    small_blind: int
    big_blind: int
    current_bets: list[int] = field(default_factory=list)
    action_history: list[Action] = field(default_factory=list)
    folded: list[bool] = field(default_factory=list)
    all_in: list[bool] = field(default_factory=list)

    def legal_actions(self) -> list[ActionType]:
        player = self.current_player
        stack = self.stacks[player]
        if stack == 0:
            return []

        max_bet = max(self.current_bets) if self.current_bets else 0
        my_bet = self.current_bets[player] if self.current_bets else 0
        to_call = max_bet - my_bet

        actions: list[ActionType] = [ActionType.FOLD]

        if to_call == 0:
            actions.append(ActionType.CHECK)
        else:
            actions.append(ActionType.CALL)

        if to_call == 0:
            if stack > 0:
                actions.append(ActionType.BET)
        else:
            min_raise = max_bet + self.big_blind
            if stack + my_bet > min_raise:
                actions.append(ActionType.RAISE)

        actions.append(ActionType.ALL_IN)
        return actions

    @property
    def active_players(self) -> list[int]:
        return [i for i in range(self.num_players)
                if not self.folded[i] and not self.all_in[i]]

    @property
    def players_in_hand(self) -> list[int]:
        return [i for i in range(self.num_players) if not self.folded[i]]
