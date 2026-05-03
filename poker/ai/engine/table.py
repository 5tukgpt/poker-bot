from __future__ import annotations

from typing import TYPE_CHECKING

from .action import Action, ActionType
from .card import Deck
from .evaluator import determine_winners
from .game_state import GameState, Street

if TYPE_CHECKING:
    from ..strategy.base import PokerStrategy


class Table:
    def __init__(
        self,
        strategies: list[PokerStrategy],
        small_blind: int = 1,
        big_blind: int = 2,
        starting_stack: int = 200,
    ) -> None:
        self.strategies = strategies
        self.num_players = len(strategies)
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stack = starting_stack
        self.stacks = [starting_stack] * self.num_players
        self.button = 0
        self.deck = Deck()

    def play_hand(self) -> list[int]:
        """Play a single hand. Returns chip deltas per player."""
        self.deck.shuffle()
        start_stacks = self.stacks.copy()

        hole_cards = [self.deck.deal(2) for _ in range(self.num_players)]
        folded = [False] * self.num_players
        all_in = [False] * self.num_players
        current_bets = [0] * self.num_players
        pot = 0
        board: list[int] = []

        # Heads-up: button = SB. 3+ players: SB = button+1, BB = button+2.
        if self.num_players == 2:
            sb_idx = self.button
            bb_idx = (self.button + 1) % self.num_players
        else:
            sb_idx = (self.button + 1) % self.num_players
            bb_idx = (self.button + 2) % self.num_players

        sb_amount = min(self.small_blind, self.stacks[sb_idx])
        bb_amount = min(self.big_blind, self.stacks[bb_idx])
        self.stacks[sb_idx] -= sb_amount
        self.stacks[bb_idx] -= bb_amount
        current_bets[sb_idx] = sb_amount
        current_bets[bb_idx] = bb_amount
        pot += sb_amount + bb_amount

        if self.stacks[sb_idx] == 0:
            all_in[sb_idx] = True
        if self.stacks[bb_idx] == 0:
            all_in[bb_idx] = True

        action_history: list[Action] = []

        for street in Street:
            if street == Street.FLOP:
                board.extend(self.deck.deal(3))
            elif street == Street.TURN:
                board.extend(self.deck.deal(1))
            elif street == Street.RIVER:
                board.extend(self.deck.deal(1))

            current_bets = [0] * self.num_players
            players_in = [i for i in range(self.num_players) if not folded[i]]
            if len(players_in) <= 1:
                break

            active = [i for i in players_in if not all_in[i]]
            if len(active) <= 1 and all(
                current_bets[i] == current_bets[active[0]] if active else True
                for i in players_in if not all_in[i]
            ):
                if len([i for i in players_in if all_in[i]]) > 0 or len(active) == 0:
                    continue

            pot = self._betting_round(
                street, hole_cards, board, folded, all_in,
                current_bets, pot, action_history,
            )

            players_in = [i for i in range(self.num_players) if not folded[i]]
            if len(players_in) <= 1:
                break

        players_in = [i for i in range(self.num_players) if not folded[i]]

        if len(players_in) == 1:
            self.stacks[players_in[0]] += pot
        else:
            while len(board) < 5:
                board.extend(self.deck.deal(1))
            hands = [hole_cards[i] for i in players_in]
            winners = determine_winners(hands, board)
            winner_indices = [players_in[w] for w in winners]
            share = pot // len(winner_indices)
            remainder = pot % len(winner_indices)
            for w in winner_indices:
                self.stacks[w] += share
            if remainder > 0:
                self.stacks[winner_indices[0]] += remainder

        self.button = (self.button + 1) % self.num_players
        deltas = [self.stacks[i] - start_stacks[i] for i in range(self.num_players)]

        state = GameState(
            num_players=self.num_players,
            stacks=self.stacks.copy(),
            pot=0,
            board=board,
            hole_cards=hole_cards,
            street=Street.RIVER,
            current_player=-1,
            button=self.button,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )
        for i, strategy in enumerate(self.strategies):
            strategy.notify_result(state, deltas[i])

        return deltas

    def _betting_round(
        self,
        street: Street,
        hole_cards: list[list[int]],
        board: list[int],
        folded: list[bool],
        all_in: list[bool],
        current_bets: list[int],
        pot: int,
        action_history: list[Action],
    ) -> int:
        if street == Street.PREFLOP:
            if self.num_players == 2:
                first_to_act = self.button  # SB/button acts first preflop in HU
            else:
                first_to_act = (self.button + 3) % self.num_players
        else:
            if self.num_players == 2:
                first_to_act = (self.button + 1) % self.num_players  # BB acts first postflop
            else:
                first_to_act = (self.button + 1) % self.num_players

        last_raiser = -1
        acted = set()
        current_idx = first_to_act

        while True:
            if folded[current_idx] or all_in[current_idx]:
                current_idx = (current_idx + 1) % self.num_players
                if current_idx == first_to_act and len(acted) >= self._count_active(folded, all_in):
                    break
                continue

            state = GameState(
                num_players=self.num_players,
                stacks=self.stacks.copy(),
                pot=pot,
                board=board.copy(),
                hole_cards=hole_cards,
                street=street,
                current_player=current_idx,
                button=self.button,
                small_blind=self.small_blind,
                big_blind=self.big_blind,
                current_bets=current_bets.copy(),
                action_history=action_history.copy(),
                folded=folded.copy(),
                all_in=all_in.copy(),
            )

            legal = state.legal_actions()
            if not legal:
                current_idx = (current_idx + 1) % self.num_players
                continue

            action = self.strategies[current_idx].choose_action(state, legal)
            action = self._validate_action(action, state, current_idx)
            action_history.append(action)
            pot = self._apply_action(action, current_bets, folded, all_in, pot)
            acted.add(current_idx)

            if action.type in (ActionType.RAISE, ActionType.BET):
                last_raiser = current_idx
                acted = {current_idx}

            current_idx = (current_idx + 1) % self.num_players

            active_count = self._count_active(folded, all_in)
            if active_count <= 1:
                break

            all_acted = len(acted) >= active_count
            if all_acted and (last_raiser == -1 or current_idx == last_raiser):
                break
            if current_idx == first_to_act and all_acted and last_raiser == -1:
                break

        return pot

    def _validate_action(self, action: Action, state: GameState, player: int) -> Action:
        legal = state.legal_actions()
        if action.type not in legal:
            if ActionType.CHECK in legal:
                return Action(ActionType.CHECK, 0, player)
            return Action(ActionType.FOLD, 0, player)

        max_bet = max(state.current_bets) if state.current_bets else 0
        my_bet = state.current_bets[player] if state.current_bets else 0
        stack = self.stacks[player]

        if action.type == ActionType.CALL:
            to_call = min(max_bet - my_bet, stack)
            return Action(ActionType.CALL, to_call, player)
        elif action.type == ActionType.ALL_IN:
            return Action(ActionType.ALL_IN, stack, player)
        elif action.type in (ActionType.BET, ActionType.RAISE):
            amount = max(action.amount, self.big_blind)
            amount = min(amount, stack)
            return Action(action.type, amount, player)

        return Action(action.type, 0, player)

    def _apply_action(
        self,
        action: Action,
        current_bets: list[int],
        folded: list[bool],
        all_in: list[bool],
        pot: int,
    ) -> int:
        p = action.player_idx

        if action.type == ActionType.FOLD:
            folded[p] = True
        elif action.type == ActionType.CHECK:
            pass
        elif action.type == ActionType.CALL:
            self.stacks[p] -= action.amount
            current_bets[p] += action.amount
            pot += action.amount
            if self.stacks[p] == 0:
                all_in[p] = True
        elif action.type in (ActionType.BET, ActionType.RAISE):
            self.stacks[p] -= action.amount
            current_bets[p] += action.amount
            pot += action.amount
            if self.stacks[p] == 0:
                all_in[p] = True
        elif action.type == ActionType.ALL_IN:
            self.stacks[p] -= action.amount
            current_bets[p] += action.amount
            pot += action.amount
            all_in[p] = True

        return pot

    @staticmethod
    def _count_active(folded: list[bool], all_in: list[bool]) -> int:
        return sum(1 for f, a in zip(folded, all_in) if not f and not a)
