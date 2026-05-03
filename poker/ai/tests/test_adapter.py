"""Tests for the dickreuter → GameState adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from poker.ai.adapter import (
    action_to_dickreuter,
    card_str_to_int,
    cards_to_ints,
    gamestage_to_street,
    to_chips,
    to_game_state,
)
from poker.ai.engine.action import Action, ActionType
from poker.ai.engine.card import Card
from poker.ai.engine.game_state import Street


class TestCardConversion:
    def test_card_string_to_int(self):
        assert card_str_to_int('As') == Card.from_str('As').to_int()
        assert card_str_to_int('2c') == Card.from_str('2c').to_int()

    def test_cards_list(self):
        result = cards_to_ints(['As', 'Kh'])
        assert len(result) == 2
        assert result[0] == Card.from_str('As').to_int()

    def test_empty_cards(self):
        assert cards_to_ints([]) == []
        assert cards_to_ints(['', None, 'As']) == [Card.from_str('As').to_int()]


class TestStageMapping:
    def test_all_stages(self):
        assert gamestage_to_street('PreFlop') == Street.PREFLOP
        assert gamestage_to_street('Flop') == Street.FLOP
        assert gamestage_to_street('Turn') == Street.TURN
        assert gamestage_to_street('River') == Street.RIVER

    def test_unknown_defaults_to_preflop(self):
        assert gamestage_to_street('Unknown') == Street.PREFLOP


class TestChipConversion:
    def test_basic_conversion(self):
        # 1 BB = $0.02 default, so $0.02 = 2 chips
        assert to_chips(0.02, bb_value=0.02) == 2
        assert to_chips(0.10, bb_value=0.02) == 10  # 5 BB

    def test_handles_strings(self):
        assert to_chips('0.04', bb_value=0.02) == 4

    def test_handles_none_and_empty(self):
        assert to_chips(None) == 0
        assert to_chips('') == 0
        assert to_chips('not a number') == 0


class TestToGameState:
    def _mock_table(self, **overrides):
        defaults = {
            'mycards': ['As', 'Kh'],
            'cardsOnTable': [],
            'gameStage': 'PreFlop',
            'totalPotValue': 0.06,  # 3 BB
            'round_pot_value': 0.06,
            'bigBlind': 0.02,
            'smallBlind': 0.01,
            'minCall': 0.02,
            'currentCallValue': 0.02,
            'currentBetValue': 0.02,
            'dealer_position': 0,
            'myFunds': 2.0,  # 100 BB
            'other_players': [
                {'funds': 2.0, 'pot': 0.02, 'status': 1, 'name': 'Villain'},
            ],
            'checkButton': False,
            'allInCallButton': False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_basic_translation(self):
        table = self._mock_table()
        state = to_game_state(table)

        assert state.num_players == 2
        assert state.street == Street.PREFLOP
        assert len(state.hole_cards[0]) == 2
        assert state.hole_cards[0][0] == Card.from_str('As').to_int()
        assert state.big_blind == 2  # $0.02 → 2 chips
        assert state.small_blind == 1

    def test_postflop_with_board(self):
        table = self._mock_table(
            gameStage='Flop',
            cardsOnTable=['Qh', 'Js', '7d'],
        )
        state = to_game_state(table)
        assert state.street == Street.FLOP
        assert len(state.board) == 3

    def test_player_stacks(self):
        table = self._mock_table(
            myFunds=4.0,  # 200 BB
            other_players=[{'funds': 2.0, 'pot': 0.02, 'status': 1}],
        )
        state = to_game_state(table)
        assert state.stacks[0] == 400  # 200 BB * 2 chips/BB
        assert state.stacks[1] == 200  # 100 BB

    def test_folded_player(self):
        table = self._mock_table(
            other_players=[{'funds': 2.0, 'pot': 0, 'status': 0}],  # status 0 = folded
        )
        state = to_game_state(table)
        assert state.folded[1] == True

    def test_legal_actions_computed(self):
        table = self._mock_table()
        state = to_game_state(table)
        legal = state.legal_actions()
        assert ActionType.FOLD in legal


class TestActionToDickreuter:
    def test_all_actions_map(self):
        for action_type in ActionType:
            result = action_to_dickreuter(Action(action_type, 0, 0))
            assert isinstance(result, str)
            assert len(result) > 0

    def test_specific_mappings(self):
        assert action_to_dickreuter(Action(ActionType.FOLD, 0, 0)) == 'Fold'
        assert action_to_dickreuter(Action(ActionType.CALL, 5, 0)) == 'Call'
        assert action_to_dickreuter(Action(ActionType.CHECK, 0, 0)) == 'Check'
