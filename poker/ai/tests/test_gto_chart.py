"""Tests for GTO chart strategy."""

from __future__ import annotations

from poker.ai.engine.action import Action, ActionType
from poker.ai.engine.card import Card
from poker.ai.engine.game_state import GameState, Street
from poker.ai.strategy.gto_chart import (
    GTOChartStrategy,
    SB_OPEN_RAISE,
    BB_3BET,
    SB_4BET_JAM,
    hand_class,
    _expand,
)


class TestHandClass:
    def test_pair(self):
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()]
        assert hand_class(hole) == 'AA'

    def test_suited(self):
        hole = [Card.from_str('As').to_int(), Card.from_str('Ks').to_int()]
        assert hand_class(hole) == 'AKs'

    def test_offsuit(self):
        hole = [Card.from_str('As').to_int(), Card.from_str('Kc').to_int()]
        assert hand_class(hole) == 'AKo'


class TestRangeExpansion:
    def test_pair_range(self):
        r = _expand('22+')
        assert 'AA' in r and '22' in r and 'KK' in r
        assert len(r) == 13  # 22 through AA

    def test_suited_range(self):
        r = _expand('A2s+')
        assert 'AKs' in r and 'A2s' in r
        # A2s, A3s, ..., AKs (12 hands)
        assert len(r) == 12


class TestRanges:
    def test_premiums_in_open(self):
        for hand in ['AA', 'KK', 'AKs', 'AQo']:
            assert hand in SB_OPEN_RAISE, f"{hand} should be in SB_OPEN_RAISE"

    def test_premiums_in_3bet(self):
        for hand in ['AA', 'KK', 'AKs']:
            assert hand in BB_3BET, f"{hand} should be in BB_3BET"

    def test_premium_in_4bet_jam(self):
        for hand in ['AA', 'KK', 'QQ', 'AKs']:
            assert hand in SB_4BET_JAM


class TestStrategy:
    def _state(self, hole, **overrides):
        defaults = dict(
            num_players=2,
            stacks=[200, 198],
            pot=3,
            board=[],
            hole_cards=[hole, [50, 51]],
            street=Street.PREFLOP,
            current_player=0,
            button=0,
            small_blind=1,
            big_blind=2,
            current_bets=[1, 2],
            action_history=[],
            folded=[False, False],
            all_in=[False, False],
        )
        defaults.update(overrides)
        return GameState(**defaults)

    def test_aa_opens(self):
        strat = GTOChartStrategy(postflop_sims=20)
        # AA preflop, SB position
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()]
        s = self._state(hole)
        action = strat.choose_action(s, s.legal_actions())
        assert action.type in (ActionType.RAISE, ActionType.BET, ActionType.CALL)

    def test_trash_folds(self):
        strat = GTOChartStrategy(postflop_sims=20)
        # 72o preflop, SB position
        hole = [Card.from_str('7c').to_int(), Card.from_str('2d').to_int()]
        s = self._state(hole)
        action = strat.choose_action(s, s.legal_actions())
        # 72o is in SB_FOLD or not in SB_OPEN_RAISE
        assert action.type in (ActionType.FOLD, ActionType.CHECK)

    def test_aa_4bets_facing_3bet(self):
        strat = GTOChartStrategy(postflop_sims=20)
        # AA in SB facing a 3-bet from BB
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()]
        s = self._state(
            hole,
            action_history=[
                Action(ActionType.RAISE, 5, 0),  # We opened
                Action(ActionType.RAISE, 15, 1),  # BB 3-bet
            ],
            current_bets=[5, 15],
            stacks=[195, 185],
        )
        action = strat.choose_action(s, s.legal_actions())
        assert action.type in (ActionType.ALL_IN, ActionType.RAISE, ActionType.CALL)
