"""Tests for BookStrategy."""

from __future__ import annotations

from poker.ai.engine.action import Action, ActionType
from poker.ai.engine.card import Card
from poker.ai.engine.game_state import GameState, Street
from poker.ai.strategy.book_strategy import (
    BookStrategy,
    board_texture,
    categorize_made_hand,
    has_strong_draw,
    cbet_size,
    value_bet_size,
)


class TestBoardTexture:
    def test_dry_board(self):
        # K72 rainbow — classic dry board
        board = [Card.from_str('Kc').to_int(),
                 Card.from_str('7d').to_int(),
                 Card.from_str('2h').to_int()]
        t = board_texture(board)
        assert t['dry'] == True
        assert t['wet'] == False
        assert t['monotone'] == False

    def test_wet_board(self):
        # 9h 8h 7c — straight + flush draw
        board = [Card.from_str('9h').to_int(),
                 Card.from_str('8h').to_int(),
                 Card.from_str('7c').to_int()]
        t = board_texture(board)
        assert t['wet'] == True
        assert t['flush_draw'] == True
        assert t['straight_draw'] == True

    def test_paired_board(self):
        board = [Card.from_str('Kc').to_int(),
                 Card.from_str('Kd').to_int(),
                 Card.from_str('7h').to_int()]
        t = board_texture(board)
        assert t['paired'] == True

    def test_monotone(self):
        board = [Card.from_str('As').to_int(),
                 Card.from_str('Ks').to_int(),
                 Card.from_str('7s').to_int()]
        t = board_texture(board)
        assert t['monotone'] == True


class TestHandCategory:
    def test_monster_set(self):
        # Pocket aces, board has another A — set of aces
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()]
        board = [Card.from_str('As').to_int(),
                 Card.from_str('7d').to_int(),
                 Card.from_str('2h').to_int()]
        cat = categorize_made_hand(hole, board)
        assert cat in ('monster', 'strong')

    def test_air(self):
        # 7-2 on a K-Q-J board
        hole = [Card.from_str('7c').to_int(), Card.from_str('2d').to_int()]
        board = [Card.from_str('Kh').to_int(),
                 Card.from_str('Qh').to_int(),
                 Card.from_str('Jh').to_int()]
        cat = categorize_made_hand(hole, board)
        assert cat == 'air'

    def test_strong_top_pair(self):
        # AK on K72 — top pair top kicker
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Kd').to_int()]
        board = [Card.from_str('Kh').to_int(),
                 Card.from_str('7s').to_int(),
                 Card.from_str('2c').to_int()]
        cat = categorize_made_hand(hole, board)
        assert cat in ('strong', 'monster')


class TestDraws:
    def test_flush_draw(self):
        # 4 hearts after flop+turn = flush draw
        hole = [Card.from_str('Ah').to_int(), Card.from_str('Kh').to_int()]
        board = [Card.from_str('5h').to_int(),
                 Card.from_str('9h').to_int(),
                 Card.from_str('2c').to_int()]
        assert has_strong_draw(hole, board) == True

    def test_no_draw(self):
        # Random hand, dry board
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Kd').to_int()]
        board = [Card.from_str('7h').to_int(),
                 Card.from_str('2s').to_int(),
                 Card.from_str('4d').to_int()]
        assert has_strong_draw(hole, board) == False


class TestBetSizing:
    def test_cbet_dry_smaller(self):
        dry = {'dry': True, 'wet': False, 'paired': False, 'flush_draw': False,
               'monotone': False, 'straight_draw': False, 'highcard': 14}
        wet = {'dry': False, 'wet': True, 'paired': False, 'flush_draw': True,
               'monotone': False, 'straight_draw': True, 'highcard': 9}
        size_dry = cbet_size(100, 2, dry, is_value=False)
        size_wet = cbet_size(100, 2, wet, is_value=False)
        assert size_dry < size_wet  # smaller on dry


class TestStrategy:
    def _state(self, hole, **overrides):
        defaults = dict(
            num_players=2, stacks=[200, 198], pot=3,
            board=[], hole_cards=[hole, [50, 51]],
            street=Street.PREFLOP, current_player=0, button=0,
            small_blind=1, big_blind=2,
            current_bets=[1, 2], action_history=[],
            folded=[False, False], all_in=[False, False],
        )
        defaults.update(overrides)
        return GameState(**defaults)

    def test_aa_opens(self):
        strat = BookStrategy(postflop_sims=20)
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()]
        s = self._state(hole)
        action = strat.choose_action(s, s.legal_actions())
        assert action.type in (ActionType.RAISE, ActionType.BET, ActionType.CALL)

    def test_trash_folds_or_checks(self):
        strat = BookStrategy(postflop_sims=20)
        hole = [Card.from_str('7c').to_int(), Card.from_str('2d').to_int()]
        s = self._state(hole)
        action = strat.choose_action(s, s.legal_actions())
        assert action.type in (ActionType.FOLD, ActionType.CHECK)

    def test_postflop_works(self):
        strat = BookStrategy(postflop_sims=20)
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Kd').to_int()]
        board = [Card.from_str('Kh').to_int(),
                 Card.from_str('7s').to_int(),
                 Card.from_str('2c').to_int()]
        s = self._state(hole, board=board, street=Street.FLOP, pot=10,
                        current_bets=[0, 0])
        action = strat.choose_action(s, s.legal_actions())
        assert isinstance(action, Action)
        assert action.type in s.legal_actions()
