"""Tests for the poker AI engine."""

import pytest
from poker.ai.engine.card import Card, Deck, Rank, Suit
from poker.ai.engine.action import Action, ActionType
from poker.ai.engine.evaluator import evaluate_hand, determine_winners
from poker.ai.engine.game_state import GameState, Street
from poker.ai.engine.table import Table
from poker.ai.strategy.heuristic import HeuristicStrategy


class TestCard:
    def test_deck_has_52_cards(self):
        deck = Deck()
        cards = deck.deal(52)
        assert len(cards) == 52
        assert len(set(cards)) == 52

    def test_int_roundtrip(self):
        for i in range(52):
            card = Card.from_int(i)
            assert card.to_int() == i

    def test_str_roundtrip(self):
        for i in range(52):
            card = Card.from_int(i)
            s = str(card)
            assert Card.from_str(s) == card

    def test_shuffle_changes_order(self):
        d1 = Deck()
        d2 = Deck()
        d2.shuffle()
        assert d1.deal(52) != d2.deal(52)

    def test_remaining(self):
        deck = Deck()
        assert deck.remaining == 52
        deck.deal(5)
        assert deck.remaining == 47


class TestEvaluator:
    def test_royal_flush_beats_pair(self):
        # Royal flush: As Ks Qs Js Ts
        royal = [Card.from_str('As').to_int(), Card.from_str('Ks').to_int()]
        board_royal = [
            Card.from_str('Qs').to_int(),
            Card.from_str('Js').to_int(),
            Card.from_str('Ts').to_int(),
        ]
        # Pair of twos
        pair = [Card.from_str('2h').to_int(), Card.from_str('2d').to_int()]
        board_pair = [
            Card.from_str('7c').to_int(),
            Card.from_str('8s').to_int(),
            Card.from_str('9h').to_int(),
        ]
        assert evaluate_hand(royal, board_royal) < evaluate_hand(pair, board_pair)

    def test_full_house_beats_flush(self):
        fh = [Card.from_str('Ah').to_int(), Card.from_str('Ad').to_int()]
        board = [
            Card.from_str('As').to_int(),
            Card.from_str('Kh').to_int(),
            Card.from_str('Kd').to_int(),
        ]
        flush = [Card.from_str('2h').to_int(), Card.from_str('3h').to_int()]
        board_f = [
            Card.from_str('5h').to_int(),
            Card.from_str('7h').to_int(),
            Card.from_str('9h').to_int(),
        ]
        assert evaluate_hand(fh, board) < evaluate_hand(flush, board_f)

    def test_determine_winners_tie(self):
        board = [
            Card.from_str('As').to_int(),
            Card.from_str('Ks').to_int(),
            Card.from_str('Qs').to_int(),
            Card.from_str('Js').to_int(),
            Card.from_str('Ts').to_int(),
        ]
        h1 = [Card.from_str('2h').to_int(), Card.from_str('3h').to_int()]
        h2 = [Card.from_str('4d').to_int(), Card.from_str('5d').to_int()]
        winners = determine_winners([h1, h2], board)
        assert winners == [0, 1]

    def test_determine_winners_single(self):
        board = [
            Card.from_str('2s').to_int(),
            Card.from_str('7d').to_int(),
            Card.from_str('9c').to_int(),
            Card.from_str('Jh').to_int(),
            Card.from_str('3s').to_int(),
        ]
        h1 = [Card.from_str('Ah').to_int(), Card.from_str('Kh').to_int()]
        h2 = [Card.from_str('4d').to_int(), Card.from_str('5d').to_int()]
        winners = determine_winners([h1, h2], board)
        assert winners == [0]


class TestGameState:
    def _make_state(self, **kwargs):
        defaults = dict(
            num_players=2, stacks=[200, 200], pot=3,
            board=[], hole_cards=[[0, 1], [2, 3]],
            street=Street.PREFLOP, current_player=0, button=0,
            small_blind=1, big_blind=2,
            current_bets=[1, 2], folded=[False, False],
            all_in=[False, False],
        )
        defaults.update(kwargs)
        return GameState(**defaults)

    def test_legal_actions_facing_bet(self):
        state = self._make_state(current_player=0)
        legal = state.legal_actions()
        assert ActionType.FOLD in legal
        assert ActionType.CALL in legal
        assert ActionType.CHECK not in legal

    def test_legal_actions_no_bet(self):
        state = self._make_state(current_bets=[0, 0])
        legal = state.legal_actions()
        assert ActionType.CHECK in legal
        assert ActionType.BET in legal
        assert ActionType.CALL not in legal

    def test_active_players(self):
        state = self._make_state(folded=[True, False])
        assert state.active_players == [1]


class TestTable:
    def test_complete_hand_runs(self):
        s1 = HeuristicStrategy(num_simulations=50)
        s2 = HeuristicStrategy(num_simulations=50)
        table = Table([s1, s2], starting_stack=200)
        deltas = table.play_hand()
        assert len(deltas) == 2
        assert sum(deltas) == 0

    def test_chip_conservation(self):
        s1 = HeuristicStrategy(num_simulations=50)
        s2 = HeuristicStrategy(num_simulations=50)
        table = Table([s1, s2], starting_stack=200)
        for _ in range(20):
            deltas = table.play_hand()
            assert sum(deltas) == 0
        assert sum(table.stacks) == 400

    def test_all_fold_awards_pot(self):
        class AlwaysFold:
            def choose_action(self, state, legal):
                return Action(ActionType.FOLD, 0, state.current_player)
            def notify_result(self, state, payoff):
                pass

        class AlwaysCall:
            def choose_action(self, state, legal):
                if ActionType.CALL in legal:
                    return Action(ActionType.CALL, 0, state.current_player)
                if ActionType.CHECK in legal:
                    return Action(ActionType.CHECK, 0, state.current_player)
                return Action(ActionType.FOLD, 0, state.current_player)
            def notify_result(self, state, payoff):
                pass

        table = Table([AlwaysFold(), AlwaysCall()], starting_stack=200)
        deltas = table.play_hand()
        assert sum(deltas) == 0
        assert deltas[1] > 0  # caller should win the blinds
