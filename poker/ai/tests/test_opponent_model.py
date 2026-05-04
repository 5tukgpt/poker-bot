"""Tests for opponent modeling."""

from __future__ import annotations

from poker.ai.engine.action import Action, ActionType
from poker.ai.engine.game_state import GameState, Street
from poker.ai.strategy.opponent_model import (
    OpponentStats,
    PlayerType,
    get_strategy_for_type,
)


def _state(player=1, street=Street.PREFLOP, history=None):
    return GameState(
        num_players=2, stacks=[200, 200], pot=3, board=[],
        hole_cards=[[0, 1], [2, 3]], street=street,
        current_player=player, button=0, small_blind=1, big_blind=2,
        current_bets=[1, 2], action_history=history or [],
        folded=[False, False], all_in=[False, False],
    )


class TestOpponentStats:
    def test_initial_state(self):
        stats = OpponentStats(name="test")
        assert stats.player_type == PlayerType.UNKNOWN
        assert stats.vpip == 0
        assert stats.pfr == 0
        assert stats.aggression_factor == 0

    def test_pfr_tracking(self):
        stats = OpponentStats(name="test")
        for _ in range(50):
            # Opponent raises preflop every hand
            state = _state(player=1)
            action = Action(ActionType.RAISE, 6, 1)
            stats.observe_action(state, action)
            stats.end_hand()
        # Should be 100% PFR
        assert stats.pfr == 100
        assert stats.vpip == 100

    def test_nit_classification(self):
        stats = OpponentStats(name="nit")
        # 50 hands: 5 VPIP (10%), 4 PFR (8%)
        for i in range(50):
            state = _state(player=1)
            if i < 5:
                action = Action(
                    ActionType.RAISE if i < 4 else ActionType.CALL,
                    6 if i < 4 else 1,
                    1,
                )
            else:
                # Opponent folds (must observe to count hand)
                action = Action(ActionType.FOLD, 0, 1)
            stats.observe_action(state, action)
            stats.end_hand()
        assert stats.player_type == PlayerType.NIT

    def test_fish_classification(self):
        stats = OpponentStats(name="fish")
        # Fish: VPIP ~50% (calls a lot), PFR ~5% (rarely raises), AF < 1
        for i in range(60):
            preflop_state = _state(player=1, street=Street.PREFLOP)
            if i < 30:
                stats.observe_action(preflop_state, Action(ActionType.CALL, 1, 1))
                if i < 3:
                    # Rare PFR (overrides previous call into a raise for that hand)
                    stats.observe_action(preflop_state, Action(ActionType.RAISE, 6, 1))
                # Postflop activity for VPIP'd hands
                postflop_state = _state(player=1, street=Street.FLOP)
                if i < 20:
                    stats.observe_action(postflop_state, Action(ActionType.CALL, 5, 1))
                if i < 3:
                    stats.observe_action(postflop_state, Action(ActionType.BET, 5, 1))
            else:
                # Folds preflop on remaining 30 hands
                stats.observe_action(preflop_state, Action(ActionType.FOLD, 0, 1))
            stats.end_hand()
        # Should be ~50% VPIP, ~5% PFR, AF=3/20=0.15 → FISH
        assert stats.player_type == PlayerType.FISH

    def test_aggression_factor(self):
        stats = OpponentStats(name="aggro")
        for _ in range(50):
            state = _state(player=1, street=Street.FLOP)
            stats.observe_action(state, Action(ActionType.BET, 10, 1))
            stats.observe_action(state, Action(ActionType.BET, 10, 1))
            stats.observe_action(state, Action(ActionType.CALL, 10, 1))
            stats.end_hand()
        # AF = 100 bets / 50 calls = 2.0
        assert abs(stats.aggression_factor - 2.0) < 0.01


class TestStrategyMapping:
    def test_unknown_returns_balanced(self):
        assert get_strategy_for_type(PlayerType.UNKNOWN) == 'balanced'

    def test_fish_value_bet(self):
        assert get_strategy_for_type(PlayerType.FISH) == 'value_bet'

    def test_nit_bluff_more(self):
        assert get_strategy_for_type(PlayerType.NIT) == 'bluff_more'

    def test_maniac_call_down(self):
        assert get_strategy_for_type(PlayerType.MANIAC) == 'call_down'
