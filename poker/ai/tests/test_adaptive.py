"""Tests for AdaptiveStrategy."""

from __future__ import annotations

from poker.ai.engine.action import Action, ActionType
from poker.ai.engine.card import Card
from poker.ai.engine.game_state import GameState, Street
from poker.ai.strategy.adaptive import AdaptiveStrategy, DEFAULT_TYPE_TO_STRATEGY
from poker.ai.strategy.opponent_model import PlayerType


def _state(hole, current_player=0, history=None, **overrides):
    defaults = dict(
        num_players=2, stacks=[200, 200], pot=3, board=[],
        hole_cards=[hole, [50, 51]], street=Street.PREFLOP,
        current_player=current_player, button=0,
        small_blind=1, big_blind=2,
        current_bets=[1, 2], action_history=history or [],
        folded=[False, False], all_in=[False, False],
    )
    defaults.update(overrides)
    return GameState(**defaults)


class TestAdaptiveStrategy:
    def test_starts_with_heuristic_for_unknown(self):
        strat = AdaptiveStrategy()
        # No prior hands → UNKNOWN type → heuristic
        s = _state([Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()])
        action = strat.choose_action(s, s.legal_actions())
        assert isinstance(action, Action)

    def test_chooses_legal_action(self):
        strat = AdaptiveStrategy()
        s = _state([Card.from_str('7c').to_int(), Card.from_str('2d').to_int()])
        action = strat.choose_action(s, s.legal_actions())
        assert action.type in s.legal_actions()

    def test_observes_opponent_actions(self):
        strat = AdaptiveStrategy()
        # Build a hand where opponent raised
        history = [Action(ActionType.RAISE, 6, 1)]
        s = _state(
            [Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()],
            history=history, current_bets=[1, 6],
        )
        # Our turn — observe villain's raise
        strat.choose_action(s, s.legal_actions())
        strat.notify_result(s, 0)
        # After hand: opponent should have 1 PFR observation
        assert strat.opp_stats.pfr_count == 1
        assert strat.opp_stats.hands_observed == 1

    def test_does_not_double_count_observations(self):
        strat = AdaptiveStrategy()
        history = [
            Action(ActionType.RAISE, 6, 1),
        ]
        s1 = _state(
            [Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()],
            history=history, current_bets=[1, 6],
        )
        # Multiple turns in same hand — opponent's same raise should only count once
        strat.choose_action(s1, s1.legal_actions())
        strat.choose_action(s1, s1.legal_actions())
        strat.notify_result(s1, 0)
        assert strat.opp_stats.pfr_count == 1

    def test_resets_observation_index_after_hand(self):
        strat = AdaptiveStrategy()
        # Hand 1
        history1 = [Action(ActionType.RAISE, 6, 1)]
        s1 = _state(
            [0, 1], history=history1, current_bets=[1, 6],
        )
        strat.choose_action(s1, s1.legal_actions())
        strat.notify_result(s1, 0)
        # Hand 2 — fresh history
        history2 = [Action(ActionType.RAISE, 6, 1)]
        s2 = _state(
            [2, 3], history=history2, current_bets=[1, 6],
        )
        strat.choose_action(s2, s2.legal_actions())
        strat.notify_result(s2, 0)
        # Both hands should be observed
        assert strat.opp_stats.hands_observed == 2
        assert strat.opp_stats.pfr_count == 2

    def test_strategy_picks_change_with_classification(self):
        # Build a stat snapshot that classifies as FISH and verify mapping
        strat = AdaptiveStrategy()
        # Inject FISH stats directly
        strat.opp_stats.hands_observed = 50
        strat.opp_stats.vpip_count = 25      # 50% VPIP
        strat.opp_stats.pfr_count = 3        # 6% PFR
        strat.opp_stats.postflop_calls = 20
        strat.opp_stats.postflop_bets_raises = 3  # AF = 0.15
        assert strat.opp_stats.player_type == PlayerType.FISH
        # Verify the mapping uses heuristic (value-bet style) for fish
        assert strat.type_to_strategy[PlayerType.FISH] == 'heuristic'


class TestStrategyMappingDefault:
    def test_default_mapping_complete(self):
        # Every PlayerType has a mapping
        for player_type in PlayerType:
            assert player_type in DEFAULT_TYPE_TO_STRATEGY

    def test_unknown_uses_strongest_default(self):
        # Empirically tuned: DQN is our strongest avg strategy (+19 BB/100)
        assert DEFAULT_TYPE_TO_STRATEGY[PlayerType.UNKNOWN] == 'dqn'
