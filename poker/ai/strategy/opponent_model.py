"""Opponent modeling: track opponent stats over time, classify type, adapt.

Standard poker statistics:
- VPIP: % of hands voluntarily put money in pot (preflop)
- PFR: % of hands raised preflop
- AF: aggression factor = (bets+raises) / calls (postflop)
- 3B: 3-bet frequency (re-raise preflop)
- WTSD: went to showdown frequency

Player types (from VPIP/PFR/AF):
- TAG (tight-aggressive):  VPIP 18-25, PFR 15-22, AF 2-3   — solid winning player
- LAG (loose-aggressive):  VPIP 30-40, PFR 25-35, AF 3-5   — tricky strong player
- Nit:                     VPIP <15,   PFR <12              — too tight, fold to aggression
- Fish (loose-passive):    VPIP >40,   PFR <10,  AF <1     — calls too much, exploit by value betting
- Maniac:                  VPIP 40+,   PFR 30+,  AF 5+     — bluffs too much, exploit by calling more
- Unknown:                 fewer than 30 hands tracked
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine.action import Action, ActionType
    from ..engine.game_state import GameState


class PlayerType(Enum):
    UNKNOWN = "unknown"
    NIT = "nit"
    TAG = "tag"
    LAG = "lag"
    FISH = "fish"
    MANIAC = "maniac"


@dataclass
class OpponentStats:
    """Tracks an opponent's behavioral stats across hands."""
    name: str = "opponent"
    hands_observed: int = 0
    vpip_count: int = 0           # voluntarily put in pot preflop
    pfr_count: int = 0            # raised preflop
    threebet_count: int = 0       # 3-bet preflop
    threebet_opportunities: int = 0
    postflop_bets_raises: int = 0
    postflop_calls: int = 0
    showdowns: int = 0
    showdown_wins: int = 0

    # Per-hand temporary state (reset each hand)
    _saw_preflop_action: bool = False
    _put_money_in_preflop: bool = False
    _raised_preflop: bool = False
    _faced_3bet_chance: bool = False
    _3bet_this_hand: bool = False

    @property
    def vpip(self) -> float:
        if self.hands_observed == 0:
            return 0.0
        return self.vpip_count / self.hands_observed * 100

    @property
    def pfr(self) -> float:
        if self.hands_observed == 0:
            return 0.0
        return self.pfr_count / self.hands_observed * 100

    @property
    def threebet_pct(self) -> float:
        if self.threebet_opportunities == 0:
            return 0.0
        return self.threebet_count / self.threebet_opportunities * 100

    @property
    def aggression_factor(self) -> float:
        """AF = (bets+raises) / calls postflop. Higher = more aggressive."""
        if self.postflop_calls == 0:
            return float(self.postflop_bets_raises) if self.postflop_bets_raises > 0 else 0.0
        return self.postflop_bets_raises / self.postflop_calls

    @property
    def player_type(self) -> PlayerType:
        if self.hands_observed < 30:
            return PlayerType.UNKNOWN

        v, p, af = self.vpip, self.pfr, self.aggression_factor

        if v < 15 and p < 12:
            return PlayerType.NIT
        if v >= 40 and p < 10 and af < 1.0:
            return PlayerType.FISH
        if v >= 40 and p >= 30 and af >= 4.0:
            return PlayerType.MANIAC
        if 18 <= v <= 26 and 15 <= p <= 22 and 1.5 <= af <= 3.5:
            return PlayerType.TAG
        if 28 <= v <= 42 and 22 <= p <= 35 and af >= 2.5:
            return PlayerType.LAG
        return PlayerType.UNKNOWN

    def summary(self) -> str:
        return (
            f"{self.name} ({self.hands_observed}h): "
            f"VPIP={self.vpip:.0f}/PFR={self.pfr:.0f}/3B={self.threebet_pct:.0f}/"
            f"AF={self.aggression_factor:.1f} → {self.player_type.value}"
        )

    def observe_action(self, state: 'GameState', action: 'Action') -> None:
        """Update stats from observing an opponent action."""
        from ..engine.action import ActionType
        from ..engine.game_state import Street

        bb = state.big_blind
        is_voluntary = (action.amount > 0 and action.type != ActionType.FOLD) \
                       and not (state.street == Street.PREFLOP and action.amount == bb)

        if state.street == Street.PREFLOP:
            if action.type == ActionType.FOLD:
                pass
            elif action.type in (ActionType.CALL, ActionType.CHECK):
                if action.amount > 0:
                    self._put_money_in_preflop = True
            elif action.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                self._put_money_in_preflop = True
                # PFR if it's the first raise
                if not self._raised_preflop:
                    self._raised_preflop = True
                # 3-bet check: if facing a previous raise, it's a 3-bet
                opp_raises_before = sum(
                    1 for a in state.action_history
                    if a.player_idx != action.player_idx
                    and a.type in (ActionType.BET, ActionType.RAISE)
                )
                if opp_raises_before >= 1 and not self._3bet_this_hand:
                    self._3bet_this_hand = True
        else:
            # Postflop tracking
            if action.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                self.postflop_bets_raises += 1
            elif action.type == ActionType.CALL:
                self.postflop_calls += 1

        self._saw_preflop_action = True

    def end_hand(self, went_to_showdown: bool = False, won_at_showdown: bool = False) -> None:
        """Call at the end of each hand to commit per-hand counters."""
        if self._saw_preflop_action:
            self.hands_observed += 1
            if self._put_money_in_preflop:
                self.vpip_count += 1
            if self._raised_preflop:
                self.pfr_count += 1
            if self._faced_3bet_chance or self._3bet_this_hand:
                self.threebet_opportunities += 1
                if self._3bet_this_hand:
                    self.threebet_count += 1

        if went_to_showdown:
            self.showdowns += 1
            if won_at_showdown:
                self.showdown_wins += 1

        # Reset per-hand state
        self._saw_preflop_action = False
        self._put_money_in_preflop = False
        self._raised_preflop = False
        self._faced_3bet_chance = False
        self._3bet_this_hand = False


def get_strategy_for_type(player_type: PlayerType) -> str:
    """Return recommended counter-strategy for a given opponent type."""
    return {
        PlayerType.UNKNOWN: 'balanced',
        PlayerType.NIT:     'bluff_more',          # they fold too much → bluff them
        PlayerType.TAG:     'balanced',            # solid players → play GTO
        PlayerType.LAG:     'tighten_up',          # aggressive → wait for hands
        PlayerType.FISH:    'value_bet',           # they call too much → bet for value, no bluffs
        PlayerType.MANIAC:  'call_down',           # they bluff too much → call wider
    }[player_type]
