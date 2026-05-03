from __future__ import annotations

from typing import TYPE_CHECKING

from ..engine.table import Table
from .stats import PlayerStats

if TYPE_CHECKING:
    from ..strategy.base import PokerStrategy


class Arena:
    def __init__(
        self,
        strategies: list[PokerStrategy],
        names: list[str] | None = None,
        small_blind: int = 1,
        big_blind: int = 2,
        starting_stack: int = 200,
    ) -> None:
        self.strategies = strategies
        self.names = names or [f"Player_{i}" for i in range(len(strategies))]
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stack = starting_stack

    def play(self, num_hands: int, verbose: bool = False) -> list[PlayerStats]:
        table = Table(
            self.strategies,
            self.small_blind,
            self.big_blind,
            self.starting_stack,
        )
        stats = [PlayerStats(name=n, big_blind=self.big_blind) for n in self.names]

        for hand_num in range(num_hands):
            deltas = table.play_hand()

            for i, delta in enumerate(deltas):
                stats[i].hands_played += 1
                stats[i].total_profit += delta
                if delta > 0:
                    stats[i].wins += 1

            if table.stacks[0] <= 0 or table.stacks[1] <= 0:
                for i in range(len(self.strategies)):
                    table.stacks[i] = self.starting_stack

            if verbose and (hand_num + 1) % 100 == 0:
                print(f"--- Hand {hand_num + 1} ---")
                for s in stats:
                    print(f"  {s.summary()}")

        return stats
