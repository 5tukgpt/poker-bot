#!/usr/bin/env python3
"""Quick simulation: two heuristic bots play N hands."""

import sys
import time

sys.path.insert(0, '.')

from poker.ai.strategy.heuristic import HeuristicStrategy
from poker.ai.sim.arena import Arena


def main() -> None:
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    s1 = HeuristicStrategy(aggression=1.0, num_simulations=300)
    s2 = HeuristicStrategy(aggression=1.0, num_simulations=300)

    arena = Arena(
        strategies=[s1, s2],
        names=["Hero", "Villain"],
        small_blind=1,
        big_blind=2,
        starting_stack=200,
    )

    print(f"Playing {num_hands} hands: Hero vs Villain (both heuristic)")
    print("=" * 60)

    start = time.time()
    stats = arena.play(num_hands, verbose=True)
    elapsed = time.time() - start

    print("=" * 60)
    print("FINAL RESULTS:")
    for s in stats:
        print(f"  {s.summary()}")
    print(f"\nCompleted in {elapsed:.1f}s ({num_hands / elapsed:.0f} hands/sec)")


if __name__ == '__main__':
    main()
