#!/usr/bin/env python3
"""Cross-product benchmark: every strategy vs every opponent.

Result: a table showing which strategy wins by how much vs each opponent.
Used to tune the AdaptiveStrategy mapping.
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, '.')

from poker.ai.sim.arena import Arena
from poker.ai.strategy.book_strategy import BookStrategy
from poker.ai.strategy.dqn import DQNAgent
from poker.ai.strategy.gto_chart import GTOChartStrategy
from poker.ai.strategy.heuristic import HeuristicStrategy
from poker.ai.strategy.opponent_model import OpponentStats


def make_dqn():
    a = DQNAgent(training=False)
    a.load('poker/ai/models/dqn_weights.npz')
    return a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands', type=int, default=2000, help='hands per matchup')
    args = parser.parse_args()

    factories = {
        'heuristic': lambda: HeuristicStrategy(num_simulations=100),
        'dqn':       make_dqn,
        'gto_chart': lambda: GTOChartStrategy(postflop_sims=100),
        'book':      lambda: BookStrategy(postflop_sims=100),
    }

    names = list(factories.keys())
    print(f"Cross-product benchmark: {len(names)} strategies x {len(names)} opponents")
    print(f"Hands per matchup: {args.hands}")
    print(f"Total matchups: {len(names) * (len(names) - 1)}")
    print("=" * 70)

    # row_strategy → col_opponent → bb_per_100, opponent_classification
    results: dict[tuple[str, str], tuple[float, str]] = {}

    start_total = time.time()
    for row in names:
        for col in names:
            if row == col:
                continue
            r_strat = factories[row]()
            c_strat = factories[col]()
            # Use opponent stats to classify the OPPONENT (col) from row's POV
            stats_observer = OpponentStats(name=col)
            arena = Arena([r_strat, c_strat], names=[row, col])
            t = time.time()
            stats = arena.play(args.hands, verbose=False)
            elapsed = time.time() - t

            # Re-play a few hands to gather classification data
            # (skip the actual classification gathering for now — we'd need to instrument arena)
            results[(row, col)] = (stats[0].bb_per_100, "?")
            print(f"  {row:11s} vs {col:11s}: {stats[0].bb_per_100:+8.1f} BB/100  [{elapsed:.0f}s]")

    elapsed_total = time.time() - start_total
    print(f"\nTotal time: {elapsed_total:.0f}s")
    print()

    # Print matrix
    print("BB/100 MATRIX (row vs col, positive = row winning):")
    print(f"{'':12s}", end='')
    for n in names:
        print(f"{n:>12s}", end='')
    print()
    for row in names:
        print(f"{row:12s}", end='')
        for col in names:
            if row == col:
                print(f"{'---':>12s}", end='')
            else:
                bb, _ = results[(row, col)]
                print(f"{bb:+12.1f}", end='')
        print()

    # Find best counter-strategy per opponent
    print("\nBEST COUNTER per opponent:")
    for col in names:
        best_strategy = None
        best_score = float('-inf')
        for row in names:
            if row == col:
                continue
            bb, _ = results[(row, col)]
            if bb > best_score:
                best_score = bb
                best_strategy = row
        print(f"  vs {col:11s} → {best_strategy:11s} ({best_score:+.1f} BB/100)")


if __name__ == '__main__':
    main()
