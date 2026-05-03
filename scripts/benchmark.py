"""Benchmark all strategies head-to-head."""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, '.')

from poker.ai.sim.arena import Arena
from poker.ai.strategy.heuristic import HeuristicStrategy


def build_strategies(num_sims: int) -> dict:
    strategies = {
        'heuristic': lambda: HeuristicStrategy(num_simulations=num_sims),
    }

    cfr_path = 'poker/ai/models/cfr_strategy.json'
    if os.path.exists(cfr_path):
        from poker.ai.strategy.cfr import CFRStrategy
        strategies['cfr'] = lambda: CFRStrategy(strategy_path=cfr_path)

    dqn_path = 'poker/ai/models/dqn_weights.npz'
    if os.path.exists(dqn_path):
        from poker.ai.strategy.dqn import DQNAgent
        def make_dqn():
            agent = DQNAgent(training=False)
            agent.load(dqn_path)
            return agent
        strategies['dqn'] = make_dqn

    return strategies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands', type=int, default=1000)
    parser.add_argument('--sims', type=int, default=100)
    args = parser.parse_args()

    strategies = build_strategies(args.sims)
    names = list(strategies.keys())

    print(f"Benchmarking {len(names)} strategies: {names}")
    print(f"Hands per matchup: {args.hands}")
    print("=" * 60)

    results = {}
    for i, name_a in enumerate(names):
        for name_b in names[i:]:
            if name_a == name_b:
                continue
            print(f"\n{name_a} vs {name_b}")
            arena = Arena(
                strategies=[strategies[name_a](), strategies[name_b]()],
                names=[name_a, name_b],
            )
            start = time.time()
            stats = arena.play(args.hands, verbose=False)
            elapsed = time.time() - start
            results[(name_a, name_b)] = stats
            for s in stats:
                print(f"  {s.summary()}")
            print(f"  ({args.hands} hands in {elapsed:.1f}s)")

    print("\n" + "=" * 60)
    print("BB/100 SUMMARY:")
    for (a, b), stats in results.items():
        sign = "+" if stats[0].bb_per_100 > 0 else ""
        print(f"  {a} vs {b}: {sign}{stats[0].bb_per_100:.1f} BB/100 (for {a})")


if __name__ == '__main__':
    main()
