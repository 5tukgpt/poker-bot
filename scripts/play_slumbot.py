#!/usr/bin/env python3
"""Play our strategies against Slumbot (the public benchmark bot).

Free public API, no login needed. ~1 second per hand.

Usage:
    python scripts/play_slumbot.py heuristic 50
    python scripts/play_slumbot.py book 100
    python scripts/play_slumbot.py dqn 50
    python scripts/play_slumbot.py gto_chart 100
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, '.')

from poker.ai.slumbot_client import play_match


STRATEGIES = {
    'heuristic': lambda: __import__('poker.ai.strategy.heuristic', fromlist=['HeuristicStrategy']).HeuristicStrategy(num_simulations=200),
    'gto_chart': lambda: __import__('poker.ai.strategy.gto_chart', fromlist=['GTOChartStrategy']).GTOChartStrategy(postflop_sims=200),
    'book':      lambda: __import__('poker.ai.strategy.book_strategy', fromlist=['BookStrategy']).BookStrategy(postflop_sims=200),
    'dqn':       None,  # special — needs model load
}


def make_dqn():
    from poker.ai.strategy.dqn import DQNAgent
    agent = DQNAgent(training=False)
    agent.load('poker/ai/models/dqn_weights.npz')
    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('strategy', choices=list(STRATEGIES.keys()), help='Strategy to test')
    parser.add_argument('hands', nargs='?', type=int, default=50, help='Number of hands (default 50)')
    args = parser.parse_args()

    if args.strategy == 'dqn':
        strategy = make_dqn()
    else:
        strategy = STRATEGIES[args.strategy]()

    print(f"Playing {args.hands} hands of {args.strategy} vs Slumbot")
    print(f"Settings: 50/100 blinds, 200BB stack")
    print("=" * 60)

    start = time.time()
    result = play_match(strategy, args.hands, verbose=True)
    elapsed = time.time() - start

    print("=" * 60)
    print(f"FINAL RESULTS for {args.strategy} vs Slumbot:")
    print(f"  Hands played: {result['hands_played']}/{args.hands}")
    print(f"  Total winnings: {result['total_winnings']:+d} chips")
    print(f"  BB/100: {result['bb_per_100']:+.1f}")
    print(f"  Errors: {result['errors']}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(result['hands_played'],1):.1f}s/hand)")

    if result['bb_per_100'] > 0:
        print(f"\n  ✓ {args.strategy} is winning vs Slumbot")
    elif result['bb_per_100'] > -50:
        print(f"\n  ~ {args.strategy} is competitive (within 50 BB/100 of Slumbot)")
    else:
        print(f"\n  ✗ {args.strategy} is losing significantly")


if __name__ == '__main__':
    main()
