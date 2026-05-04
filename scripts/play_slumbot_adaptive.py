#!/usr/bin/env python3
"""Play AdaptiveStrategy against Slumbot. Tracks Slumbot's stats live and adapts."""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, '.')

from poker.ai.slumbot_client import play_match
from poker.ai.strategy.adaptive import AdaptiveStrategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('hands', nargs='?', type=int, default=200)
    parser.add_argument('--verbose-switching', action='store_true',
                        help='Print when strategy switches')
    args = parser.parse_args()

    adaptive = AdaptiveStrategy(
        opponent_name='Slumbot',
        verbose_switching=args.verbose_switching,
    )

    print(f"Playing {args.hands} hands of AdaptiveStrategy vs Slumbot")
    print("(Adaptive learns Slumbot's tendencies and switches strategy as it goes)")
    print("=" * 65)

    start = time.time()
    result = play_match(adaptive, args.hands, verbose=True)
    elapsed = time.time() - start

    print("=" * 65)
    print(f"FINAL RESULTS:")
    print(f"  Hands played: {result['hands_played']}/{args.hands}")
    print(f"  BB/100: {result['bb_per_100']:+.1f}")
    print(f"  Errors: {result['errors']}")
    print(f"  Time: {elapsed:.0f}s")
    print()
    print(f"  Slumbot profile after {adaptive.opp_stats.hands_observed} hands:")
    print(f"    {adaptive.get_opponent_summary()}")
    print(f"  Strategy used: {adaptive.type_to_strategy[adaptive.opp_stats.player_type]}")


if __name__ == '__main__':
    main()
