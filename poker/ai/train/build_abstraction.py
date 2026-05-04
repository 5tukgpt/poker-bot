"""Precompute equity-clustering buckets for CFR.

Run once. Output saved to poker/ai/models/abstraction.json (~few MB).
Subsequent CFR training/play uses the cached buckets.
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, '.')

from poker.ai.strategy.abstraction import EquityBucketer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflop-clusters', type=int, default=50)
    parser.add_argument('--flop-clusters', type=int, default=50)
    parser.add_argument('--turn-clusters', type=int, default=50)
    parser.add_argument('--river-clusters', type=int, default=50)
    parser.add_argument('--rollouts', type=int, default=200,
                        help='Equity rollouts per histogram (precomputation)')
    parser.add_argument('--postflop-samples', type=int, default=3000,
                        help='Random hands sampled per postflop street for clustering')
    parser.add_argument('--output', type=str, default='poker/ai/models/abstraction.json')
    args = parser.parse_args()

    print(f"Precomputing CFR abstraction:")
    print(f"  preflop: {args.preflop_clusters} buckets")
    print(f"  flop: {args.flop_clusters} buckets")
    print(f"  turn: {args.turn_clusters} buckets")
    print(f"  river: {args.river_clusters} buckets")
    print(f"  rollouts/histogram: {args.rollouts}")
    print(f"  postflop samples: {args.postflop_samples}")
    print("=" * 60)

    start = time.time()
    bucketer = EquityBucketer.precompute(
        clusters_per_street={
            'preflop': args.preflop_clusters,
            'flop': args.flop_clusters,
            'turn': args.turn_clusters,
            'river': args.river_clusters,
        },
        save_path=args.output,
        rollouts=args.rollouts,
        postflop_samples=args.postflop_samples,
        verbose=True,
    )
    elapsed = time.time() - start

    print("=" * 60)
    print(f"Total: {elapsed:.1f}s")
    print(f"Total buckets: {bucketer.total_buckets()}")
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
