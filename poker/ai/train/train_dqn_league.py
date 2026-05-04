"""Train DQN against a rotating league of opponents for generalist strength.

Instead of training only against the heuristic (which leads to overfitting),
this trainer rotates between heuristic, book strategy, GTO chart, and DQN's
own past versions. Result: a more robust DQN that handles different opponent
styles.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time

sys.path.insert(0, '.')

from poker.ai.sim.arena import Arena
from poker.ai.strategy.book_strategy import BookStrategy
from poker.ai.strategy.dqn import DQNAgent
from poker.ai.strategy.gto_chart import GTOChartStrategy
from poker.ai.strategy.heuristic import HeuristicStrategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-hands', type=int, default=20000)
    parser.add_argument('--hands-per-opponent', type=int, default=2000)
    parser.add_argument('--output', type=str, default='poker/ai/models/dqn_league.npz')
    parser.add_argument('--starting-weights', type=str,
                        help='Optional: load existing weights to continue training')
    args = parser.parse_args()

    # The student we're training
    dqn = DQNAgent(training=True)
    if args.starting_weights:
        dqn.load(args.starting_weights)
        print(f"Loaded starting weights from {args.starting_weights}")

    # Rotating opponents (excluding 'self' which we add dynamically)
    opponents = [
        ('heuristic', lambda: HeuristicStrategy(num_simulations=100)),
        ('gto_chart', lambda: GTOChartStrategy(postflop_sims=100)),
        ('book',      lambda: BookStrategy(postflop_sims=100)),
    ]

    print(f"League training: {args.total_hands} hands across {len(opponents)+1} opponent types")
    print(f"Hands per opponent rotation: {args.hands_per_opponent}")
    print("=" * 60)

    total_hands_played = 0
    rotation = 0
    start = time.time()

    while total_hands_played < args.total_hands:
        # Cycle through opponents (heuristic, gto, book, self, heuristic, ...)
        opp_idx = rotation % (len(opponents) + 1)

        if opp_idx < len(opponents):
            opp_name, opp_factory = opponents[opp_idx]
            opponent = opp_factory()
        else:
            opp_name = 'self'
            # Snapshot current DQN as opponent (frozen, not training)
            opponent = DQNAgent(training=False)
            opponent.q_network.copy_weights_from(dqn.q_network)
            opponent.target_network.copy_weights_from(dqn.target_network)

        hands_this_round = min(args.hands_per_opponent, args.total_hands - total_hands_played)
        print(f"[Rotation {rotation+1}] vs {opp_name} for {hands_this_round} hands "
              f"(total so far: {total_hands_played}, eps: {dqn._epsilon():.3f})")

        arena = Arena([dqn, opponent], names=['DQN', opp_name])
        stats = arena.play(hands_this_round, verbose=False)

        total_hands_played += hands_this_round
        rotation += 1

        bb_per_100 = stats[0].bb_per_100
        print(f"  Result: DQN {bb_per_100:+.1f} BB/100  ({stats[0].total_profit:+d} chips)")

    elapsed = time.time() - start
    print("=" * 60)
    print(f"Trained on {total_hands_played} hands in {elapsed:.0f}s")
    print(f"Final epsilon: {dqn._epsilon():.3f}")
    print(f"Saving to {args.output}")
    dqn.save(args.output)
    print("Done.")


if __name__ == '__main__':
    main()
