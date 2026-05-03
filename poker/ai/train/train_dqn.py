"""Train DQN agent via self-play."""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, '.')

from poker.ai.strategy.dqn import DQNAgent
from poker.ai.strategy.heuristic import HeuristicStrategy
from poker.ai.sim.arena import Arena


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands', type=int, default=10000)
    parser.add_argument('--output', type=str, default='poker/ai/models/dqn_weights.npz')
    parser.add_argument('--opponent', choices=['heuristic', 'self'], default='heuristic')
    args = parser.parse_args()

    print(f"Training DQN for {args.hands} hands vs {args.opponent}")
    print("=" * 60)

    dqn = DQNAgent(training=True)
    if args.opponent == 'heuristic':
        opponent = HeuristicStrategy(num_simulations=100)
    else:
        opponent = DQNAgent(training=True)

    arena = Arena(
        strategies=[dqn, opponent],
        names=["DQN", "Opponent"],
    )

    start = time.time()
    stats = arena.play(args.hands, verbose=True)
    elapsed = time.time() - start

    print("=" * 60)
    print(f"Trained on {args.hands} hands in {elapsed:.1f}s")
    for s in stats:
        print(f"  {s.summary()}")
    print(f"\nFinal epsilon: {dqn._epsilon():.3f}")
    print(f"Total steps: {dqn.total_steps}")
    print(f"Saving weights to {args.output}")
    dqn.save(args.output)
    print("Done.")


if __name__ == '__main__':
    main()
