"""Train CFR strategy via external-sampling MCCFR."""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, '.')

from poker.ai.strategy.cfr import CFRTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--output', type=str, default='poker/ai/models/cfr_strategy.json')
    parser.add_argument('--small-blind', type=int, default=1)
    parser.add_argument('--big-blind', type=int, default=2)
    parser.add_argument('--stack', type=int, default=200)
    args = parser.parse_args()

    print(f"Training CFR for {args.iterations} iterations")
    print(f"  SB={args.small_blind}, BB={args.big_blind}, stack={args.stack}")
    print("=" * 60)

    trainer = CFRTrainer(
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        starting_stack=args.stack,
    )

    start = time.time()
    trainer.train(args.iterations, verbose=True)
    elapsed = time.time() - start

    print("=" * 60)
    print(f"Trained {args.iterations} iters in {elapsed:.1f}s")
    print(f"Info sets discovered: {len(trainer.regret_sum)}")
    print(f"Saving to {args.output}")
    trainer.save(args.output)
    print("Done.")


if __name__ == '__main__':
    main()
