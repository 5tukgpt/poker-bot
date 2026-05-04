# Poker Bot

Fork of `dickreuter/Poker` (GPL-3.0) with custom AI engine replacing the original decision maker.

## Architecture

- `poker/` — original dickreuter platform integration (scraper, mouse, GUI, tools)
- `poker/ai/` — **custom AI engine** (game logic, strategies, simulation, training)
- `poker/ai/engine/` — card types, game state, evaluator (phevaluator), table game loop
- `poker/ai/strategy/` — swappable strategies: heuristic, CFR, DQN
- `poker/ai/sim/` — self-play arena + stats tracking (BB/100, VPIP, PFR)
- `poker/ai/train/` — training loops for CFR and DQN

## Key Conventions

- All chip amounts are `int`, never `float`
- Card encoding: integers 0-51 (`(rank-2)*4 + suit`), matching phevaluator
- Strategy contract: `choose_action(GameState, list[ActionType]) -> Action`
- Pure numpy for ML (no PyTorch) — adapted from trading bot DQN pattern
- Heads-up: button = SB, acts first preflop. BB acts first postflop.

## Dependencies

- `phevaluator` — hand evaluation (pip)
- `numpy` — array ops, DQN backbone
- `pytest` — testing
- Original fork deps: opencv-python, pyqt6, tesserocr, pyautogui (for platform layer)

## Commands

```bash
source .venv/bin/activate
python scripts/play.py 1000           # Heuristic vs heuristic simulation
python -m pytest poker/ai/tests/ -v   # Run AI engine tests
```

## Strategy Interface

```python
class PokerStrategy(Protocol):
    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action: ...
    def notify_result(self, state: GameState, payoff: int) -> None: ...
```

## Status

- ✅ Phase 0 DONE: Forked dickreuter/Poker, venv with phevaluator + numpy + pytest
- ✅ Phase 1 DONE: Engine + heuristic + simulation (24 hands/sec, BB/100 tracking)
- ✅ Phase 3 DONE: CFR (5 buckets, 4 actions, MCCFR) + DQN (pure numpy, Double DQN, 64-dim state)
- ⏳ Phase 2 TODO: Wire AI to platform scraper — build `poker/ai/adapter.py` mapping dickreuter table state → GameState; replace `Decision` in `main.py`
- ⏳ Phase 4 TODO: Hybrid ensemble, opponent modeling, Deep CFR

## Training Commands

```bash
python poker/ai/train/train_cfr.py --iterations 10000  # ~25s
python poker/ai/train/train_dqn.py --hands 10000       # ~40s
python scripts/benchmark.py --hands 1000                # all strategies head-to-head
```

## Current Benchmark — In-house cross-product (1000 hands/matchup)

```
               heuristic         dqn   gto_chart        book
heuristic            ---       -50.1       +35.8       +57.0
dqn                +42.7         ---       +14.3        +0.0
gto_chart          -10.1       -17.4         ---       -22.0
book               -52.5        -7.1       -62.6         ---
```

**In-house ranking (avg BB/100):** dqn (+19) > heuristic (+14) > gto_chart (-16) > book (-41)

## Slumbot Benchmark (1000 hands each)

Slumbot is the strongest publicly available HU NLHE bot.

| Strategy | BB/100 vs Slumbot |
|---|---|
| Heuristic | -66.3 |
| DQN | -68.7 |
| Adaptive | -67.2 |

**All three are statistically tied** within ±25 BB/100 confidence interval at 1000 hands.

**Known bug:** `slumbot_state_to_gamestate` has translation issues causing
strategies to misplay (e.g., adaptive open-folding from BB losing 100 chips/hand).
The -65 BB/100 deficit vs Slumbot is partly real, partly the bug. Need to fix
the state translation before claiming definitive Slumbot performance.

## Sessions

Big developments and fixes are recorded in commit history. Notable:
- CFR was algorithmically buggy until commit dc3f126 (4-5x improvement)
- DQN matches heuristic strength after 10K self-play hands
- "By the book" HU NLHE strategy added but loses to pure-equity heuristic
- Opponent modeling + AdaptiveStrategy built (commit 186ade9)
- Cross-product matrix tuning (commit 0b94c84) confirmed DQN as strongest baseline

## Strategy Inventory

| Strategy | Approach | Status |
|---|---|---|
| `heuristic` | Preflop categories + Monte Carlo equity vs pot odds | Strongest baseline |
| `dqn` | Pure-numpy Double DQN, 64-dim state, 10K self-play hands | Near-parity with heuristic |
| `gto_chart` | Hard-coded HU NLHE preflop ranges + equity postflop | Competitive baseline, no training cost |
| `book` | "By the book" HU NLHE — c-bet logic, board texture, mixed strategies, proper bet sizing | Comprehensive but loses to heuristic; rules give up edge in bot-vs-bot |
| `cfr` | External-sampling MCCFR with 5-bucket rule abstraction | **Deprecated** — k-means experiment proved CFR-from-scratch is impractical on a laptop (3.26M info sets after 30K iters, far from converged) |
| `ensemble` | Weighted vote across heuristic + CFR + DQN | Available but currently dragged by stale CFR |

## Key Lessons

1. **CFR-from-scratch is hard.** Even with bug fixes and k-means abstraction, training to convergence requires solver-class hardware. Borrowing published GTO charts is more pragmatic.
2. **DQN works.** Pure-numpy Double DQN reached competitive play in 10K self-play hands. Self-play training is a viable evolution path.
3. **Hand-coded heuristics are surprisingly strong.** Equity vs pot odds is a powerful baseline that's hard to beat with little training.
