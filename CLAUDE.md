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

## Current Benchmark

After 50K CFR iterations + 10K DQN training hands. 500 hands per matchup:

```
           heuristic       cfr       dqn  ensemble
heuristic        ---   +1860.0      -0.6     +40.5
cfr          -1860.0       ---   -1500.2   -1855.0
dqn             +0.6   +1500.2       ---      +8.6
ensemble       -40.5   +1855.0      -8.6       ---
```

Ranking: **heuristic ≈ ensemble ≈ dqn** (within 50 BB/100), CFR far behind.

Notes:
- DQN reached parity with heuristic after 10K self-play hands (epsilon=0.05)
- CFR's 5-bucket abstraction is too crude → known limitation, needs richer abstraction or longer training
- Ensemble is dragged down slightly by CFR's bad votes; rebalancing weights could help
