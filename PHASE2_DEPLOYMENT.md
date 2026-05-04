# Phase 2: Live Platform Deployment Guide

When you're ready to deploy the bot on a real poker site, follow this guide.

## Prerequisites

- A poker client (PokerStars, GGPoker, PartyPoker, Bovada, etc.)
- macOS or Windows machine running the bot
- Python venv with deps installed (already done)

## Step 1: Pick a poker client

Recommendations by use case:

| Use case | Client | Notes |
|---|---|---|
| Learn / test | **PokerStars play money** | Free, biggest player base, good UI |
| Real money small | **PokerStars / GGPoker** | Highest traffic, ToS bans bots — use at own risk |
| Private games | **ClubGG / PPPoker** | Private clubs, less anti-bot enforcement |
| Crypto-friendly | **CoinPoker / SwC Poker** | Crypto deposits, generally tolerate bots more |
| Bot-only (legal) | **MIT Pokerbots** annual | Compete legitimately |

**Strong recommendation:** Start with PokerStars play money. Free, no risk, and the existing dickreuter scraper has the most maturity for it.

## Step 2: One-time setup (~30 minutes)

### A. Install missing dependencies

```bash
cd ~/Projects/poker-bot
source .venv/bin/activate
pip install opencv-python pyqt6 tesserocr pyautogui virtualbox
brew install tesseract  # macOS
```

### B. Configure the table scraper

The existing dickreuter GUI lets you "teach" the bot where elements appear on screen:

```bash
python -m poker.gui.gui_launcher  # may fail without MongoDB; see Step C
```

You'll point and click on:
- Hole card positions (2 spots)
- Community card positions (5 spots)
- Pot value display area
- Each player's stack display
- Button position indicator
- Each action button (Fold/Call/Raise/Check)
- Bet sizing slider/buttons

This produces a "table_dict" that gets saved (currently to MongoDB).

### C. Replace MongoDB with local storage

The existing fork talks to a remote MongoDB at deepermind-pokerbot.com which may
be down. Modifications needed in `poker/tools/`:

1. **mongo_manager.py** — add a `LocalStorageManager` class that uses local JSON
   files instead of HTTP calls. Stub methods:
   - `get_table(name)` → load from `poker/tables/<name>.json`
   - `save_table(name, dict)` → save to `poker/tables/<name>.json`
   - `increment_plays(name)` → no-op or local counter
2. **strategy_handler.py** — use local strategy files instead of remote
3. **game_logger.py** — log to local SQLite or skip entirely

Estimated work: 2-3 hours.

## Step 3: Wire AI into main.py (15 minutes)

Replace lines 206-207 of `poker/main.py`:

```python
# OLD:
m = run_montecarlo_wrapper(strategy, ...)
d = Decision(table, history, strategy, self.game_logger)
d.make_decision(table, history, strategy, self.game_logger)

# NEW:
from poker.ai.decision_v2 import DecisionV2
from scripts.main_ai import build_strategy

# Load once at startup; reuse for all hands so AdaptiveStrategy stats accumulate
if not hasattr(self, '_ai_strategy'):
    self._ai_strategy = build_strategy('adaptive')  # or 'heuristic' / 'dqn' / etc.

d = DecisionV2(table, strategy_engine=self._ai_strategy)
d.make_decision(table, history, strategy, self.game_logger)
```

The AdaptiveStrategy will track each opponent's stats over time and pick the
best counter-strategy.

## Step 4: Smoke test on play money (~1 hour)

1. Open poker client at a 6-max NL play money table
2. Run `python -m poker.main`
3. Watch the bot play — check the GUI shows correct cards/pot/stacks
4. Verify it makes reasonable decisions (no all-in 72o nonsense)
5. After 50 hands, check the win/loss rate

If it's losing big on play money, investigate:
- Card recognition errors (most common)
- Stack/pot OCR errors
- Action button click misses

## Step 5: Choose deployment level

After play-money success:

| Level | Risk | Reward |
|---|---|---|
| **Play money only** | None — free | Honing the bot |
| **Free home games** | None — friends know | Real poker conditions, no money |
| **Micro stakes ($0.01/$0.02)** | ToS violation risk + small $ | Real money validation |
| **Higher stakes** | Same ToS + bigger $ | Profit (if bot is good) |

**Strongly recommend:** Play-money or private home games. Avoid real-money sites
that ban bots — your account + funds will be frozen if caught.

## Step 6: Monitor + iterate

While playing, monitor:
- BB/100 — should be >0 after 1000+ hands
- Errors in OCR/scraping (logs)
- Adaptive strategy switches (verbose mode)
- Strategy classification of opponents (fish/TAG/etc.)

Iteration: after each session, look at hand histories where the bot lost big.
Often it's a state translation issue, not strategy weakness.

## Useful Files

- `poker/ai/adapter.py` — translates scraper state → our GameState
- `poker/ai/decision_v2.py` — drop-in Decision replacement
- `poker/ai/strategy/adaptive.py` — the recommended live strategy
- `scripts/main_ai.py` — strategy factory, smoke-test entry point

## What's NOT Covered

- Multi-table support (single table only for now)
- Tournament mode (cash games only)
- Hand history database (the original GameLogger writes to MongoDB; replace with SQLite)

## Estimated Total Effort

- One-time setup: ~3-4 hours
- Per-platform tuning: ~1-2 hours each
- Ongoing monitoring: 30 min/day during early weeks
