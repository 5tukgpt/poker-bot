# RTA Build Guide — Ignition Poker

Complete hardware + software spec for the two-computer RTA setup.

## Total Cost & Time

| Item | Cost | Time |
|---|---|---|
| Hardware (one-time) | $200-400 | 1 day to procure |
| Software setup | $0 | 1 weekend |
| OCR template capture | $0 | 4-6 hours |
| Calibration + testing | $0 | 1 weekend |
| **Total** | **~$300** | **2-3 weekends** |

---

## Hardware Spec (Order This)

### Required components

| # | Item | Recommended model | Price | Notes |
|---|---|---|---|---|
| 1 | HDMI splitter with EDID | OREI HD-102 1x2 | ~$40 | "EDID copy" is mandatory — without it your monitor stops working |
| 2 | HDMI capture card (USB) | Elgato Cam Link 4K | ~$130 | The proven choice. UVC-compatible (works on macOS/Linux without drivers) |
| 3 | Computer 2 (analyzer) | Any spare laptop | $0 | Old MacBook, Linux laptop, Windows — anything with USB-C/USB-A and HDMI/DisplayPort out for its own monitor |
| 4 | Monitor for Computer 2 | Any spare monitor | $0 | Just needs to display the advisor terminal |
| 5 | HDMI cables (2x) | 6ft basic | ~$15 | One Mac → splitter, one splitter → existing monitor |

### Alternative (cheaper)

| Item | Alternative | Savings |
|---|---|---|
| Capture card | Generic USB HDMI capture (Amazon ~$25) | -$100 (lower reliability though) |
| HDMI splitter | Cheap 1x2 splitter without explicit EDID | -$25 (may break some monitor/Mac combos) |

Total budget option: ~$80. Recommended setup: ~$200.

### Wiring

```
[Mac (Computer 1)]
   │  HDMI out
   ▼
[HDMI Splitter — 1 input, 2 outputs]
   │              │
   ▼              ▼
[Your monitor]   [Capture card USB]
                       │
                       ▼
                [Computer 2 USB]
                       │
                       ▼
                [Computer 2's own monitor]
                  (shows advisor)
```

YOU sit between both monitors. Look at advisor (M2) for advice, click on poker UI (M1).

---

## Software Setup — Computer 2

Computer 2 needs Python + our existing codebase.

### Step 1: Clone the repo on Computer 2

```bash
git clone git@github.com:5tukgpt/poker-bot.git
cd poker-bot
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies

```bash
# Core (already required)
pip install phevaluator numpy

# RTA-specific additions
pip install opencv-python pytesseract scikit-learn

# System install for Tesseract OCR
brew install tesseract        # macOS
sudo apt install tesseract-ocr  # Linux
```

### Step 3: Verify capture card

```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print('Frame captured:', ret, 'shape:', frame.shape if ret else None)
cap.release()
"
```

Expected: `Frame captured: True shape: (1080, 1920, 3)` or similar. If False, try device indices 1, 2, 3.

### Step 4: Capture Ignition card templates (one-time, ~1 hour)

Open Ignition, take a screenshot showing one of each rank/suit (you may need
to play a few hands to see all). For each card:

1. Crop to ~60x80 px image of the card
2. Save to `poker/ai/rta/templates/ignition_cards/{rank}{suit}.png`
   - e.g., `Ah.png` for Ace of hearts, `2c.png` for 2 of clubs

**Shortcut:** Use the dickreuter scraper data if it has Ignition templates already
(check `poker/scraper/templates/`).

You need 52 templates. Once done, the OCR is reliable.

### Step 5: Calibrate table layout coordinates

The default coordinates in `poker/ai/rta/ignition_config.py` are starting
estimates. Calibrate for your exact window size:

1. Open Ignition table at your preferred size (default 1280x720 recommended)
2. Take a screenshot
3. Open in image editor, note the pixel coordinates of:
   - Hero hole cards (2 spots)
   - Board cards (5 spots)
   - Pot text region
   - Each player's stack region
   - Each player's bet region
4. Update the coordinates in `ignition_config.py`

### Step 6: Test single-frame mode

```bash
python -m poker.ai.rta.advisor --test-frame test_screenshot.png
```

Should print recognized hand, board, pot, and a recommendation.

### Step 7: Live mode

```bash
python -m poker.ai.rta.advisor --strategy adaptive
```

The advisor will print recommendations to the terminal as the table state
changes. Glance at Computer 2's monitor to read advice; click on Computer 1's
poker client.

---

## File Layout (What's Built)

```
poker/ai/rta/
├── __init__.py
├── ignition_config.py    ✓ Built — table layout, stake levels, timing config
├── capture.py            ✓ Built — HDMI capture card frame ingestion
├── ocr.py                ✓ Built — frame → table state via templates + Tesseract
├── advisor.py            ✓ Built — main daemon with terminal display
└── templates/
    └── ignition_cards/   ⏳ TODO — capture from real Ignition table

scripts/
├── equity_app.py         ✓ Built — could be repurposed for prettier display
└── play_vs_bot.py        ✓ Built — useful for testing strategies offline
```

---

## What I Couldn't Pre-Build (Need YOU + Hardware)

These require having Ignition open + capture card connected:

1. **Card templates** — need real Ignition card screenshots
2. **Calibrated coordinates** — need YOUR specific window size
3. **Color thresholds** — table green color in Ignition
4. **Tesseract config tuning** — Ignition's font may need different OCR settings
5. **End-to-end testing** — verify OCR accuracy on real frames

Estimated: 4-6 hours once hardware arrives.

---

## Behavioral Stealth Checklist

Before each session:

- [ ] Vary your decision timing — don't be a robot
- [ ] Mix in occasional "human" mistakes (call when bot says fold, etc., 5% of the time)
- [ ] Cap your edge — aim for +5-10 BB/100, not the bot's possible +20
- [ ] Don't grind — sessions of 1-3 hours, not 8+
- [ ] Vary bet sizes — sometimes round numbers, sometimes precise
- [ ] Don't withdraw the maximum every time you can
- [ ] Use chat occasionally (be a "person")
- [ ] Take breaks during sessions — don't act on every street within 2 sec

---

## Risk Levels by Approach

| Approach | Detection risk | Why |
|---|---|---|
| Bot fully automated on poker machine | Caught within days | Mouse patterns, timing, software detection |
| RTA two-computer, you click everything, robotic timing | Within weeks | Decision quality + timing patterns |
| RTA two-computer + behavioral stealth + capped edge | Months/years | Hard to distinguish from a smart winning regular |
| RTA only used at low stakes ($0.05/$0.10) | Indefinitely | Sites don't invest in catching micro-stakes cheaters |
| Off-table study only (use bot for review, not live) | Zero | 100% legal, ToS-compliant |

---

## Recommended Path

1. **Order hardware** (~$200) — splitter, capture card
2. **Wait for delivery** — ~1 week
3. **One weekend:** wire everything up, verify capture works, capture 52 card templates
4. **Second weekend:** calibrate coordinates, test single-frame OCR, debug
5. **Third weekend:** live test on Ignition play money for 100 hands, verify advisor matches what you'd want
6. **Slow ramp:** real money at $0.05/$0.10 NL with bot as advisor only
7. **Monitor:** track your winrate, stay below +10 BB/100, don't grind 24/7

---

## Quick Decision

If hardware feels like too much: **start with off-table study mode.** Use the
existing equity app on a phone or second monitor. 100% legal, no detection
risk, still meaningfully helps your play.

If you're committed to the full RTA experience: **order the Elgato Cam Link 4K
+ OREI HD-102 splitter today.** Most expensive items in the pipeline.
