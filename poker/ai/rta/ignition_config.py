"""Ignition Poker / Bovada platform-specific configuration.

Ignition + Bovada run on the PaiWangLuo network with shared infrastructure.
This config covers their 6-max NL Hold'em cash game tables (anonymous mode).

To use this config, the OCR pipeline reads card/text positions from the
captured table screenshot using these coordinates (relative to table origin).

NOTE: Coordinates are STARTING ESTIMATES based on standard Ignition layout
at 1280x720 table window. You must calibrate for YOUR specific window size
using `poker/ai/rta/calibrate.py` (TBD).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CardRegion:
    """Pixel rectangle for a card (relative to table top-left)."""
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class TextRegion:
    """Pixel rectangle for a text field (pot/stack/bet)."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class IgnitionTableLayout:
    """6-max table coordinate map. All coords relative to table window top-left.

    Standard Ignition table window: 1280x720 px (dealer view at default zoom).
    Player seats numbered 0-5 starting from hero (bottom center) clockwise:
        0 = hero (bottom)
        1 = bottom-right
        2 = right
        3 = top
        4 = left
        5 = bottom-left
    """

    # Table window dimensions
    window_width: int = 1280
    window_height: int = 720

    # Hole cards (hero only — opponents are face down on anonymous tables)
    hero_card1: CardRegion = field(default_factory=lambda: CardRegion(548, 530, 60, 80))
    hero_card2: CardRegion = field(default_factory=lambda: CardRegion(615, 530, 60, 80))

    # Community cards (5 board positions)
    board_cards: list[CardRegion] = field(default_factory=lambda: [
        CardRegion(425, 295, 60, 80),
        CardRegion(495, 295, 60, 80),
        CardRegion(565, 295, 60, 80),
        CardRegion(635, 295, 60, 80),
        CardRegion(705, 295, 60, 80),
    ])

    # Pot total (center of table)
    pot_text: TextRegion = field(default_factory=lambda: TextRegion(560, 250, 160, 30))

    # Per-seat: chip stack, current bet, dealer button indicator
    # Order: hero (seat 0) then clockwise
    seat_stacks: list[TextRegion] = field(default_factory=lambda: [
        TextRegion(540, 620, 200, 25),  # hero
        TextRegion(960, 540, 200, 25),  # bottom-right
        TextRegion(1080, 280, 200, 25),  # right
        TextRegion(540, 80, 200, 25),    # top
        TextRegion(40, 280, 200, 25),    # left
        TextRegion(160, 540, 200, 25),   # bottom-left
    ])

    seat_bets: list[TextRegion] = field(default_factory=lambda: [
        TextRegion(580, 480, 120, 25),   # hero
        TextRegion(820, 480, 120, 25),
        TextRegion(900, 320, 120, 25),
        TextRegion(580, 200, 120, 25),
        TextRegion(220, 320, 120, 25),
        TextRegion(300, 480, 120, 25),
    ])

    # Dealer button position indicators (small disc sprite)
    seat_button_pos: list[tuple[int, int]] = field(default_factory=lambda: [
        (660, 580),   # hero
        (920, 510),
        (1020, 320),
        (660, 130),
        (120, 320),
        (240, 510),
    ])

    # Action buttons (where to click — needed for monitoring what's available)
    fold_button: TextRegion = field(default_factory=lambda: TextRegion(720, 660, 100, 35))
    check_call_button: TextRegion = field(default_factory=lambda: TextRegion(840, 660, 100, 35))
    bet_raise_button: TextRegion = field(default_factory=lambda: TextRegion(960, 660, 100, 35))
    bet_amount_field: TextRegion = field(default_factory=lambda: TextRegion(1100, 660, 100, 35))


# Standard Ignition cash game configurations
@dataclass(frozen=True)
class IgnitionStakeLevel:
    name: str
    small_blind: float  # in dollars
    big_blind: float
    min_buyin_bb: int
    max_buyin_bb: int


IGNITION_CASH_STAKES = {
    'micro_2nl':   IgnitionStakeLevel('$0.01/$0.02 NL',  0.01, 0.02, 40, 100),
    'micro_5nl':   IgnitionStakeLevel('$0.02/$0.05 NL',  0.02, 0.05, 40, 100),
    'micro_10nl':  IgnitionStakeLevel('$0.05/$0.10 NL',  0.05, 0.10, 40, 100),
    'low_25nl':    IgnitionStakeLevel('$0.10/$0.25 NL',  0.10, 0.25, 40, 100),
    'low_50nl':    IgnitionStakeLevel('$0.25/$0.50 NL',  0.25, 0.50, 40, 100),
    'low_100nl':   IgnitionStakeLevel('$0.50/$1.00 NL',  0.50, 1.00, 40, 100),
    'mid_200nl':   IgnitionStakeLevel('$1/$2 NL',        1.00, 2.00, 40, 100),
    'mid_400nl':   IgnitionStakeLevel('$2/$4 NL',        2.00, 4.00, 40, 100),
    'mid_1000nl':  IgnitionStakeLevel('$5/$10 NL',       5.00, 10.00, 40, 100),
}


# Card recognition templates — we'll need to capture these from a real Ignition table
# Each card (e.g., "Ah", "2c") needs a template image at ~60x80 px for OpenCV matchTemplate
# Stored in: poker/ai/rta/templates/ignition_cards/{rank}{suit}.png
CARD_TEMPLATES_DIR = 'poker/ai/rta/templates/ignition_cards'

# Tesseract OCR config tuned for Ignition's font (white text on dark green)
TESSERACT_CONFIG = '--psm 7 -c tessedit_char_whitelist=$0123456789.,'


# Specific Ignition behaviors to mimic (avoid bot-like patterns)
IGNITION_HUMAN_TIMING = {
    'min_decision_seconds': 1.5,   # never decide faster than this
    'max_decision_seconds': 25.0,  # max think time
    'preflop_call_avg': 3.0,       # average for simple preflop calls
    'preflop_call_stdev': 1.5,
    'postflop_decision_avg': 8.0,  # postflop should look more human
    'postflop_decision_stdev': 5.0,
    'tank_chance': 0.05,           # 5% chance to "tank" 15-25 sec
}
