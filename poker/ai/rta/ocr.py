"""OCR pipeline: capture-card frame → poker GameState.

Uses OpenCV template matching for cards (very reliable when templates match)
and Tesseract OCR for numeric text (pot, stacks, bets).

Card recognition strategy:
1. Crop card region from frame
2. Compare against pre-captured templates of all 52 cards
3. Best-matching template = the card

For Ignition specifically, card sprites are consistent so template matching
gets >99% accuracy once you have good templates. To capture templates,
take a screenshot of one of each card from a real Ignition table and crop
to ~60x80 px each, save as `templates/ignition_cards/{rank}{suit}.png`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:
    HAS_CV = False
    cv2 = None
    np = None

try:
    import pytesseract
    HAS_TESS = True
except ImportError:
    HAS_TESS = False
    pytesseract = None

from .ignition_config import (
    CARD_TEMPLATES_DIR,
    CardRegion,
    IgnitionTableLayout,
    TESSERACT_CONFIG,
    TextRegion,
)


RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['c', 'd', 'h', 's']


class CardRecognizer:
    """Identifies cards via template matching."""

    def __init__(self, templates_dir: str = CARD_TEMPLATES_DIR) -> None:
        if not HAS_CV:
            raise ImportError("opencv-python required")
        self.templates: dict[str, np.ndarray] = {}
        self._load_templates(templates_dir)

    def _load_templates(self, templates_dir: str) -> None:
        if not Path(templates_dir).exists():
            print(f"WARN: card templates dir not found: {templates_dir}")
            return
        for rank in RANKS:
            for suit in SUITS:
                code = rank + suit
                path = Path(templates_dir) / f"{code}.png"
                if path.exists():
                    self.templates[code] = cv2.imread(str(path), cv2.IMREAD_COLOR)

    def recognize(self, card_image: 'np.ndarray', threshold: float = 0.85) -> str | None:
        """Return card code (e.g., 'Ah') or None if no match exceeds threshold."""
        if not self.templates:
            return None
        best_code = None
        best_score = 0.0
        for code, template in self.templates.items():
            if template.shape != card_image.shape:
                resized = cv2.resize(template, (card_image.shape[1], card_image.shape[0]))
            else:
                resized = template
            result = cv2.matchTemplate(card_image, resized, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if score > best_score:
                best_score = score
                best_code = code
        if best_score >= threshold:
            return best_code
        return None


class TextReader:
    """Reads numeric values (pot, stacks, bets) using Tesseract."""

    def __init__(self) -> None:
        if not HAS_TESS:
            print("WARN: pytesseract not installed — text OCR disabled")

    def read_number(self, image: 'np.ndarray') -> float | None:
        """Extract a dollar amount like '$12.50' from an image. Returns None if unparseable."""
        if not HAS_TESS or not HAS_CV:
            return None
        # Preprocess: convert to grayscale, threshold for high contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config=TESSERACT_CONFIG)
        return self._parse_dollar_amount(text)

    @staticmethod
    def _parse_dollar_amount(text: str) -> float | None:
        # Strip $ and commas, keep digits + decimal
        cleaned = re.sub(r'[^\d.]', '', text)
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None


def crop(frame: 'np.ndarray', region) -> 'np.ndarray':
    """Crop a CardRegion or TextRegion from the frame."""
    if isinstance(region, (CardRegion, TextRegion)):
        return frame[region.y:region.y + region.height,
                     region.x:region.x + region.width]
    raise TypeError(f"Bad region type: {type(region)}")


class IgnitionOCR:
    """Top-level OCR pipeline: frame → table state dict."""

    def __init__(self, layout: IgnitionTableLayout | None = None) -> None:
        self.layout = layout or IgnitionTableLayout()
        self.card_recognizer = CardRecognizer()
        self.text_reader = TextReader()

    def parse_frame(self, frame: 'np.ndarray') -> dict:
        """Read all relevant fields from the frame. Returns dict ready for adapter."""
        layout = self.layout

        # Hole cards (hero only — opponents face down)
        hole_cards = []
        for region in (layout.hero_card1, layout.hero_card2):
            card = self.card_recognizer.recognize(crop(frame, region))
            if card:
                hole_cards.append(card)

        # Board cards (3 = flop, 4 = turn, 5 = river)
        board_cards = []
        for region in layout.board_cards:
            card = self.card_recognizer.recognize(crop(frame, region))
            if card:
                board_cards.append(card)

        # Pot
        pot = self.text_reader.read_number(crop(frame, layout.pot_text))

        # Stacks (per seat)
        stacks = []
        for region in layout.seat_stacks:
            stack = self.text_reader.read_number(crop(frame, region))
            stacks.append(stack)

        # Current bets (per seat)
        bets = []
        for region in layout.seat_bets:
            bet = self.text_reader.read_number(crop(frame, region))
            bets.append(bet if bet is not None else 0)

        # Determine game stage from board card count
        stage = {0: 'PreFlop', 3: 'Flop', 4: 'Turn', 5: 'River'}.get(len(board_cards), 'PreFlop')

        return {
            'hole_cards': hole_cards,
            'board_cards': board_cards,
            'pot': pot or 0,
            'stacks': stacks,
            'bets': bets,
            'game_stage': stage,
            'num_seats': len(stacks),
        }


def state_dict_to_gamestate(state_dict: dict, hero_seat: int = 0):
    """Convert OCR output dict → our internal GameState.

    For RTA we always treat hero as player 0 (current_player).
    """
    from poker.ai.engine.action import Action
    from poker.ai.engine.card import Card
    from poker.ai.engine.game_state import GameState, Street

    def card_str_to_int(s: str) -> int:
        rank_map = {'2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
                    '8': '8', '9': '9', 'T': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}
        return Card.from_str(rank_map[s[0]] + s[1].lower()).to_int()

    hole = [card_str_to_int(c) for c in state_dict['hole_cards']]
    board = [card_str_to_int(c) for c in state_dict['board_cards']]
    street = {'PreFlop': Street.PREFLOP, 'Flop': Street.FLOP,
              'Turn': Street.TURN, 'River': Street.RIVER}[state_dict['game_stage']]

    # Convert dollars to chips (treat $0.02 as 2 chips for our internal scale)
    # The actual scale depends on the stake level — TODO: pass bb_value as parameter
    bb_dollar = 0.02  # default to micro-stakes
    chip_scale = 2.0 / bb_dollar

    pot_chips = int(state_dict['pot'] * chip_scale) if state_dict['pot'] else 0
    stacks = [int((s or 0) * chip_scale) for s in state_dict['stacks']]
    bets = [int(b * chip_scale) for b in state_dict['bets']]

    num_players = len([s for s in stacks if s > 0])
    if num_players < 2:
        return None

    return GameState(
        num_players=num_players,
        stacks=stacks[:num_players],
        pot=pot_chips,
        board=board,
        hole_cards=[hole] + [[] for _ in range(num_players - 1)],
        street=street,
        current_player=hero_seat,
        button=0,  # TODO: detect from button position indicator
        small_blind=int(bb_dollar / 2 * chip_scale),
        big_blind=int(bb_dollar * chip_scale),
        current_bets=bets[:num_players],
        action_history=[],
        folded=[False] * num_players,
        all_in=[False] * num_players,
    )
