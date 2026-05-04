"""RTA daemon: capture → OCR → strategy → display recommendation.

Runs on Computer 2 (the analysis machine). Reads frames from the capture card,
parses the table state, asks our strategy what to do, and displays the
recommendation in big readable text.

Usage on Computer 2:
    python -m poker.ai.rta.advisor

Optional flags:
    --strategy adaptive    # which strategy to use (heuristic/dqn/adaptive/book)
    --device 0             # capture card device index
    --bb-dollars 0.02      # big blind in dollars (defines chip scale)
    --test-frame path.png  # process a single image instead of live capture
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poker.ai.engine.game_state import GameState

sys.path.insert(0, '.')

from poker.ai.rta.capture import CaptureSource
from poker.ai.rta.ocr import IgnitionOCR, state_dict_to_gamestate
from poker.ai.strategy.base import PokerStrategy


class TerminalDisplay:
    """Prints recommendations to terminal. Refreshes on each frame."""

    def __init__(self) -> None:
        self.last_state_hash: int = 0

    def show(self, state: 'GameState | None', action: str | None,
              equity: float | None, pot_odds: float | None) -> None:
        # Only update when state changes
        h = self._hash_state(state)
        if h == self.last_state_hash:
            return
        self.last_state_hash = h

        # Clear screen and print header
        print("\033[H\033[J", end='')  # ANSI clear screen
        print("=" * 60)
        print("              POKER RTA ADVISOR")
        print("=" * 60)

        if state is None:
            print("\n  (waiting for table state...)\n")
            return

        # Display state
        from poker.ai.engine.card import Card
        rank_chars = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                      9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        suit_chars = {0: '♣', 1: '♦', 2: '♥', 3: '♠'}
        def card_str(c: int) -> str:
            card = Card.from_int(c)
            return f"{rank_chars[card.rank]}{suit_chars[card.suit]}"

        hero_hole = state.hole_cards[0] if state.hole_cards and state.hole_cards[0] else []
        print(f"\n  YOUR HAND: {' '.join(card_str(c) for c in hero_hole) or '(unknown)'}")
        print(f"  BOARD:     {' '.join(card_str(c) for c in state.board) or '(preflop)'}")
        print(f"  STREET:    {state.street.name}")
        print(f"  POT:       ${state.pot / 100:.2f}")  # convert chips back to dollars
        print(f"  STACK:     ${state.stacks[0] / 100:.2f}")
        if state.current_bets:
            max_bet = max(state.current_bets)
            my_bet = state.current_bets[0]
            to_call = max_bet - my_bet
            if to_call > 0:
                print(f"  TO CALL:   ${to_call / 100:.2f}")

        if equity is not None:
            print(f"\n  EQUITY:    {equity*100:.1f}%")
        if pot_odds is not None:
            print(f"  POT ODDS:  {pot_odds*100:.1f}%")

        if action:
            # Big readable recommendation
            print()
            print("  +" + "-" * 56 + "+")
            print("  |" + f"  {action}".ljust(56) + "|")
            print("  +" + "-" * 56 + "+")

        print()

    def _hash_state(self, state: 'GameState | None') -> int:
        if state is None:
            return 0
        return hash((
            tuple(state.hole_cards[0]) if state.hole_cards else (),
            tuple(state.board),
            state.pot,
            tuple(state.current_bets) if state.current_bets else (),
        ))


def build_strategy(name: str) -> PokerStrategy:
    if name == 'heuristic':
        from poker.ai.strategy.heuristic import HeuristicStrategy
        return HeuristicStrategy(num_simulations=300)
    if name == 'adaptive':
        from poker.ai.strategy.adaptive import AdaptiveStrategy
        return AdaptiveStrategy(opponent_name='Live opponent', verbose_switching=False)
    if name == 'dqn':
        from poker.ai.strategy.dqn import DQNAgent
        agent = DQNAgent(training=False)
        agent.load('poker/ai/models/dqn_weights.npz')
        return agent
    raise ValueError(f"Unknown strategy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='heuristic', choices=['heuristic', 'adaptive', 'dqn'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--test-frame', help='Process a single PNG/JPG instead of live capture')
    parser.add_argument('--bb-dollars', type=float, default=0.02)
    args = parser.parse_args()

    print(f"Initializing RTA advisor (strategy={args.strategy})")
    strategy = build_strategy(args.strategy)
    ocr = IgnitionOCR()
    display = TerminalDisplay()

    if args.test_frame:
        # Single-frame test mode
        try:
            import cv2
        except ImportError:
            print("opencv-python required for test mode")
            sys.exit(1)
        frame = cv2.imread(args.test_frame)
        if frame is None:
            print(f"Could not read {args.test_frame}")
            sys.exit(1)
        process_frame(frame, ocr, strategy, display)
        return

    # Live mode
    cap = CaptureSource(device_index=args.device, fps_target=2)
    print(f"Capturing from device {args.device} at 2 FPS...")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            frame = cap.read_frame()
            if frame is None:
                continue
            process_frame(frame, ocr, strategy, display)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


def process_frame(frame, ocr: IgnitionOCR, strategy: PokerStrategy, display: TerminalDisplay) -> None:
    """Parse one frame and show recommendation."""
    state_dict = ocr.parse_frame(frame)
    state = state_dict_to_gamestate(state_dict)
    if state is None:
        display.show(None, None, None, None)
        return

    legal = state.legal_actions()
    if not legal:
        display.show(state, "(no actions available)", None, None)
        return

    try:
        action = strategy.choose_action(state, legal)
        action_text = format_action(action, state)
    except Exception as e:
        action_text = f"(strategy error: {e})"

    # Compute equity for display
    equity = None
    pot_odds = None
    try:
        from poker.ai.strategy.equity import monte_carlo_equity
        if state.hole_cards[0] and len(state.hole_cards[0]) == 2:
            num_opp = max(1, state.num_players - 1)
            equity = monte_carlo_equity(state.hole_cards[0], state.board, num_opp, 200)
            max_bet = max(state.current_bets) if state.current_bets else 0
            my_bet = state.current_bets[0] if state.current_bets else 0
            to_call = max_bet - my_bet
            if to_call > 0:
                pot_odds = to_call / (state.pot + to_call)
    except Exception:
        pass

    display.show(state, action_text, equity, pot_odds)


def format_action(action, state) -> str:
    """Convert Action object to display string."""
    from poker.ai.engine.action import ActionType
    if action.type == ActionType.FOLD:
        return "FOLD"
    if action.type == ActionType.CHECK:
        return "CHECK"
    if action.type == ActionType.CALL:
        return f"CALL ${action.amount / 100:.2f}"
    if action.type == ActionType.BET:
        return f"BET ${action.amount / 100:.2f}"
    if action.type == ActionType.RAISE:
        return f"RAISE TO ${action.amount / 100:.2f}"
    if action.type == ActionType.ALL_IN:
        return f"ALL IN (${state.stacks[0] / 100:.2f})"
    return str(action)


if __name__ == '__main__':
    main()
