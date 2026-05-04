#!/usr/bin/env python3
"""Play heads-up NL Hold'em against the bot in your terminal.

Usage:
    python scripts/play_vs_bot.py
    python scripts/play_vs_bot.py --strategy adaptive
    python scripts/play_vs_bot.py --strategy heuristic --hands 50
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, '.')

from poker.ai.engine.action import Action, ActionType
from poker.ai.engine.card import Card
from poker.ai.engine.game_state import GameState, Street
from poker.ai.engine.table import Table
from poker.ai.strategy.adaptive import AdaptiveStrategy
from poker.ai.strategy.base import BaseStrategy
from poker.ai.strategy.heuristic import HeuristicStrategy


SUITS = {'c': '♣', 'd': '♦', 'h': '♥', 's': '♠'}


def card_str(c: int) -> str:
    card = Card.from_int(c)
    rank_chars = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                  9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    suit_chars = {0: '♣', 1: '♦', 2: '♥', 3: '♠'}
    return f"{rank_chars[card.rank]}{suit_chars[card.suit]}"


class HumanPlayer(BaseStrategy):
    """Reads action from terminal input."""

    def __init__(self, name: str = "You") -> None:
        self.name = name

    def choose_action(self, state: GameState, legal_actions: list[ActionType]) -> Action:
        p = state.current_player
        hole = state.hole_cards[p]
        max_bet = max(state.current_bets) if state.current_bets else 0
        my_bet = state.current_bets[p] if state.current_bets else 0
        to_call = max_bet - my_bet
        my_stack = state.stacks[p]

        # Display state
        print()
        print("=" * 50)
        print(f"  Street: {state.street.name}")
        if state.board:
            print(f"  Board: {' '.join(card_str(c) for c in state.board)}")
        else:
            print(f"  Board: (preflop)")
        print(f"  Pot: {state.pot}")
        print(f"  Your hand: {' '.join(card_str(c) for c in hole)}")
        print(f"  Your stack: {my_stack}    Opponent stack: {state.stacks[1-p]}")
        if to_call > 0:
            print(f"  Bet to call: {to_call}")
        else:
            print(f"  No bet to call")
        legal_str = [a.name for a in legal_actions]
        print(f"  Legal actions: {', '.join(legal_str)}")
        print()

        while True:
            raw = input("  Your action [f/c/k/b<amount>/r<amount>/a]: ").strip().lower()
            if not raw:
                continue
            if raw in ('q', 'quit'):
                print("Goodbye.")
                sys.exit(0)
            try:
                if raw == 'f':
                    if ActionType.FOLD not in legal_actions:
                        print("  Can't fold (no bet to face)")
                        continue
                    return Action(ActionType.FOLD, 0, p)
                if raw == 'k':
                    if ActionType.CHECK not in legal_actions:
                        print("  Can't check (must call or fold)")
                        continue
                    return Action(ActionType.CHECK, 0, p)
                if raw == 'c':
                    if ActionType.CALL not in legal_actions:
                        print("  Nothing to call")
                        continue
                    return Action(ActionType.CALL, to_call, p)
                if raw == 'a':
                    if ActionType.ALL_IN not in legal_actions:
                        print("  Can't all-in")
                        continue
                    return Action(ActionType.ALL_IN, my_stack, p)
                if raw[0] == 'b':
                    amount = int(raw[1:])
                    if ActionType.BET not in legal_actions:
                        print("  Can't bet (must raise instead)")
                        continue
                    return Action(ActionType.BET, amount, p)
                if raw[0] == 'r':
                    amount = int(raw[1:])
                    if ActionType.RAISE not in legal_actions:
                        print("  Can't raise (no bet to raise)")
                        continue
                    return Action(ActionType.RAISE, amount, p)
                print(f"  Unknown command: {raw!r}")
            except (ValueError, IndexError):
                print(f"  Bad input: {raw!r}. Try f/c/k/b50/r100/a/q")


def build_bot(name: str):
    if name == 'heuristic':
        return HeuristicStrategy(num_simulations=200)
    if name == 'adaptive':
        return AdaptiveStrategy(opponent_name='Human', verbose_switching=True)
    raise ValueError(f"Unknown strategy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='adaptive', choices=['heuristic', 'adaptive'])
    parser.add_argument('--hands', type=int, default=30)
    parser.add_argument('--stack', type=int, default=200)
    parser.add_argument('--bb', type=int, default=2)
    args = parser.parse_args()

    print(f"Playing heads-up vs {args.strategy} bot")
    print(f"Stack: {args.stack} chips, Blinds: {args.bb//2}/{args.bb}")
    print(f"Commands: f=fold c=call k=check b<n>=bet n r<n>=raise n a=all-in q=quit")
    print()

    you = HumanPlayer("You")
    bot = build_bot(args.strategy)
    table = Table([you, bot], small_blind=args.bb // 2, big_blind=args.bb,
                  starting_stack=args.stack)

    starting_chips = args.stack
    for hand_num in range(args.hands):
        print(f"\n========== Hand {hand_num+1} ==========")
        button_label = "You" if table.button == 0 else "Bot"
        print(f"Button: {button_label}")
        try:
            deltas = table.play_hand()
        except KeyboardInterrupt:
            break
        you_delta = deltas[0]
        bot_delta = deltas[1]
        print(f"\n  Hand result: You {you_delta:+d} chips, Bot {bot_delta:+d} chips")
        print(f"  Stacks: You={table.stacks[0]}    Bot={table.stacks[1]}")

        # Reset stacks if anyone busted
        if table.stacks[0] <= 0 or table.stacks[1] <= 0:
            print("  Someone busted, resetting stacks.")
            table.stacks = [starting_chips, starting_chips]

    you_total = table.stacks[0] - starting_chips
    print()
    print(f"After {hand_num+1} hands: You {you_total:+d} chips")
    if you_total > 0:
        print("  ✓ You won!")
    else:
        print("  ✗ Bot won.")


if __name__ == '__main__':
    main()
