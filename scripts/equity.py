#!/usr/bin/env python3
"""Live poker equity calculator. Use this WHILE playing.

Card format: rank + suit (lowercase)
  Ranks: 2 3 4 5 6 7 8 9 t j q k a
  Suits: c (clubs), d (diamonds), h (hearts), s (spades)
  Example: ah = ace of hearts, td = ten of diamonds, 2c = two of clubs

USAGE:

  Quick equity:
    python scripts/equity.py ah ks                       # AK preflop, 1 opp
    python scripts/equity.py ah ks -- qh jh 2c           # AK on Q-J-2 flop
    python scripts/equity.py ah ks -- qh jh 2c -o 3      # vs 3 opponents

  With pot odds & recommendation:
    python scripts/equity.py ah ks -- qh jh 2c -p 30 -c 10
    (pot=30, bet to call=10 → "CALL" or "FOLD")

  Interactive mode (faster while playing):
    python scripts/equity.py -i

  Faster (fewer simulations, ~0.1s):
    python scripts/equity.py ah ks -n 200
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, '.')

from poker.ai.engine.card import Card
from poker.ai.strategy.equity import monte_carlo_equity


def parse_card(s: str) -> int:
    """Parse a card string like 'ah' or 'Th' into our int encoding."""
    s = s.strip().lower()
    if len(s) != 2:
        raise ValueError(f"Bad card: {s!r} (need 2 chars, e.g. 'ah')")
    rank_char, suit_char = s[0], s[1]
    # Normalize rank (we use uppercase in Card.from_str)
    rank_map = {'2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
                '8': '8', '9': '9', 't': 'T', 'j': 'J', 'q': 'Q', 'k': 'K', 'a': 'A'}
    if rank_char not in rank_map:
        raise ValueError(f"Bad rank: {rank_char!r} (use 2-9, t, j, q, k, a)")
    if suit_char not in 'cdhs':
        raise ValueError(f"Bad suit: {suit_char!r} (use c, d, h, s)")
    return Card.from_str(rank_map[rank_char] + suit_char).to_int()


def parse_cards(strings: list[str]) -> list[int]:
    return [parse_card(s) for s in strings]


def equity_check(
    hole_strs: list[str],
    board_strs: list[str],
    num_opponents: int,
    num_sims: int,
    pot: float | None,
    to_call: float | None,
) -> None:
    if len(hole_strs) != 2:
        print(f"Error: need exactly 2 hole cards, got {len(hole_strs)}", file=sys.stderr)
        sys.exit(1)
    if len(board_strs) not in (0, 3, 4, 5):
        print(f"Error: board must be 0/3/4/5 cards, got {len(board_strs)}", file=sys.stderr)
        sys.exit(1)

    hole = parse_cards(hole_strs)
    board = parse_cards(board_strs)

    # Compute equity
    t = time.time()
    equity = monte_carlo_equity(hole, board, num_opponents, num_sims)
    elapsed = time.time() - t

    # Display
    street_name = {0: 'PREFLOP', 3: 'FLOP', 4: 'TURN', 5: 'RIVER'}[len(board)]
    hole_disp = ' '.join(str(Card.from_int(c)) for c in hole)
    board_disp = ' '.join(str(Card.from_int(c)) for c in board) if board else '(none)'

    print(f"  Hand: {hole_disp}    Board: {board_disp}    Street: {street_name}")
    print(f"  Equity: {equity*100:.1f}% vs {num_opponents} opp ({num_sims} sims, {elapsed*1000:.0f}ms)")

    if pot is not None and to_call is not None and to_call > 0:
        pot_odds = to_call / (pot + to_call)
        edge = equity - pot_odds
        print(f"  Pot odds: {pot_odds*100:.1f}% (need {pot_odds*100:.1f}% to break even)")

        if edge > 0.10:
            rec = f"RAISE — you're a clear favorite (+{edge*100:.0f}% edge)"
        elif edge > 0.02:
            rec = f"CALL — profitable (+{edge*100:.0f}% edge)"
        elif edge > -0.02:
            rec = f"COIN FLIP — call is roughly break-even"
        else:
            rec = f"FOLD — you're behind ({edge*100:.0f}% edge)"
        print(f"  → {rec}")
    elif pot is not None:
        print(f"  Pot: ${pot:.2f} (no bet to call)")
        if equity > 0.65:
            print(f"  → BET FOR VALUE — bet ${pot*0.67:.2f} (67% pot)")
        elif equity > 0.50:
            print(f"  → THIN VALUE — small bet ${pot*0.33:.2f} (33% pot)")
        else:
            print(f"  → CHECK — not strong enough to value bet")


def interactive() -> None:
    print("=== Poker Equity Helper ===")
    print("Card format: 'ah' = ace of hearts, 'td' = ten of diamonds, '2c' = two of clubs")
    print("Type 'q' to quit.\n")

    while True:
        try:
            hole_input = input("Your cards (e.g. 'ah ks'): ").strip().lower()
            if hole_input in ('q', 'quit', 'exit'):
                break
            hole_strs = hole_input.split()
            if len(hole_strs) != 2:
                print("  Need exactly 2 cards.\n")
                continue

            board_input = input("Board (blank for preflop, or '3 cards' / '4' / '5'): ").strip().lower()
            board_strs = board_input.split() if board_input else []

            opp_input = input("Opponents [1]: ").strip()
            num_opp = int(opp_input) if opp_input else 1

            pot_input = input("Pot size $ (or skip): ").strip()
            pot = float(pot_input) if pot_input else None

            call_input = input("Bet to call $ (or skip): ").strip()
            to_call = float(call_input) if call_input else None

            equity_check(hole_strs, board_strs, num_opp, 1000, pot, to_call)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
            break
        except Exception as e:
            print(f"  Error: {e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Poker equity calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('cards', nargs='*', help='Hole cards then "--" then board cards')
    parser.add_argument('-o', '--opponents', type=int, default=1, help='Number of opponents (default 1)')
    parser.add_argument('-n', '--sims', type=int, default=1000, help='Monte Carlo simulations (default 1000)')
    parser.add_argument('-p', '--pot', type=float, help='Current pot size (for pot odds)')
    parser.add_argument('-c', '--to-call', type=float, help='Bet amount you need to call')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive prompt mode')
    args = parser.parse_args()

    if args.interactive:
        interactive()
        return

    if not args.cards:
        parser.print_help()
        return

    # Split cards on '--' to separate hole from board
    if '--' in args.cards:
        idx = args.cards.index('--')
        hole_strs = args.cards[:idx]
        board_strs = args.cards[idx + 1:]
    else:
        # Heuristic: first 2 are hole, rest are board
        hole_strs = args.cards[:2]
        board_strs = args.cards[2:]

    equity_check(hole_strs, board_strs, args.opponents, args.sims, args.pot, args.to_call)


if __name__ == '__main__':
    main()
