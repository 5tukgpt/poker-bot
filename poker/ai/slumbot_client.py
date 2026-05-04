"""Slumbot API client. Test our strategies against the public benchmark.

Slumbot is a heads-up NL Hold'em bot maintained by Eric Jackson
(https://www.slumbot.com). Free public API, JSON over HTTP.

Game settings (fixed by Slumbot):
- Heads-up NLHE
- Blinds: 50 / 100
- Stack: 20,000 (200 BB)
- Stacks reset each hand

Action notation:
- 'k' = check
- 'c' = call
- 'bX' = bet to X (e.g., 'b200' = bet to 200 total this street)
- 'f' = fold
- '/' = street transition

Reference: https://github.com/Gongsta/Poker-AI/blob/main/slumbot/slumbot_api.py
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any

from .adapter import card_str_to_int
from .engine.action import Action, ActionType
from .engine.card import Card
from .engine.game_state import GameState, Street
from .strategy.base import PokerStrategy


HOST = "slumbot.com"
NUM_STREETS = 4
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000


class SlumbotError(Exception):
    pass


def _post(endpoint: str, body: dict) -> dict:
    """POST JSON to slumbot.com endpoint. Returns parsed JSON response."""
    url = f"https://{HOST}/api/{endpoint}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise SlumbotError(f"HTTP {e.code}: {e.read().decode()}")
    except urllib.error.URLError as e:
        raise SlumbotError(f"Network error: {e}")


def new_hand(token: str | None = None) -> dict:
    body = {"token": token} if token else {}
    return _post("new_hand", body)


def act(token: str, action_str: str) -> dict:
    return _post("act", {"token": token, "incr": action_str})


def parse_slumbot_action(action: str) -> dict:
    """Parse the action string from Slumbot into game state.

    Returns dict with:
      st (street: 0-3)
      pos (whose turn next: 0 or 1, -1 if hand over)
      street_last_bet_to (chips bet TO this street, not increment)
      total_last_bet_to (chips put in this hand TO date)
      last_bet_size (size of most recent bet/raise as increment)
      last_bettor (who bet last, -1 if no bet)
    """
    st = 0
    street_last_bet_to = BIG_BLIND
    total_last_bet_to = BIG_BLIND
    last_bet_size = BIG_BLIND - SMALL_BLIND
    last_bettor = 0
    sz = len(action)
    pos = 1  # BB acts after SB preflop (position 0 = SB = button)

    if sz == 0:
        return {
            "st": st, "pos": pos,
            "street_last_bet_to": street_last_bet_to,
            "total_last_bet_to": total_last_bet_to,
            "last_bet_size": last_bet_size,
            "last_bettor": last_bettor,
        }

    check_or_call_ends_street = False
    i = 0
    while i < sz:
        c = action[i]
        i += 1
        if c == "k":
            if check_or_call_ends_street:
                if i < sz and action[i] == "/":
                    i += 1
                if st == NUM_STREETS - 1:
                    pos = -1
                else:
                    pos = 0
                    st += 1
                street_last_bet_to = 0
                check_or_call_ends_street = False
            else:
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
        elif c == "c":
            if check_or_call_ends_street:
                if i < sz and action[i] == "/":
                    i += 1
                if st == NUM_STREETS - 1:
                    pos = -1
                else:
                    pos = 0
                    st += 1
                street_last_bet_to = 0
                check_or_call_ends_street = False
            else:
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
            last_bet_size = 0
            last_bettor = -1
        elif c == "f":
            pos = -1
            return {
                "st": st, "pos": pos,
                "street_last_bet_to": street_last_bet_to,
                "total_last_bet_to": total_last_bet_to,
                "last_bet_size": last_bet_size,
                "last_bettor": last_bettor,
            }
        elif c == "b":
            j = i
            while i < sz and "0" <= action[i] <= "9":
                i += 1
            new_street_last_bet_to = int(action[j:i])
            new_last_bet_size = new_street_last_bet_to - street_last_bet_to
            last_bet_size = new_last_bet_size
            street_last_bet_to = new_street_last_bet_to
            total_last_bet_to += last_bet_size
            last_bettor = pos
            pos = (pos + 1) % 2
            check_or_call_ends_street = True
        elif c == "/":
            continue

    return {
        "st": st, "pos": pos,
        "street_last_bet_to": street_last_bet_to,
        "total_last_bet_to": total_last_bet_to,
        "last_bet_size": last_bet_size,
        "last_bettor": last_bettor,
    }


def parse_action_history(action_str: str, our_pos: int) -> list[Action]:
    """Parse Slumbot's action string into a list of Action objects.

    Slumbot positions: 0 = button/SB acts first preflop, 1 = BB.
    Each Action gets a 'street' index (0=preflop, 1=flop, 2=turn, 3=river)
    so observers know what street the action occurred on.
    """
    actions: list[Action] = []
    sz = len(action_str)
    i = 0
    pos = 0
    street = 0
    street_last_bet_to = BIG_BLIND

    while i < sz:
        c = action_str[i]
        i += 1
        if c == "/":
            street += 1
            pos = 1  # BB acts first postflop in HU
            street_last_bet_to = 0
            continue
        if c == "k":
            actions.append(Action(ActionType.CHECK, 0, pos, street=street))
            pos = (pos + 1) % 2
        elif c == "c":
            call_amount = street_last_bet_to
            actions.append(Action(ActionType.CALL, call_amount, pos, street=street))
            pos = (pos + 1) % 2
        elif c == "f":
            actions.append(Action(ActionType.FOLD, 0, pos, street=street))
            break
        elif c == "b":
            j = i
            while i < sz and "0" <= action_str[i] <= "9":
                i += 1
            new_total = int(action_str[j:i])
            increment = new_total - street_last_bet_to
            if new_total >= STACK_SIZE:
                actions.append(Action(ActionType.ALL_IN, increment, pos, street=street))
            elif street_last_bet_to == 0:
                actions.append(Action(ActionType.BET, increment, pos, street=street))
            else:
                actions.append(Action(ActionType.RAISE, increment, pos, street=street))
            street_last_bet_to = new_total
            pos = (pos + 1) % 2

    return actions


def slumbot_state_to_gamestate(
    response: dict, action: str, our_pos: int,
) -> GameState | None:
    """Build a GameState from Slumbot's response. Returns None if hand over."""
    parsed = parse_slumbot_action(action)
    if parsed.get("pos", -1) == -1:
        return None
    if parsed["pos"] != our_pos:
        return None  # not our turn

    hole_strs = response.get("hole_cards", [])
    board_strs = response.get("board", [])
    hole = [card_str_to_int(c) for c in hole_strs]
    board = [card_str_to_int(c) for c in board_strs]

    street = {0: Street.PREFLOP, 1: Street.FLOP, 2: Street.TURN, 3: Street.RIVER}[parsed["st"]]

    # Slumbot uses position 0 = button (SB), 1 = BB.
    # In our engine, we use button index for who has the button.
    # We're at our_pos; opponent at 1 - our_pos.
    our_total_in = parsed["total_last_bet_to"] if parsed["last_bettor"] != our_pos else 0
    opp_total_in = parsed["total_last_bet_to"] if parsed["last_bettor"] == our_pos else parsed["total_last_bet_to"]

    # Simpler: pot = total chips put in by both
    # If last bettor was opponent, their bet equals total_last_bet_to.
    # Our committed equals total_last_bet_to - last_bet_size (we matched up to before their last action).
    if parsed["last_bettor"] == our_pos:
        our_committed = parsed["total_last_bet_to"]
        opp_committed = parsed["total_last_bet_to"] - parsed["last_bet_size"]
    else:
        opp_committed = parsed["total_last_bet_to"]
        our_committed = parsed["total_last_bet_to"] - parsed["last_bet_size"]

    pot = our_committed + opp_committed
    our_stack = STACK_SIZE - our_committed
    opp_stack = STACK_SIZE - opp_committed

    # Build state with us as player 0 always for strategy logic
    # (our strategies look at state.current_player; we pass 0)
    state = GameState(
        num_players=2,
        stacks=[our_stack, opp_stack],
        pot=pot,
        board=board,
        hole_cards=[hole, []],
        street=street,
        current_player=0,
        button=0 if our_pos == 0 else 1,
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND,
        current_bets=[
            our_committed if street != Street.PREFLOP else (parsed["street_last_bet_to"] if parsed["last_bettor"] == our_pos else parsed["street_last_bet_to"] - parsed["last_bet_size"]),
            opp_committed if street != Street.PREFLOP else (parsed["street_last_bet_to"] if parsed["last_bettor"] != our_pos else parsed["street_last_bet_to"] - parsed["last_bet_size"]),
        ],
        action_history=_remap_history(parse_action_history(action, our_pos), our_pos),
        folded=[False, False],
        all_in=[our_stack == 0, opp_stack == 0],
    )
    return state


def _remap_history(actions: list[Action], our_pos: int) -> list[Action]:
    """Remap player_idx so we are always player 0 in our internal state."""
    if our_pos == 0:
        return actions  # already aligned
    return [Action(a.type, a.amount, 1 - a.player_idx, street=a.street) for a in actions]


def our_action_to_slumbot(action: Action, parsed_state: dict) -> str:
    """Convert our Action object to Slumbot's action notation."""
    if action.type == ActionType.FOLD:
        return "f"
    if action.type == ActionType.CHECK:
        return "k"
    if action.type == ActionType.CALL:
        return "c"
    if action.type in (ActionType.BET, ActionType.RAISE):
        # Slumbot uses TOTAL bet TO this street, not increment
        # If we're betting amount X (increment), total = current_street_bet + X
        total = parsed_state["street_last_bet_to"] + action.amount
        # Clamp to stack
        max_to = STACK_SIZE
        total = min(total, max_to)
        # Min raise: must be at least last bet size more
        min_total = parsed_state["street_last_bet_to"] + max(parsed_state["last_bet_size"], BIG_BLIND)
        total = max(total, min_total)
        total = min(total, max_to)
        return f"b{total}"
    if action.type == ActionType.ALL_IN:
        return f"b{STACK_SIZE}"
    return "f"


def play_hand(strategy: PokerStrategy, token: str | None) -> tuple[int, str]:
    """Play one hand against Slumbot. Returns (winnings_in_chips, new_token)."""
    response = new_hand(token)
    if "error_msg" in response:
        raise SlumbotError(response["error_msg"])
    token = response.get("token", token)
    last_state: GameState | None = None

    while True:
        action_str = response.get("action", "")
        our_pos = response.get("client_pos", 0)
        winnings = response.get("winnings")
        if winnings is not None:
            _notify_end_of_hand(strategy, response, action_str, our_pos, winnings, last_state)
            return winnings, token

        parsed = parse_slumbot_action(action_str)
        if parsed.get("pos") == -1:
            w = response.get("winnings", 0)
            _notify_end_of_hand(strategy, response, action_str, our_pos, w, last_state)
            return w, token

        if parsed["pos"] != our_pos:
            w = response.get("winnings", 0)
            _notify_end_of_hand(strategy, response, action_str, our_pos, w, last_state)
            return w, token

        state = slumbot_state_to_gamestate(response, action_str, our_pos)
        if state is None:
            w = response.get("winnings", 0)
            _notify_end_of_hand(strategy, response, action_str, our_pos, w, last_state)
            return w, token

        last_state = state
        legal = state.legal_actions()
        if not legal:
            w = response.get("winnings", 0)
            _notify_end_of_hand(strategy, response, action_str, our_pos, w, last_state)
            return w, token

        action = strategy.choose_action(state, legal)
        slumbot_action = our_action_to_slumbot(action, parsed)

        try:
            response = act(token, slumbot_action)
            token = response.get("token", token)
        except SlumbotError as e:
            print(f"  ! Slumbot rejected action {slumbot_action!r}: {e}")
            response = act(token, "f")
            token = response.get("token", token)
            w = response.get("winnings", 0)
            _notify_end_of_hand(strategy, response, "", our_pos, w, last_state)
            return w, token


def _notify_end_of_hand(
    strategy: PokerStrategy,
    response: dict,
    action_str: str,
    our_pos: int,
    winnings: int,
    last_state: GameState | None,
) -> None:
    """Build a final-state and call notify_result so strategies can commit per-hand stats."""
    if last_state is None:
        # No state was ever built (rare — Slumbot acted and we folded immediately?)
        # Build minimal state for stats tracking
        from .engine.game_state import Street
        hole_strs = response.get("hole_cards", [])
        board_strs = response.get("board", [])
        hole = [card_str_to_int(c) for c in hole_strs] if hole_strs else []
        board = [card_str_to_int(c) for c in board_strs] if board_strs else []
        last_state = GameState(
            num_players=2,
            stacks=[STACK_SIZE, STACK_SIZE],
            pot=0, board=board,
            hole_cards=[hole, []],
            street=Street.PREFLOP, current_player=0,
            button=0 if our_pos == 0 else 1,
            small_blind=SMALL_BLIND, big_blind=BIG_BLIND,
            current_bets=[0, 0], action_history=[],
            folded=[False, False], all_in=[False, False],
        )

    # Update action_history with the final action sequence so strategies can observe
    if action_str:
        try:
            final_actions = parse_action_history(action_str, our_pos)
            last_state = GameState(
                num_players=last_state.num_players,
                stacks=last_state.stacks,
                pot=last_state.pot,
                board=last_state.board,
                hole_cards=last_state.hole_cards,
                street=last_state.street,
                current_player=last_state.current_player,
                button=last_state.button,
                small_blind=last_state.small_blind,
                big_blind=last_state.big_blind,
                current_bets=last_state.current_bets,
                action_history=_remap_history(final_actions, our_pos),
                folded=last_state.folded,
                all_in=last_state.all_in,
            )
        except Exception:
            pass

    try:
        strategy.notify_result(last_state, winnings)
    except Exception as e:
        print(f"  ! notify_result raised: {e}")


def play_match(strategy: PokerStrategy, num_hands: int, verbose: bool = True) -> dict:
    """Play N hands. Returns stats dict with total winnings, BB/100, etc."""
    token = None
    total_winnings = 0
    hands_played = 0
    errors = 0

    for i in range(num_hands):
        try:
            winnings, token = play_hand(strategy, token)
            total_winnings += winnings
            hands_played += 1
            if verbose and (i + 1) % 25 == 0:
                bb_per_100 = (total_winnings / BIG_BLIND / hands_played) * 100
                print(f"  Hand {i+1}/{num_hands}: total={total_winnings:+d} chips, BB/100={bb_per_100:+.1f}")
        except SlumbotError as e:
            errors += 1
            print(f"  Error on hand {i+1}: {e}")
            if errors > 5:
                print("  Too many errors, stopping.")
                break
        except Exception as e:
            errors += 1
            print(f"  Unexpected error on hand {i+1}: {e}")
            if errors > 5:
                break

    bb_per_100 = (total_winnings / BIG_BLIND / max(hands_played, 1)) * 100
    return {
        "hands_played": hands_played,
        "total_winnings": total_winnings,
        "bb_per_100": bb_per_100,
        "errors": errors,
    }
