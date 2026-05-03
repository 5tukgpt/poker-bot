from __future__ import annotations

import random
from enum import IntEnum
from typing import NamedTuple


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


RANK_SYMBOLS = {
    2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
    9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A',
}

SUIT_SYMBOLS = {0: 'c', 1: 'd', 2: 'h', 3: 's'}


class Card(NamedTuple):
    rank: Rank
    suit: Suit

    def to_int(self) -> int:
        return (self.rank - 2) * 4 + self.suit

    @staticmethod
    def from_int(i: int) -> Card:
        return Card(Rank((i // 4) + 2), Suit(i % 4))

    def __str__(self) -> str:
        return RANK_SYMBOLS[self.rank] + SUIT_SYMBOLS[self.suit]

    @staticmethod
    def from_str(s: str) -> Card:
        rank_map = {v: k for k, v in RANK_SYMBOLS.items()}
        suit_map = {v: k for k, v in SUIT_SYMBOLS.items()}
        return Card(Rank(rank_map[s[0]]), Suit(suit_map[s[1]]))


class Deck:
    def __init__(self) -> None:
        self._cards: list[int] = list(range(52))
        self._index = 0

    def shuffle(self) -> None:
        random.shuffle(self._cards)
        self._index = 0

    def deal(self, n: int = 1) -> list[int]:
        cards = self._cards[self._index:self._index + n]
        self._index += n
        return cards

    @property
    def remaining(self) -> int:
        return 52 - self._index
