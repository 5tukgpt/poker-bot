from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlayerStats:
    name: str
    big_blind: int = 2
    hands_played: int = 0
    total_profit: int = 0
    vpip_count: int = 0
    pfr_count: int = 0
    wins: int = 0

    @property
    def bb_per_100(self) -> float:
        if self.hands_played == 0:
            return 0.0
        bb_profit = self.total_profit / self.big_blind
        return (bb_profit / self.hands_played) * 100

    @property
    def vpip(self) -> float:
        if self.hands_played == 0:
            return 0.0
        return self.vpip_count / self.hands_played * 100

    @property
    def pfr(self) -> float:
        if self.hands_played == 0:
            return 0.0
        return self.pfr_count / self.hands_played * 100

    @property
    def win_rate(self) -> float:
        if self.hands_played == 0:
            return 0.0
        return self.wins / self.hands_played * 100

    def summary(self) -> str:
        return (
            f"{self.name}: "
            f"BB/100={self.bb_per_100:+.1f}  "
            f"Profit={self.total_profit:+d}  "
            f"VPIP={self.vpip:.1f}%  "
            f"PFR={self.pfr:.1f}%  "
            f"WinRate={self.win_rate:.1f}%  "
            f"Hands={self.hands_played}"
        )
