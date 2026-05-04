"""Equity-distribution clustering for CFR card abstraction.

Computes equity histograms (distribution of hand strength against random
opponents) and clusters them via k-means. Produces N buckets per street
that capture both hand strength AND draw potential — a flush draw and a
made pair can have similar EHS but different histograms.

Standard references: Johanson 2013 (PCA + k-means abstractions), Pluribus
supplementary materials (OCHS bucketing).

Usage:
    bucketer = EquityBucketer.precompute(
        clusters_per_street={'preflop': 169, 'flop': 50, 'turn': 50, 'river': 50},
        save_path='poker/ai/models/abstraction.json',
    )
    bucket = bucketer.bucket(hole, board, street)
"""

from __future__ import annotations

import json
import os
import random
import time
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from ..engine.card import Card, Rank
from ..engine.evaluator import evaluate_hand

if TYPE_CHECKING:
    from sklearn.cluster import KMeans

DEFAULT_HISTOGRAM_BINS = 10
DEFAULT_ROLLOUTS = 200


def compute_equity_histogram(
    hole: list[int],
    board: list[int],
    num_rollouts: int = DEFAULT_ROLLOUTS,
    num_bins: int = DEFAULT_HISTOGRAM_BINS,
) -> np.ndarray:
    """Estimate the distribution of equity for (hole, board) over random opponents.

    Returns a histogram (length num_bins) summing to 1, where bin i represents
    P(equity in [i/num_bins, (i+1)/num_bins]).
    """
    known = set(hole + board)
    remaining = [c for c in range(52) if c not in known]
    bins = np.zeros(num_bins, dtype=np.float32)

    for _ in range(num_rollouts):
        deck = remaining.copy()
        random.shuffle(deck)
        idx = 0
        sim_board = board.copy()
        while len(sim_board) < 5:
            sim_board.append(deck[idx])
            idx += 1
        opp = [deck[idx], deck[idx + 1]]

        my_rank = evaluate_hand(hole, sim_board)
        opp_rank = evaluate_hand(opp, sim_board)
        if my_rank < opp_rank:
            equity = 1.0
        elif my_rank == opp_rank:
            equity = 0.5
        else:
            equity = 0.0

        bin_idx = min(int(equity * num_bins), num_bins - 1)
        bins[bin_idx] += 1

    return bins / bins.sum()


def canonical_preflop_class(hole: list[int]) -> str:
    """Map a 2-card hole to one of 169 canonical hand classes (e.g. 'AA', 'AKs', 'AKo')."""
    c1, c2 = Card.from_int(hole[0]), Card.from_int(hole[1])
    r1, r2 = max(c1.rank, c2.rank), min(c1.rank, c2.rank)
    rank_chars = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                  9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    if r1 == r2:
        return rank_chars[r1] + rank_chars[r2]
    suffix = 's' if c1.suit == c2.suit else 'o'
    return rank_chars[r1] + rank_chars[r2] + suffix


def all_preflop_hands() -> list[list[int]]:
    """One representative hole-card combo for each of 169 canonical preflop classes."""
    seen: dict[str, list[int]] = {}
    for c1, c2 in combinations(range(52), 2):
        cls = canonical_preflop_class([c1, c2])
        if cls not in seen:
            seen[cls] = [c1, c2]
    return list(seen.values())


def sample_postflop_hands(num_samples: int, num_board_cards: int) -> list[tuple[list[int], list[int]]]:
    """Sample (hole, board) pairs uniformly at random."""
    samples = []
    for _ in range(num_samples):
        deck = list(range(52))
        random.shuffle(deck)
        hole = deck[:2]
        board = deck[2:2 + num_board_cards]
        samples.append((hole, board))
    return samples


class EquityBucketer:
    """Maps hands to clustered buckets per street."""

    STREETS = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}

    def __init__(self) -> None:
        # centroids[street_name] = ndarray of shape (n_clusters, num_bins)
        self.centroids: dict[str, np.ndarray] = {}
        # preflop_lookup[canonical_class] = bucket_idx (avoids recomputing histograms at runtime)
        self.preflop_lookup: dict[str, int] = {}
        self.num_bins = DEFAULT_HISTOGRAM_BINS
        # Memoize postflop bucket lookups: (hole_canonical, board_canonical) -> bucket_idx
        self._cache: dict[tuple, int] = {}
        # Use fewer rollouts at runtime; can be tuned
        self._runtime_rollouts = 30

    def bucket(self, hole: list[int], board: list[int], street: int) -> int:
        """Assign a bucket given hole + board + street."""
        if street == 0:
            cls = canonical_preflop_class(hole)
            return self.preflop_lookup.get(cls, 0)

        street_name = {1: 'flop', 2: 'turn', 3: 'river'}.get(street, 'river')
        if street_name not in self.centroids:
            return 0

        # Memoize: (sorted_hole, sorted_board) is the canonical key
        cache_key = (tuple(sorted(hole)), tuple(sorted(board)))
        if cache_key in self._cache:
            return self._cache[cache_key]

        hist = compute_equity_histogram(
            hole, board, num_rollouts=self._runtime_rollouts, num_bins=self.num_bins
        )
        bucket_idx = int(_nearest_centroid(hist, self.centroids[street_name]))
        self._cache[cache_key] = bucket_idx
        return bucket_idx

    def total_buckets(self) -> int:
        n = len(set(self.preflop_lookup.values())) if self.preflop_lookup else 0
        for centroids in self.centroids.values():
            n += centroids.shape[0]
        return n

    @classmethod
    def precompute(
        cls,
        clusters_per_street: dict[str, int],
        save_path: str | None = None,
        rollouts: int = DEFAULT_ROLLOUTS,
        postflop_samples: int = 5000,
        verbose: bool = True,
    ) -> 'EquityBucketer':
        """Build the bucketer by clustering equity histograms.

        Args:
            clusters_per_street: e.g. {'preflop': 50, 'flop': 50, 'turn': 50, 'river': 50}
            save_path: optional JSON path to save centroids
            rollouts: rollouts per histogram during precomputation
            postflop_samples: how many random hands to sample per postflop street for clustering
        """
        from sklearn.cluster import KMeans

        bucketer = cls()
        bucketer.num_bins = DEFAULT_HISTOGRAM_BINS

        # Preflop: enumerate all 169 canonical classes
        if verbose:
            print(f"Computing 169 preflop equity histograms...")
        t0 = time.time()
        preflop_hands = all_preflop_hands()
        preflop_classes = [canonical_preflop_class(h) for h in preflop_hands]
        preflop_hists = np.array([
            compute_equity_histogram(h, [], num_rollouts=rollouts, num_bins=bucketer.num_bins)
            for h in preflop_hands
        ])

        n_preflop = clusters_per_street.get('preflop', 50)
        if n_preflop >= 169:
            # Just use one bucket per canonical class
            for i, cls_name in enumerate(preflop_classes):
                bucketer.preflop_lookup[cls_name] = i
        else:
            km = KMeans(n_clusters=n_preflop, n_init=10, random_state=42).fit(preflop_hists)
            for cls_name, label in zip(preflop_classes, km.labels_):
                bucketer.preflop_lookup[cls_name] = int(label)
        if verbose:
            print(f"  Preflop done in {time.time()-t0:.1f}s, {len(set(bucketer.preflop_lookup.values()))} buckets")

        # Postflop: sample random hands and cluster
        for street_name, num_board_cards in [('flop', 3), ('turn', 4), ('river', 5)]:
            n_clusters = clusters_per_street.get(street_name, 50)
            if verbose:
                print(f"Sampling {postflop_samples} {street_name} hands...")
            t0 = time.time()
            samples = sample_postflop_hands(postflop_samples, num_board_cards)
            hists = np.array([
                compute_equity_histogram(hole, board, num_rollouts=rollouts, num_bins=bucketer.num_bins)
                for hole, board in samples
            ])
            if verbose:
                print(f"  Histograms in {time.time()-t0:.1f}s, clustering...")
            t0 = time.time()
            km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42).fit(hists)
            bucketer.centroids[street_name] = km.cluster_centers_
            if verbose:
                print(f"  {street_name}: {n_clusters} buckets in {time.time()-t0:.1f}s")

        if save_path:
            bucketer.save(save_path)
            if verbose:
                print(f"Saved to {save_path}")

        return bucketer

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'num_bins': self.num_bins,
            'preflop_lookup': self.preflop_lookup,
            'centroids': {k: v.tolist() for k, v in self.centroids.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'EquityBucketer':
        with open(path) as f:
            data = json.load(f)
        b = cls()
        b.num_bins = data['num_bins']
        b.preflop_lookup = data['preflop_lookup']
        b.centroids = {k: np.array(v) for k, v in data['centroids'].items()}
        return b


def _nearest_centroid(hist: np.ndarray, centroids: np.ndarray) -> int:
    """Return index of nearest centroid by L2 distance."""
    distances = np.linalg.norm(centroids - hist[None, :], axis=1)
    return int(np.argmin(distances))
