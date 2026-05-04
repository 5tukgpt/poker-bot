"""Tests for equity-based card abstraction."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from poker.ai.engine.card import Card
from poker.ai.strategy.abstraction import (
    EquityBucketer,
    all_preflop_hands,
    canonical_preflop_class,
    compute_equity_histogram,
)


class TestPreflopClassification:
    def test_169_canonical_classes(self):
        hands = all_preflop_hands()
        classes = {canonical_preflop_class(h) for h in hands}
        assert len(classes) == 169

    def test_aa_class(self):
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()]
        assert canonical_preflop_class(hole) == 'AA'

    def test_aks_vs_ako(self):
        suited = [Card.from_str('As').to_int(), Card.from_str('Ks').to_int()]
        offsuit = [Card.from_str('As').to_int(), Card.from_str('Kc').to_int()]
        assert canonical_preflop_class(suited) == 'AKs'
        assert canonical_preflop_class(offsuit) == 'AKo'


class TestEquityHistogram:
    def test_shape_and_sum(self):
        hole = [Card.from_str('As').to_int(), Card.from_str('Ks').to_int()]
        hist = compute_equity_histogram(hole, [], num_rollouts=50, num_bins=10)
        assert hist.shape == (10,)
        assert abs(hist.sum() - 1.0) < 1e-5

    def test_premium_skewed_high(self):
        # AA should have most mass in high-equity bins
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()]
        hist = compute_equity_histogram(hole, [], num_rollouts=200, num_bins=10)
        # More mass in top half than bottom half
        assert hist[5:].sum() > hist[:5].sum()

    def test_trash_skewed_low(self):
        # 72o should have most mass in low-equity bins
        hole = [Card.from_str('7c').to_int(), Card.from_str('2d').to_int()]
        hist = compute_equity_histogram(hole, [], num_rollouts=200, num_bins=10)
        # More mass in bottom half than top half
        assert hist[:5].sum() > hist[5:].sum()


class TestBucketer:
    def test_precompute_small(self):
        # Tiny clusters for fast test
        bucketer = EquityBucketer.precompute(
            clusters_per_street={'preflop': 10, 'flop': 5, 'turn': 5, 'river': 5},
            rollouts=20,
            postflop_samples=50,
            verbose=False,
        )
        assert bucketer.total_buckets() > 0
        assert len(bucketer.preflop_lookup) == 169  # all classes mapped

    def test_bucket_consistent(self):
        bucketer = EquityBucketer.precompute(
            clusters_per_street={'preflop': 5, 'flop': 5, 'turn': 5, 'river': 5},
            rollouts=20,
            postflop_samples=50,
            verbose=False,
        )
        hole = [Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()]
        b1 = bucketer.bucket(hole, [], street=0)
        b2 = bucketer.bucket(hole, [], street=0)
        assert b1 == b2  # preflop is deterministic

    def test_premium_and_trash_different_buckets(self):
        bucketer = EquityBucketer.precompute(
            clusters_per_street={'preflop': 50, 'flop': 5, 'turn': 5, 'river': 5},
            rollouts=50,
            postflop_samples=50,
            verbose=False,
        )
        aa = bucketer.bucket([Card.from_str('Ac').to_int(), Card.from_str('Ad').to_int()], [], 0)
        seven_two = bucketer.bucket([Card.from_str('7c').to_int(), Card.from_str('2d').to_int()], [], 0)
        assert aa != seven_two

    def test_save_load(self):
        bucketer = EquityBucketer.precompute(
            clusters_per_street={'preflop': 10, 'flop': 5, 'turn': 5, 'river': 5},
            rollouts=20,
            postflop_samples=30,
            verbose=False,
        )
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            bucketer.save(path)
            loaded = EquityBucketer.load(path)
            assert loaded.preflop_lookup == bucketer.preflop_lookup
            for k in bucketer.centroids:
                assert np.allclose(loaded.centroids[k], bucketer.centroids[k])
        finally:
            os.unlink(path)
