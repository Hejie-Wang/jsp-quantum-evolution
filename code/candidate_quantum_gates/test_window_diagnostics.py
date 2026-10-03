"""Tests for the T01 window diagnostics (window_diagnostics.py)."""
import itertools
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "candidate_quantum_medium"))

import medium_qjsp as mq
from window_diagnostics import (check_choice, d_imp_from_incumbent,
                                enumerate_pool, generate_d0, build_window,
                                swap_reachability)


class WindowDiagnosticsTests(unittest.TestCase):
    def test_checker_agrees_with_production_decoder(self):
        rng = np.random.default_rng(23)
        for trial in range(20):
            inst = generate_d0(3, 3, int(rng.integers(0, 1000)))
            sizes = [3, 3, 3]
            pool = tuple(
                tuple(itertools.permutations(inst.groups[m]))
                for m in range(inst.machines))
            mq.validate_pool(inst, pool)
            for choice in [(0, 0, 0), (1, 2, 0), (2, 2, 2)]:
                if any(a >= len(pool[m]) for m, a in enumerate(choice)):
                    continue
                mine = check_choice(inst, pool, choice)
                prod = mq.decode(inst, [pool[m][a] for m, a in enumerate(choice)])
                self.assertEqual(mine["feasible"], prod["feasible"],
                                 f"feasibility mismatch, trial {trial}")
                if mine["feasible"]:
                    self.assertEqual(mine["makespan"], prod["makespan"])

    def test_enumeration_matches_exhaustive_decode(self):
        inst = generate_d0(2, 2, 5)
        pool = tuple(
            tuple(itertools.permutations(inst.groups[m]))
            for m in range(inst.machines))
        enum = enumerate_pool(inst, pool)
        brute = []
        for choice in itertools.product(*(range(len(p)) for p in pool)):
            d = mq.decode(inst, [pool[m][a] for m, a in enumerate(choice)])
            brute.append(d["makespan"] if d["feasible"] else None)
        self.assertEqual(enum["pool_optimum"], min(v for v in brute if v is not None))
        self.assertEqual(enum["feasible_combinations"],
                         sum(1 for v in brute if v is not None))

    def test_d_imp_and_swap_reachability_constructed(self):
        # Incumbent = the worst feasible combination of a full 2x2 pool; the
        # best combination is the improving target, so d_imp >= 1 and the
        # one-swap graph must reach some improvement with a finite bottleneck.
        inst = generate_d0(2, 2, 3)
        pool = tuple(
            tuple(itertools.permutations(inst.groups[m]))
            for m in range(inst.machines))
        enum = enumerate_pool(inst, pool, u0=10**9)
        makespans = enum["makespans"]
        sizes = [len(p) for p in pool]
        best = min(m for m in makespans if m is not None)
        worst = max(m for m in makespans if m is not None)
        self.assertGreater(best, 0)
        inc = next(c for c, m in zip(itertools.product(*(range(k) for k in sizes)),
                                     makespans) if m == worst)
        d = d_imp_from_incumbent(inc, sizes, best + 1, makespans)
        self.assertGreaterEqual(d, 1)
        swap = swap_reachability(pool, inc, best + 1, makespans)
        self.assertTrue(swap["improvement_reachable"])
        self.assertLessEqual(swap["bottleneck_makespan"], worst)

    def test_no_improvement_window_reports_none(self):
        inst = generate_d0(3, 3, 0)
        pool, inc, u0 = build_window(inst, 2, 0)
        enum = enumerate_pool(inst, pool, u0=u0)
        if enum["improving_combinations"] == 0:
            sizes = [len(p) for p in pool]
            self.assertIsNone(
                d_imp_from_incumbent(inc, sizes, u0, enum["makespans"]))
            self.assertFalse(
                swap_reachability(pool, inc, u0, enum["makespans"])
                ["improvement_reachable"])


if __name__ == "__main__":
    unittest.main()
