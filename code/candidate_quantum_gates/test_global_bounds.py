"""Tests for T02 certified global bounds (global_bounds.py)."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "candidate_quantum_medium"))

import medium_qjsp as mq
import data_identity
from global_bounds import (full_model_bound, subset_relaxation_bound,
                           threshold_infeasibility, trivial_bound)


class GlobalBoundsTests(unittest.TestCase):
    def test_l0_and_threshold_match_enumeration(self):
        for jobs, machines, seed in ((2, 2, 0), (3, 3, 0), (4, 3, 5)):
            inst = data_identity.build_d0(jobs, machines, seed)
            c_star = data_identity.enumerate_exact_optimum(inst)["exact_optimum"]
            l0 = trivial_bound(inst)
            self.assertLessEqual(l0, c_star)
            self.assertEqual(
                threshold_infeasibility(inst, c_star - 1, 30)["verdict"],
                "infeasible")
            self.assertEqual(
                threshold_infeasibility(inst, c_star, 30)["verdict"],
                "feasible")

    def test_relaxation_monotone_and_below_optimum(self):
        inst = data_identity.build_d0(3, 3, 1)
        c_star = data_identity.enumerate_exact_optimum(inst)["exact_optimum"]
        single = subset_relaxation_bound(inst, (0,), 30)["certified_bound"]
        pair = subset_relaxation_bound(inst, (0, 1), 30)["certified_bound"]
        full = subset_relaxation_bound(inst, range(inst.machines), 30)
        self.assertIsNotNone(single)
        self.assertIsNotNone(pair)
        self.assertLessEqual(single, pair)
        self.assertLessEqual(pair, c_star)
        self.assertEqual(full["status"], "OPTIMAL")
        self.assertEqual(full["objective"], c_star)

    def test_unproven_threshold_never_raises_lower_bound(self):
        # A tiny time limit on a 20x20 instance will typically end UNKNOWN;
        # the contract requires that no non-INFEASIBLE verdict raises L.
        inst = mq.Instance.read(Path(__file__).resolve().parents[2]
                                / "task_data" / "ta21.txt")
        r = threshold_infeasibility(inst, 1, 0.05)  # trivially feasible probe
        if r["verdict"] != "infeasible":
            self.assertIsNone(r["raises_lower_bound_to"])

    def test_d1_ta01_reproduces_known_range(self):
        inst = mq.Instance.read(Path(__file__).resolve().parents[2]
                                / "task_data" / "tai15_15_01_test.txt")
        l0 = trivial_bound(inst)
        known = 1231
        self.assertLessEqual(l0, known)
        best = None
        for m in range(inst.machines):
            b = subset_relaxation_bound(inst, (m,), 30)["certified_bound"]
            self.assertLessEqual(b, known)
            best = b if best is None else max(best, b)
        self.assertGreaterEqual(best, l0)

    def test_full_model_matches_enumeration(self):
        inst = data_identity.build_d0(3, 3, 2)
        c_star = data_identity.enumerate_exact_optimum(inst)["exact_optimum"]
        r = full_model_bound(inst, 30)
        self.assertEqual(r["status"], "OPTIMAL")
        self.assertEqual(r["objective"], c_star)
        self.assertGreaterEqual(c_star, r["dual_bound"])


if __name__ == "__main__":
    unittest.main()
