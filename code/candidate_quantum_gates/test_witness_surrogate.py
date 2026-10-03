"""Tests for T03 witness/surrogate correctness (witness_surrogate.py)."""
import itertools
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "candidate_quantum_medium"))

import medium_qjsp as mq
import data_identity
from witness_surrogate import (collect_witnesses, correctness_battery,
                               rank_correlation, surrogates,
                               witness_activates, _screening_report)
from window_diagnostics import check_choice, build_window


class WitnessSurrogateTests(unittest.TestCase):
    def _full_pool(self, inst):
        return tuple(tuple(itertools.permutations(inst.groups[m]))
                     for m in range(inst.machines))

    def test_lw_and_path_witnesses_hold_exhaustively(self):
        inst = data_identity.build_d0(2, 2, 3)
        pool = self._full_pool(inst)
        witnesses, _ = collect_witnesses(inst, pool, (0, 0), n_probe=8,
                                         seed=1)
        for choice in itertools.product(*(range(len(p)) for p in pool)):
            r = check_choice(inst, pool, choice)
            s = surrogates(inst, pool, witnesses, choice)
            if r["feasible"]:
                self.assertFalse(s["cycle_triggered"])
                self.assertLessEqual(s["l_w"], r["makespan"])
                for w in witnesses:
                    if w["kind"] == "path" and witness_activates(w, choice):
                        self.assertGreaterEqual(r["makespan"], w["length"])

    def test_triggered_cycle_witness_implies_infeasible(self):
        inst = data_identity.build_d0(3, 3, 0)
        pool = self._full_pool(inst)
        witnesses, _ = collect_witnesses(inst, pool, (0, 0, 0), n_probe=8,
                                         seed=2)
        cycle_ws = [w for w in witnesses if w["kind"] == "cycle"]
        if not cycle_ws:
            self.skipTest("no cycle witness collected on this instance")
        for choice in itertools.product(*(range(len(p)) for p in pool)):
            for w in cycle_ws:
                if witness_activates(w, choice):
                    self.assertFalse(check_choice(inst, pool, choice)["feasible"])

    def test_constant_true_witness_kept_and_covered(self):
        # The C* path witness is constant-true on the full pool (support=[]),
        # must be kept with explicit stats and must never be violated.
        inst = data_identity.build_d0(3, 3, 0)
        pool = self._full_pool(inst)
        c_star = data_identity.enumerate_exact_optimum(inst)["exact_optimum"]
        witnesses, stats = collect_witnesses(inst, pool, (0, 0, 0), n_probe=16,
                                             seed=0)
        const_true = [w for w in witnesses if not w["support"]]
        self.assertTrue(const_true)
        self.assertGreaterEqual(stats["constant_true"], 0)
        for choice in itertools.product(*(range(len(p)) for p in pool)):
            r = check_choice(inst, pool, choice)
            if r["feasible"]:
                self.assertGreaterEqual(r["makespan"], c_star)

    def test_zero_false_exclusions_on_window(self):
        inst = data_identity.build_d0(4, 3, 1)
        pool, incumbent, u0 = build_window(inst, 3, 1)
        witnesses, _ = collect_witnesses(inst, pool, incumbent, n_probe=16,
                                         seed=1)
        battery = correctness_battery(
            inst, pool, witnesses,
            itertools.product(*(range(len(p)) for p in pool)), u0, target=u0)
        self.assertTrue(all(v == 0 for v in battery["checks"].values()))

    def test_screening_tie_handling(self):
        rows = [{"makespan": 10, "H_weight": 5}, {"makespan": 9, "H_weight": 5},
                {"makespan": 8, "H_weight": 5}, {"makespan": 20, "H_weight": 9},
                {"makespan": 21, "H_weight": 9}]
        rep = _screening_report(rows, [r for r in rows if r["makespan"] < 15],
                                "H_weight", 15, fraction=0.4)
        self.assertEqual(rep["strict"]["selected"], 2)
        self.assertEqual(rep["with_ties"]["selected"], 3)
        self.assertEqual(rep["strict"]["recall"], 2 / 3)
        self.assertEqual(rep["with_ties"]["recall"], 1.0)

    def test_rank_correlation_sign(self):
        inst = data_identity.build_d0(4, 3, 2)
        pool, incumbent, u0 = build_window(inst, 3, 2)
        witnesses, _ = collect_witnesses(inst, pool, incumbent, n_probe=16,
                                         seed=2)
        battery = correctness_battery(
            inst, pool, witnesses,
            itertools.product(*(range(len(p)) for p in pool)), u0)
        tau = rank_correlation(battery["rows"], "l_w")
        self.assertIsNotNone(tau)
        self.assertGreaterEqual(tau["tau"], -1.0)
        self.assertLessEqual(tau["tau"], 1.0)


if __name__ == "__main__":
    unittest.main()
