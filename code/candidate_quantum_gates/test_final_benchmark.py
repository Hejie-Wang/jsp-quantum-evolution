#!/usr/bin/env python3
"""Unit tests for the T10 end-to-end benchmark module.

Run inside the qskit environment:

    python code/candidate_quantum_gates/test_final_benchmark.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

GATES_DIR = Path(__file__).resolve().parent
if str(GATES_DIR) not in sys.path:
    sys.path.insert(0, str(GATES_DIR))

import data_identity  # noqa: E402
import final_benchmark as fb  # noqa: E402
from window_diagnostics import build_window, check_choice  # noqa: E402

D1_REPORT = sorted((GATES_DIR.parent / "candidate_quantum_medium" / "results"
                    / "corrected_20261001").glob("*_milp.json"))


def tiny_window(seed=2):
    inst = data_identity.build_d0(3, 3, seed)
    pool, incumbent, u0 = build_window(inst, 3, seed + 1000)
    best = min(check_choice(inst, pool, c)["makespan"]
               for c in __import__("itertools").product(
                   *(range(len(m)) for m in pool))
               if check_choice(inst, pool, c)["feasible"])
    return inst, pool, incumbent, u0, best


class ProposerTests(unittest.TestCase):
    def test_uniform_proposer_stays_inside_the_pool(self):
        proposer = fb._Uniform([3, 3, 3], seed=1)
        for _ in range(20):
            choice = proposer.propose()
            self.assertEqual(len(choice), 3)
            self.assertTrue(all(0 <= a < 3 for a in choice))
        cost = proposer.cost()
        self.assertEqual(cost["graph_evaluations"], 1)
        self.assertEqual(cost["shots"], 0)

    def test_sa_proposer_counts_witness_evaluations(self):
        inst, pool, incumbent, u0, best = tiny_window()
        from witness_surrogate import collect_witnesses
        import search_loop as sl
        import circuits
        raw, _ = collect_witnesses(inst, pool, incumbent, n_probe=8, seed=0)
        pool_obj = circuits.CandidatePool(tuple(tuple(tuple(int(v) for v in o)
                                                      for o in m) for m in pool))
        specs = sl.graph_witness_specs(pool_obj, [
            __import__("medium_qjsp").Witness(w["kind"],
                                              tuple(tuple(r) for r in w["relations"]),
                                              int(w["length"]), ()) for w in raw])
        proposer = fb.SAProposer(inst, pool, specs, u0 - 1, steps=10, seed=3)
        choice = proposer.propose()
        self.assertEqual(len(choice), len(pool))
        self.assertEqual(proposer.cost()["witness_evaluations"], 10)


class ArmTests(unittest.TestCase):
    def test_cp_sat_arm_reports_a_status_and_matching_hit_flag(self):
        inst, pool, incumbent, u0, best = tiny_window()
        arm = fb.run_cp_sat(inst, 3.0, best)
        self.assertEqual(arm["arm"], "cp_sat")
        self.assertIn("solver_status", arm)
        if arm["upper_bound"] is not None:
            self.assertEqual(arm["hit"], arm["upper_bound"] <= best)

    def test_dual_arm_shares_the_target_and_records_its_budget(self):
        inst, pool, incumbent, u0, best = tiny_window()
        arm = fb.run_dual_arm(inst, pool, arm="uniform_dual", budget=2.0,
                              target=best, proposer_factory=fb.ARMS["uniform_dual"],
                              incumbent=incumbent, seed=0)
        self.assertEqual(arm["arm"], "uniform_dual")
        self.assertLessEqual(arm["wall_seconds"], 2.5)
        self.assertGreaterEqual(arm["upper_bound"], best - 1)
        self.assertIn("gap_integral", arm)

    def test_joint_mainline_reports_an_arrival_upper_bound(self):
        inst, pool, incumbent, u0, best = tiny_window()
        arm = fb.run_joint_mainline(inst, pool, incumbent, 2.0, best, seed=0)
        self.assertTrue(arm["cost_breakdown"]["arrival_is_upper_bound"])
        self.assertEqual(arm["hit"], arm["upper_bound"] <= best)


class D1Tests(unittest.TestCase):
    def test_d1_loader_validates_the_old_pool(self):
        if not D1_REPORT:
            self.skipTest("no D1 pool report available")
        inst, pool, incumbent, u0 = fb.load_d1_pool(D1_REPORT[0])
        self.assertEqual(len(pool), inst.machines)
        self.assertEqual(len(incumbent), len(pool))
        self.assertGreater(u0, 0)
        self.assertTrue(check_choice(inst, pool, incumbent)["feasible"])

    def test_d1_layer_marks_the_variational_arm_as_resource_limited(self):
        if not D1_REPORT:
            self.skipTest("no D1 pool report available")
        rows = fb.run_d1(budget=0.2, max_instances=1, include_quantum=False)
        self.assertEqual(len(rows), 1)
        arms = {arm["arm"]: arm for arm in rows[0]["arms"]}
        self.assertEqual(arms["quantum_dual"]["solver_status"],
                         "skipped_resource_limit")
        self.assertIn("prod(K_m)", arms["quantum_dual"]["reason"])
        self.assertGreater(rows[0]["candidate_subspace"], 65536)
        self.assertEqual(rows[0]["target"], rows[0]["incumbent_makespan"] - 1)


class GateTests(unittest.TestCase):
    def _row(self, arm_values, no_target=False):
        arms = [{"arm": name, "hit": arrival is not None,
                 "arrival_seconds": arrival, "wall_seconds": arrival or 5.0,
                 "upper_bound": 10, "lower_bound": 8, "solver_status": "complete",
                 "graph_evaluations": 1, "certificate_calls": 0, "cost_breakdown": {}}
                for name, arrival in arm_values.items()]
        return {"scale": "3x3", "seed": 0, "arms": arms, "no_target": no_target,
                "budget_seconds": 5.0}

    def test_gate_skips_windows_without_a_target(self):
        rows = [self._row({"uniform_dual": 1.0, "sa_dual": 2.0}, no_target=True)]
        self.assertEqual(fb.evaluate_gate(rows, budget=5.0), [])

    def test_gate_reports_ratios_against_the_baseline(self):
        rows = [self._row({"uniform_dual": 1.0, "sa_dual": 2.0, "quantum_dual": 1.2})]
        summary = {row["arm"]: row for row in fb.evaluate_gate(rows, budget=5.0)}
        self.assertAlmostEqual(summary["sa_dual"]["cost_ratio_vs_baseline"]["mean"],
                               2.0, places=9)
        self.assertFalse(summary["sa_dual"]["passes_benefit_gate"])
        self.assertAlmostEqual(summary["quantum_dual"]["cost_ratio_vs_baseline"]["mean"],
                               1.2, places=9)

    def test_gate_treats_a_miss_as_the_full_budget(self):
        rows = [self._row({"uniform_dual": 1.0, "quantum_dual": None})]
        summary = {row["arm"]: row for row in fb.evaluate_gate(rows, budget=5.0)}
        self.assertAlmostEqual(summary["quantum_dual"]["cost_ratio_vs_baseline"]["mean"],
                               5.0, places=9)
        self.assertEqual(summary["quantum_dual"]["success_difference"]["mean"], -1.0)

    def test_paired_bootstrap_interval_contains_the_mean(self):
        result = fb.paired_bootstrap([0.8, 1.0, 1.2, 0.9])
        self.assertLessEqual(result["ci95"][0], result["mean"])
        self.assertGreaterEqual(result["ci95"][1], result["mean"])
        self.assertIsNone(fb.paired_bootstrap([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
