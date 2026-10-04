#!/usr/bin/env python3
"""Unit tests for the T05 encoding-equivalence module.

Run inside the qskit environment:

    python code/candidate_quantum_gates/test_encoding_equivalence.py
"""
from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

GATES_DIR = Path(__file__).resolve().parent
if str(GATES_DIR) not in sys.path:
    sys.path.insert(0, str(GATES_DIR))

import data_identity  # noqa: E402
import encoding_equivalence as ee  # noqa: E402
import medium_qjsp as mq  # noqa: E402
from window_diagnostics import build_window  # noqa: E402
from witness_surrogate import collect_witnesses  # noqa: E402

#: Two-machine pool whose orders carry the synthetic witnesses below.
SYNTH_POOL = (((0, 1), (1, 0)), ((1, 0), (0, 1)))


def synthetic_witnesses(include_constant_true=False):
    """Predicate set over ``SYNTH_POOL``: one cycle cut and two path cuts.

    Relations and allowed-label sets are consistent with ``SYNTH_POOL``, so the
    gate layer (which re-derives labels from the relations) and the reference
    predicate layer agree by construction.
    """
    witnesses = [
        {"kind": "cycle", "length": 9, "relations": [(0, 0, 1), (1, 1, 0)],
         "support": [0, 1], "allowed": {0: [0], 1: [0]}},
        {"kind": "path", "length": 7, "relations": [(0, 0, 1)],
         "support": [0], "allowed": {0: [0]}},
        {"kind": "path", "length": 3, "relations": [(1, 1, 0)],
         "support": [1], "allowed": {1: [0]}},
    ]
    if include_constant_true:
        witnesses.append({"kind": "path", "length": 8, "relations": [],
                          "support": [], "allowed": {}})
    return witnesses


def small_window(seed=0, k=2):
    inst = data_identity.build_d0(2, 2, seed)
    pool, incumbent, u0 = build_window(inst, k, seed + 1000)
    return inst, pool, incumbent, u0


class AndGadgetTests(unittest.TestCase):
    def test_truth_table_is_exact(self):
        for p, q, y in itertools.product((0, 1), repeat=3):
            value = ee._and_gadget_penalty(p, q, y)
            self.assertGreaterEqual(value, 0.0)
            self.assertEqual(value == 0.0, y == (p and q), (p, q, y))

    def test_product_penalty_only_fires_on_full_support(self):
        rules = ee.compile_predicates([2, 2], synthetic_witnesses(), 5)
        self.assertEqual(len(rules["cuts"]), 2)
        # choice (1, 1): the cycle needs label 0 on both machines -> inactive
        self.assertEqual(ee.min_energy_over_aux_and(rules, (1, 1)), 0.0)
        # choice (0, 0): the cycle fires, so the eliminant is strictly positive
        self.assertGreater(ee.min_energy_over_aux_and(rules, (0, 0)), 0.0)
        self.assertGreater(ee.min_energy_over_aux_qubo(rules, (0, 0)), 0.0)


class PredicateCompilationTests(unittest.TestCase):
    def test_cycle_always_included_and_path_threshold(self):
        rules = ee.compile_predicates([2, 2], synthetic_witnesses(), 5)
        self.assertEqual([c["kind"] for c in rules["cuts"]], ["cycle", "path"])
        self.assertFalse(rules["infeasible_by_witness"])

    def test_path_witness_below_target_is_dropped(self):
        rules = ee.compile_predicates([2, 2], synthetic_witnesses(), 9)
        self.assertEqual([c["kind"] for c in rules["cuts"]], ["cycle"])

    def test_constant_true_witness_is_recorded_not_dropped(self):
        rules = ee.compile_predicates([2, 2],
                                      synthetic_witnesses(include_constant_true=True), 5)
        self.assertEqual(rules["constant_true_witnesses"], [3])
        self.assertTrue(rules["infeasible_by_witness"])
        # every combination now pays the constant penalty
        self.assertGreater(ee.min_energy_over_aux_qubo(rules, (1, 1)), 0.0)
        self.assertGreater(ee.min_energy_over_aux_and(rules, (1, 1)), 0.0)


class MinOverAuxTests(unittest.TestCase):
    def test_closed_form_and_dp_match_brute_force(self):
        rules = ee.compile_predicates([2, 2],
                                      synthetic_witnesses(include_constant_true=True), 5)
        q_form, _ = ee.build_squared_residual_qubo(rules)
        a_form, _ = ee.build_and_quadratization(rules)
        for choice in ee.all_choices(rules["sizes"]):
            bits = ee.choice_bits(rules["sizes"], choice)
            self.assertAlmostEqual(
                ee.brute_force_min_energy(q_form, rules["data_bits"], bits, cap=16),
                ee.min_energy_over_aux_qubo(rules, choice), places=9)
            self.assertAlmostEqual(
                ee.brute_force_min_energy(a_form, rules["data_bits"], bits, cap=16),
                ee.min_energy_over_aux_and(rules, choice), places=9)


class SparseQuboTests(unittest.TestCase):
    def test_sparse_matrix_equals_production_encoder(self):
        inst, pool, incumbent, u0 = small_window()
        witnesses, _ = collect_witnesses(inst, pool, incumbent, n_probe=8, seed=0)
        rules = ee.compile_predicates([len(p) for p in pool], witnesses, u0 - 1)
        form, meta = ee.build_squared_residual_qubo(rules)
        cuts = [{"support": tuple(c["support"]),
                 "allowed": {m: tuple(c["allowed"][m]) for m in c["support"]}}
                for c in rules["cuts"]]
        dense, production_meta = mq.build_qubo(pool, cuts)
        self.assertEqual(dense.shape, (meta["total_vars"], meta["total_vars"]))
        np.testing.assert_allclose(form.dense(), dense, atol=1e-12)
        self.assertAlmostEqual(form.constant, float(production_meta["constant"]), places=12)

    def test_zero_energy_equals_reference_feasibility(self):
        inst, pool, incumbent, u0 = small_window()
        witnesses, _ = collect_witnesses(inst, pool, incumbent, n_probe=8, seed=0)
        rules = ee.compile_predicates([len(p) for p in pool], witnesses, u0 - 1)
        checks, aux_checked = ee.verify_zero_set(rules)
        self.assertEqual(checks["qubo_zero_mismatch"], 0)
        self.assertEqual(checks["and_zero_mismatch"], 0)
        self.assertEqual(checks["aux_minimisation_mismatch"], 0)
        self.assertGreater(aux_checked, 0)


class NativePhaseTests(unittest.TestCase):
    def _states(self, rules):
        return [(choice, (-1.0) ** ee.ref_violations(rules, choice))
                for choice in ee.all_choices(rules["sizes"])]

    def test_phase_matches_reference_and_work_is_uncomputed(self):
        witnesses = synthetic_witnesses()
        rules = ee.compile_predicates([2, 2], witnesses, 5)
        pool_obj, circuit = ee.build_native_phase(SYNTH_POOL, witnesses, 5)
        report = ee.native_phase_diagonal(pool_obj, circuit, self._states(rules),
                                          max_qubits=12)
        self.assertEqual(report["checked"], 4)
        self.assertLessEqual(report["max_abs_error"], ee.PHASE_TOLERANCE)
        self.assertEqual(report["dirty_work_states"], 0)

    def test_global_phase_of_constant_true_witness(self):
        witnesses = synthetic_witnesses(include_constant_true=True)
        rules = ee.compile_predicates([2, 2], witnesses, 5)
        pool_obj, circuit = ee.build_native_phase(SYNTH_POOL, witnesses, 5)
        report = ee.native_phase_diagonal(pool_obj, circuit, self._states(rules),
                                          max_qubits=12)
        self.assertLessEqual(report["max_abs_error"], ee.PHASE_TOLERANCE)
        self.assertEqual(report["dirty_work_states"], 0)


class ExactSolverTests(unittest.TestCase):
    def test_cpsat_minimum_matches_enumeration(self):
        rules = ee.compile_predicates([2, 2], synthetic_witnesses(), 5)
        expected_qubo = min(ee.min_energy_over_aux_qubo(rules, c)
                            for c in ee.all_choices(rules["sizes"]))
        expected_and = min(ee.min_energy_over_aux_and(rules, c)
                           for c in ee.all_choices(rules["sizes"]))
        qubo = ee.cpsat_min_energy(rules, "qubo", time_limit=10.0)
        and_sat = ee.cpsat_min_energy(rules, "and", time_limit=10.0)
        self.assertIn(qubo["status"], ("OPTIMAL", "FEASIBLE"))
        self.assertIn(and_sat["status"], ("OPTIMAL", "FEASIBLE"))
        self.assertAlmostEqual(qubo["objective"], expected_qubo, places=9)
        self.assertAlmostEqual(and_sat["objective"], expected_and, places=9)

    def test_linear_master_reports_infeasible_for_constant_true_cut(self):
        rules = ee.compile_predicates([2, 2],
                                      synthetic_witnesses(include_constant_true=True), 5)
        status, choice = ee.linear_master_feasible(rules, time_limit=10.0)
        self.assertEqual(status, "infeasible")
        self.assertIsNone(choice)

    def test_linear_master_finds_feasible_combination(self):
        rules = ee.compile_predicates([2, 2], synthetic_witnesses(), 5)
        status, choice = ee.linear_master_feasible(rules, time_limit=10.0)
        self.assertIn(status, ("optimal", "limit_with_incumbent"))
        self.assertTrue(ee.ref_feasible(rules, choice))


class PenaltyWeightTests(unittest.TestCase):
    def test_flip_point_respects_closed_form_bound(self):
        inst, pool, incumbent, u0 = small_window(seed=1)
        witnesses, _ = collect_witnesses(inst, pool, incumbent, n_probe=8, seed=1)
        rules = ee.compile_predicates([len(p) for p in pool], witnesses, u0 - 1)
        makespans = [ee.check_choice(inst, pool, c)["makespan"]
                     for c in ee.all_choices(rules["sizes"])]
        sweep = ee.penalty_weight_sweep(rules, makespans, encoding="and")
        if sweep["closed_form_bound"] is None or sweep["first_infeasible_scale"] is None:
            self.skipTest("window has no infeasible combination or no flip in the grid")
        self.assertGreaterEqual(sweep["first_infeasible_scale"],
                                sweep["closed_form_bound"] * 0.999)


if __name__ == "__main__":
    unittest.main(verbosity=2)
