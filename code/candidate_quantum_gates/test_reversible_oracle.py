#!/usr/bin/env python3
"""Unit tests for the T11 reversible micro-oracle.

Run inside the qskit environment:

    python code/candidate_quantum_gates/test_reversible_oracle.py
"""
from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

GATES_DIR = Path(__file__).resolve().parent
if str(GATES_DIR) not in sys.path:
    sys.path.insert(0, str(GATES_DIR))

import data_identity  # noqa: E402
import reversible_oracle as ro  # noqa: E402
from window_diagnostics import build_window, check_choice  # noqa: E402


def window(seed=0, k=2):
    inst = data_identity.build_d0(2, 2, seed)
    pool, incumbent, u0 = build_window(inst, k, seed + 1000)
    return inst, pool, u0


class WireCircuitTests(unittest.TestCase):
    def test_uncompute_restores_every_wire(self):
        circuit = ro.WireCircuit()
        a, b = circuit.alloc(2, "w")
        mark = circuit.mark()
        circuit.x(a)
        circuit.cx(a, b)
        before = list(circuit.gates)
        circuit.uncompute(mark)
        # the mirrored block is exactly the reverse of the recorded one
        self.assertEqual(circuit.gates[len(before):], list(reversed(before)))

    def test_controlled_pattern_xor_uses_one_mcx(self):
        circuit = ro.WireCircuit()
        controls = circuit.alloc(2, "c")
        target = circuit.alloc(1, "t")
        circuit.controlled_pattern_xor(controls, (1, 0), target)
        gates = circuit.gates
        self.assertEqual(sum(1 for g in gates if g[0] == "mcx"), 1)
        self.assertEqual(sum(1 for g in gates if g[0] == "x"), 2)


class StructuralTableTests(unittest.TestCase):
    def test_two_by_two_table_matches_independent_checker(self):
        inst, pool, u0 = window(seed=0)
        table = ro.structural_table(inst, pool, u0 - 1)
        self.assertEqual(table["k"], 2)
        self.assertEqual(len(table["entries"]), 4)
        for entry in table["entries"]:
            checked = check_choice(inst, pool, tuple(entry["choice"]))
            self.assertEqual(entry["acyclic"], checked["feasible"])
            self.assertEqual(entry["makespan"], checked["makespan"])
        # the window's optimum is 14, so T=13 admits nothing and T=14 admits one
        self.assertEqual(sum(1 for e in table["entries"] if e["satisfies"]), 0)
        reaching = ro.structural_table(inst, pool, u0)
        self.assertEqual(sum(1 for e in reaching["entries"] if e["satisfies"]), 1)

    def test_non_binary_machine_is_rejected(self):
        inst = data_identity.build_d0(3, 3, 0)
        pool, _incumbent, _u0 = build_window(inst, 3, 1000)
        with self.assertRaises(ValueError):
            ro.structural_table(inst, pool, 1)


class OracleTests(unittest.TestCase):
    def test_answer_matches_predicate_and_work_wires_reset(self):
        for seed in (0, 1, 2):
            inst, pool, u0 = window(seed=seed)
            for target in (u0 - 1, u0):
                report = ro.verify_oracle(inst, pool, target)
                self.assertEqual(report["checked"], 4)
                self.assertEqual(report["answer_mismatches"], 0, (seed, target))
                self.assertEqual(report["dirty_work_states"], 0, (seed, target))
                self.assertLessEqual(report["max_probability_error"], 1e-12)

    def test_gate_counts_follow_the_counted_model(self):
        inst, pool, u0 = window(seed=0)
        empty = ro.build_oracle(inst, pool, u0 - 1)["circuit"].counts()
        # two direction CNOTs + the answer CNOT + the mirrored uncompute
        self.assertEqual(empty["gates"], 5)
        self.assertEqual(empty["cx"], 5)
        one = ro.build_oracle(inst, pool, u0)
        self.assertEqual(one["satisfying_patterns"], 1)
        self.assertEqual(one["circuit"].counts()["mcx"], 2)  # compute + uncompute
        self.assertEqual(one["layout"]["total_wires"], 8)

    def test_infeasible_target_has_zero_success_probability(self):
        inst, pool, u0 = window(seed=0)
        self.assertEqual(ro.success_probability(inst, pool, u0 - 1), 0.0)
        self.assertGreater(ro.success_probability(inst, pool, u0), 0.0)

    def test_breakeven_is_none_when_no_state_is_satisfying(self):
        inst, pool, u0 = window(seed=0)
        analysis = ro.breakeven_analysis(inst, pool, u0 - 1, repeats=20)
        self.assertEqual(analysis["success_probability"], 0.0)
        self.assertIsNone(analysis["breakeven_gate_seconds"])
        self.assertTrue(all(row["quantum_wins"] is None for row in analysis["rows"]))
        feasible = ro.breakeven_analysis(inst, pool, u0, repeats=20)
        self.assertGreater(feasible["breakeven_gate_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
