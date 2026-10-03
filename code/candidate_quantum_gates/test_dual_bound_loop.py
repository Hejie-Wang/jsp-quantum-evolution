#!/usr/bin/env python3
"""Unit tests for the T08 dual-bound closed loop.

Run inside the qskit environment:

    python code/candidate_quantum_gates/test_dual_bound_loop.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

GATES_DIR = Path(__file__).resolve().parent
if str(GATES_DIR) not in sys.path:
    sys.path.insert(0, str(GATES_DIR))

import data_identity  # noqa: E402
import dual_bound_loop as dbl  # noqa: E402
import global_bounds as gb  # noqa: E402
from window_diagnostics import build_window, check_choice  # noqa: E402


def window(seed=0, jobs=3, machines=3, k=3):
    inst = data_identity.build_d0(jobs, machines, seed)
    pool, incumbent, u0 = build_window(inst, k, seed + 1000)
    return inst, pool


class InitTests(unittest.TestCase):
    def test_initial_bounds_are_verified_and_logged(self):
        inst, pool = window()
        loop = dbl.DualBoundLoop(inst, pool, time_budget=1.0, seed=0)
        self.assertEqual(loop.L, gb.trivial_bound(inst))
        self.assertEqual(loop.L_scope, "global")
        self.assertLessEqual(loop.L, loop.U)
        check = check_choice(inst, pool, loop.U_choice)
        self.assertEqual(check["makespan"], loop.U)
        self.assertEqual(loop.events[0]["kind"], "bound")
        self.assertEqual(loop.events[0]["bound_kind"], "trivial_bound")
        self.assertEqual(loop.events[1]["kind"], "init")
        self.assertGreaterEqual(len(loop.witnesses), 1)


class ChannelTests(unittest.TestCase):
    def test_improvement_only_from_verified_schedules(self):
        inst, pool = window()
        loop = dbl.DualBoundLoop(inst, pool, time_budget=1.0, seed=0)
        best = min(loop.U,
                   min(check_choice(inst, pool, c)["makespan"]
                       for c in __import__("itertools").product(
                           *(range(len(m)) for m in pool))
                       if check_choice(inst, pool, c)["feasible"]))
        loop.proposer = dbl.FixedProposer(min(
            [c for c in __import__("itertools").product(
                *(range(len(m)) for m in pool))
             if check_choice(inst, pool, c)["feasible"]
             and check_choice(inst, pool, c)["makespan"] == best]))
        before = loop.U
        loop.propose_and_verify()
        self.assertEqual(loop.U, min(before, best))
        for event in loop.events:
            if event["kind"] == "improvement":
                self.assertEqual(
                    event["makespan"],
                    check_choice(inst, pool, tuple(event["choice"]))["makespan"])

    def test_empty_proposal_never_raises_the_lower_bound(self):
        inst, pool = window()
        loop = dbl.DualBoundLoop(inst, pool, time_budget=1.0, seed=0,
                                 proposer=dbl.FixedProposer(None))
        before = loop.L
        for _ in range(5):
            loop.propose_and_verify()
        self.assertEqual(loop.L, before)
        self.assertEqual(loop.failures["empty_proposal"], 5)

    def test_all_infeasible_samples_never_raise_the_lower_bound(self):
        inst, pool = window()
        infeasible = dbl._an_infeasible_choice(inst, pool)
        loop = dbl.DualBoundLoop(inst, pool, time_budget=1.0, seed=0,
                                 proposer=dbl.FixedProposer(infeasible))
        before = loop.L
        for _ in range(5):
            loop.propose_and_verify()
        self.assertEqual(loop.L, before)
        self.assertGreaterEqual(loop.failures["sampling_failure"], 5)

    def test_zero_budget_certificate_proves_nothing(self):
        inst, pool = window()
        loop = dbl.DualBoundLoop(inst, pool, time_budget=1.0, seed=0)
        before = loop.L
        raised = loop.certify(time_limit=0.0)
        self.assertFalse(raised)
        self.assertEqual(loop.L, before)
        self.assertEqual(loop.failures["certificate_unproven"], 1)

    def test_certificate_can_prove_optimality_on_a_tiny_instance(self):
        inst, pool = window(seed=0, jobs=2, machines=2)
        loop = dbl.DualBoundLoop(inst, pool, time_budget=2.0, seed=0)
        raised = loop.certify(time_limit=5.0)
        self.assertTrue(raised)
        self.assertEqual(loop.L, loop.U)
        self.assertEqual(loop.L_scope, "global")
        self.assertTrue(loop.summary()["gap_closed"])


class PoolChangeTests(unittest.TestCase):
    def test_pool_change_reprojects_witnesses_and_drops_pool_bounds(self):
        inst, pool = window(seed=0)
        loop = dbl.DualBoundLoop(inst, pool, time_budget=1.0, seed=0)
        loop.pool_scope_bounds = {"stale": 999}
        before, scope = loop.L, loop.L_scope
        other = build_window(inst, 3, 2000)[0]
        report = loop.change_pool(other)
        self.assertEqual(loop.L, before)
        self.assertEqual(loop.L_scope, scope)
        self.assertEqual(report["pool_scope_bounds_dropped"], 1)
        for witness in loop.witnesses:
            self.assertIsNotNone(__import__("medium_qjsp").witness_support(witness, other))
        self.assertEqual(loop.events[-1]["kind"], "pool_change")


class InvariantTests(unittest.TestCase):
    def test_d0_invariants_hold_at_every_event(self):
        for jobs, machines, seed in ((2, 2, 0), (3, 3, 0)):
            inst = data_identity.build_d0(jobs, machines, seed)
            pool, _incumbent, _u0 = build_window(inst, 3, seed + 1000)
            loop = dbl.DualBoundLoop(inst, pool, time_budget=1.0,
                                     cert_time_limit=0.5, seed=seed)
            loop.run(max_rounds=40, cert_every=5)
            exact = data_identity.enumerate_exact_optimum(inst)["exact_optimum"]
            self.assertEqual(dbl.check_invariants(loop, exact), [])
            self.assertGreaterEqual(loop.U, exact)

    def test_event_log_is_replayable(self):
        inst, pool = window(seed=1)
        loop = dbl.DualBoundLoop(inst, pool, time_budget=1.0, seed=1)
        loop.run(max_rounds=20, cert_every=4)
        times = [event["elapsed_seconds"] for event in loop.events]
        self.assertEqual(times, sorted(times))
        self.assertEqual([event["index"] for event in loop.events],
                         list(range(len(loop.events))))
        for event in loop.events:
            for field in ("kind", "lower_bound", "upper_bound", "bounds_before_after",
                          "graph_evaluations", "witnesses", "pool_hash"):
                self.assertIn(field, event)


if __name__ == "__main__":
    unittest.main(verbosity=2)
