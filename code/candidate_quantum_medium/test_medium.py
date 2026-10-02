#!/usr/bin/env python3
"""Cross-validation of the medium-scale machinery against the 3x3 demo fixture.

The fixture, pools and expected values come from code/candidate_quantum_demo.
The MILP cut loop starts with a verified incumbent and its path witness, finds
the pool optimum 11 and certifies it by proving T=10 infeasible. Enumeration of
the 8 combinations appears only in these tests, never in the solver.
"""
from __future__ import annotations

from itertools import product
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

DEMO_DIR = __import__("pathlib").Path(__file__).parent.parent / "candidate_quantum_demo"
__import__("sys").path.insert(0, str(DEMO_DIR))
import demo as demo_ref  # noqa: E402  (official 3x3 reference implementation)

from medium_qjsp import (Instance, build_qubo, compile_cuts, decode,
                         qubo_violation, run_candidate_loop, separate,
                         serial_sgs, validate_schedule, witness_support,
                         _flat, Witness, build_pool, validate_pool,
                         solve_master_milp, one_hot_choice, decode_ising_samples,
                         solve_qubo_kaiwu, content_hash)
from verify_result import verify_report

DURATIONS = ((3, 2, 2), (2, 1, 4), (4, 3, 1))
ROUTES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
POOLS = (((0, 5, 7), (7, 5, 0)),
         ((1, 3, 8), (3, 1, 8)),
         ((2, 4, 6), (4, 6, 2)))
CHOICES = tuple(product(range(2), repeat=3))


def fixture():
    return Instance(np.array(DURATIONS), np.array(ROUTES), "fixture_3x3")


def all_witnesses(inst):
    seen = {}
    for choice in CHOICES:
        w = separate(inst, decode(inst, [POOLS[m][a] for m, a in enumerate(choice)]))
        seen.setdefault(w.key(), w)
    return list(seen.values())


class MediumMachineryTests(unittest.TestCase):

    def test_decode_matches_demo_table(self):
        inst = fixture()
        for choice in CHOICES:
            mine = decode(inst, [POOLS[m][a] for m, a in enumerate(choice)])
            ref = demo_ref.decode(choice)
            self.assertEqual(mine["feasible"], ref["feasible"], choice)
            if ref["feasible"]:
                self.assertEqual(mine["makespan"], ref["makespan"], choice)
                self.assertEqual(validate_schedule(inst, mine["starts"]),
                                 ref["makespan"], choice)
            else:
                self.assertEqual(set(mine["nodes"]), set(ref["nodes"]), choice)

    def test_witness_projection_and_validity(self):
        inst = fixture()
        witnesses = all_witnesses(inst)
        self.assertEqual(len(witnesses), 6)  # demo reference count
        cuts, dropped = compile_cuts(POOLS, witnesses)
        self.assertEqual(dropped, 0)
        for choice in CHOICES:
            result = decode(inst, [POOLS[m][a] for m, a in enumerate(choice)])
            for cut in cuts:
                active = all(choice[m] in cut["allowed"][m] for m in cut["support"])
                if not active:
                    continue
                if cut["kind"] == "cycle":
                    self.assertFalse(result["feasible"],
                                     f"active cycle cut on feasible {choice}")
                elif result["feasible"]:
                    self.assertGreaterEqual(result["makespan"], cut["length"], choice)

    def test_qubo_zero_iff_all_cuts_hold(self):
        inst = fixture()
        witnesses = all_witnesses(inst)
        cuts, _ = compile_cuts(POOLS, witnesses)
        q, meta = build_qubo(POOLS, cuts)
        sizes = [2, 2, 2]
        n_slack = meta["total_vars"] - meta["base_vars"]
        self.assertEqual(meta["base_vars"], 6)
        for choice in CHOICES:
            x = np.zeros(meta["total_vars"], dtype=float)
            for m, a in enumerate(choice):
                x[_flat(sizes, m, a)] = 1.0
            # enumerate slack assignments; minimum energy must be 0 iff no cut
            # and no one-hot constraint is violated
            best = min(float(np.asarray(x0) @ q @ np.asarray(x0)) + meta["constant"]
                       for s in range(1 << n_slack)
                       for x0 in [x + np.append(np.zeros(6),
                                                [(s >> b) & 1 for b in range(n_slack)])])
            oh, bad = qubo_violation(POOLS, cuts, x)
            self.assertEqual(oh, 0)
            self.assertEqual(best == 0.0, bad == 0, choice)

    def test_milp_cut_loop_recovers_pool_optimum(self):
        inst = fixture()
        report = run_candidate_loop(inst, POOLS, master="milp",
                                    max_iterations=30, log=None)
        self.assertEqual(report["best_makespan"], 11)
        self.assertEqual(report["certificate"], "pool_lower_bound:11")

    def test_kaiwu_master_evaluates_samples_without_claiming_proof(self):
        try:
            import kaiwu  # noqa: F401
        except ImportError:
            self.skipTest("kaiwu SDK not installed in this environment")
        inst = fixture()
        report = run_candidate_loop(inst, POOLS, master="kaiwu",
                                    max_iterations=30, log=None,
                                    sa_options={"iterations_per_t": 50,
                                                "size_limit": 30})
        self.assertIsNotNone(report["best_makespan"])
        self.assertLessEqual(report["best_makespan"], 16)
        self.assertGreaterEqual(report["best_makespan"], 11)
        self.assertGreater(report["evaluation_count"], 1)
        self.assertGreater(sum(s.get("master_valid_samples", 0) for s in report["trace"]), 0)
        self.assertIsNone(report["proof_certificate"])

    def test_sgs_schedule_validity(self):
        inst = fixture()
        import numpy as np
        rng = np.random.default_rng(0)
        for rule in ("spt", "lpt", "mwr", "est", "random"):
            starts, makespan = serial_sgs(inst, rule, rng)
            self.assertEqual(validate_schedule(inst, starts), makespan)

    def test_qubo_matches_squared_residuals_for_every_binary_state(self):
        pool = (((0,), (1,)), ((2,), (3,)))
        cuts = [{"support": (0, 1), "allowed": {0: (0,), 1: (0,)}},
                {"support": (0,), "allowed": {0: (1,)}}]
        q, meta = build_qubo(pool, cuts)
        for bits in product((0, 1), repeat=meta["total_vars"]):
            x = np.array(bits)
            direct = ((x[0] + x[1] - 1) ** 2
                      + (x[2] + x[3] - 1) ** 2
                      + (x[0] + x[2] + x[4] - 1) ** 2 + x[1] ** 2)
            self.assertEqual(float(x @ q @ x + meta["constant"]), direct, bits)
        self.assertTrue(np.allclose(q, np.triu(q)))

    def test_three_machine_slack_range(self):
        pool = (((0,), (1,)),) * 3
        cut = {"support": (0, 1, 2), "allowed": {m: (0,) for m in range(3)}}
        q, meta = build_qubo(pool, [cut])
        self.assertEqual(meta["slack_bits"], [2])
        for bits in product((0, 1), repeat=8):
            x = np.array(bits)
            direct = sum((x[2*m] + x[2*m+1] - 1) ** 2 for m in range(3))
            direct += (x[0] + x[2] + x[4] + x[6] + 2*x[7] - 2) ** 2
            self.assertEqual(x @ q @ x + meta["constant"], direct)

    def test_always_active_cut_and_path_thresholds(self):
        inst = fixture()
        pool = tuple((row[0],) for row in POOLS)
        witness = separate(inst, decode(inst, [row[0] for row in pool]))
        self.assertEqual(witness.length, 16)
        for target, count in ((15, 1), (16, 0), (17, 0)):
            cuts, dropped = compile_cuts(pool, [witness], target)
            self.assertEqual((len(cuts), dropped), (count, 0))
        cuts, _ = compile_cuts(pool, [witness], 15)
        self.assertEqual(cuts[0]["support"], ())
        q, meta = build_qubo(pool, cuts)
        x = np.ones(3)
        self.assertEqual(x @ q @ x + meta["constant"], 1)
        self.assertEqual(solve_master_milp(pool, cuts)[0], "infeasible")
        report = run_candidate_loop(inst, pool, log=None)
        self.assertEqual(report["proof_certificate"], "pool_lower_bound:16")

    def test_never_active_witness_is_dropped(self):
        pool = tuple((row[0],) for row in POOLS)
        w = Witness("path", ((0, 7, 0),), 20)
        self.assertEqual(compile_cuts(pool, [w], 10), ([], 1))

    def test_milp_status_and_incumbent_validation(self):
        valid = np.array([1, 0, 1, 0, 1, 0])
        cases = [(0, valid, "optimal"), (1, valid, "limit_with_incumbent"),
                 (1, None, "unknown"), (2, None, "infeasible"),
                 (4, None, "error"), (1, np.full(6, .5), "unknown"),
                 (0, np.ones(6), "unknown"), (0, np.full(6, np.nan), "unknown"),
                 (0, np.full(6, 2), "unknown")]
        for code, x, expected in cases:
            with self.subTest(code=code, expected=expected), patch(
                    "scipy.optimize.milp", return_value=SimpleNamespace(status=code, x=x)) as call:
                status, choice = solve_master_milp(POOLS, [], time_limit=.2)
                self.assertEqual(status, expected)
                self.assertEqual(call.call_args.kwargs["options"]["time_limit"], .2)
                self.assertEqual(choice, [0, 0, 0] if expected in (
                    "optimal", "limit_with_incumbent") else None)

    def test_milp_rejects_cut_violating_incumbent(self):
        cuts = [{"support": (0,), "allowed": {0: (0,)}}]
        with patch("scipy.optimize.milp", return_value=SimpleNamespace(
                status=1, x=np.array([1, 0, 1, 0, 1, 0]))):
            self.assertEqual(solve_master_milp(POOLS, cuts), ("unknown", None))

    def test_timeout_never_proves_pool_optimality(self):
        with patch("medium_qjsp.solve_master_milp", return_value=("unknown", None)) as call:
            report = run_candidate_loop(fixture(), POOLS, time_limit=.5, log=None)
        self.assertIsNone(report["proof_certificate"])
        self.assertEqual(report["stop_reason"], "master_unknown")
        self.assertEqual(report["best_makespan"], 16)
        self.assertLessEqual(call.call_args.kwargs["time_limit"], .5)

    def test_invalid_one_hot_is_not_repaired(self):
        for bits in ((0, 0, 1, 0, 1, 0), (1, 1, 1, 0, 1, 0),
                     (.5, .5, 1, 0, 1, 0)):
            with self.assertRaises(ValueError):
                one_hot_choice(POOLS, bits)
        with patch("medium_qjsp.solve_qubo_kaiwu", return_value=[(0, (1, 1, 1, 0, 1, 0))]):
            report = run_candidate_loop(fixture(), POOLS, master="kaiwu",
                                        max_iterations=1, log=None)
        self.assertEqual(report["best_makespan"], 16)
        self.assertEqual(report["evaluation_count"], 1)
        self.assertEqual(report["trace"][0]["illegal_one_hot_fraction"], 1)
        self.assertIsNone(report["proof_certificate"])

    def test_budget_exit_retains_verified_incumbent(self):
        for master in ("milp", "kaiwu"):
            report = run_candidate_loop(fixture(), POOLS, master=master,
                                        time_limit=0, log=None)
            self.assertEqual(report["best_makespan"], 16)
            self.assertEqual(validate_schedule(fixture(), report["best_starts"]), 16)
            self.assertIsNone(report["proof_certificate"])
            self.assertEqual(report["stop_reason"], "time_limit")

    def test_annealing_error_retains_incumbent(self):
        with patch("medium_qjsp.solve_qubo_kaiwu", side_effect=RuntimeError("fixture failure")):
            report = run_candidate_loop(fixture(), POOLS, master="kaiwu", log=None)
        self.assertEqual(report["stop_reason"], "annealing_error")
        self.assertEqual(report["best_makespan"], 16)
        self.assertIsNone(report["proof_certificate"])

    def test_variable_cap_precedes_allocation(self):
        with patch("medium_qjsp.np.zeros") as zeros:
            with self.assertRaises(ValueError):
                build_qubo(POOLS, [], max_vars=5)
            zeros.assert_not_called()
        report = run_candidate_loop(fixture(), POOLS, master="kaiwu",
                                    max_qubo_vars=5, log=None)
        self.assertEqual(report["stop_reason"], "resource_limit")
        self.assertEqual(report["best_makespan"], 16)

    def test_invalid_pools_rejected(self):
        invalid = [POOLS[:-1], ((),) + POOLS[1:],
                   ((POOLS[0][0], POOLS[0][0]),) + POOLS[1:],
                   (((0, 5, 5),),) + POOLS[1:],
                   (((0., 5., 7.),),) + POOLS[1:]]
        for pool in invalid:
            with self.assertRaises(ValueError):
                validate_pool(fixture(), pool)

    def test_pool_builder_retains_source_schedule(self):
        pool, info = build_pool(fixture(), candidates_per_machine=2, n_schedules=20)
        result = decode(fixture(), [p[0] for p in pool])
        self.assertEqual(result["makespan"], info["makespan"])
        self.assertEqual(validate_schedule(fixture(), result["starts"]), info["makespan"])

    def test_gauge_decoding_and_sdk_keyword_arguments(self):
        q, meta = build_qubo((((0,), (1,)),), [])
        spins = np.array([[1, -1, 1], [-1, 1, -1], [-1, 1, 1]])
        samples = decode_ising_samples(spins, q, meta)
        self.assertEqual(samples, [(0., (1, 0)), (0., (0, 1))])
        self.assertEqual(decode_ising_samples(None, q, meta), [])
        with self.assertRaises(ValueError):
            decode_ising_samples([[0, 1, 1]], q, meta)
        try:
            import kaiwu
        except ImportError:
            self.skipTest("kaiwu SDK not installed")
        with patch.object(kaiwu, "SimulatedAnnealingOptimizer") as optimizer:
            optimizer.return_value.solve.return_value = spins
            self.assertEqual(solve_qubo_kaiwu(q, meta), samples)
            self.assertEqual(optimizer.return_value.solve.call_args.kwargs,
                             {"negtail_flip": True, "sort_solutions": True})

    def test_sdk_conversion_preserves_all_binary_energies(self):
        try:
            from kaiwu import qubo_matrix_to_ising_matrix
        except ImportError:
            self.skipTest("kaiwu SDK not installed")
        pool = (((0,), (1,)), ((2,), (3,)))
        cuts = [{"support": (0, 1), "allowed": {0: (0,), 1: (0,)}}]
        q, meta = build_qubo(pool, cuts)
        original = q.copy()
        ising, bias = qubo_matrix_to_ising_matrix(q)
        np.testing.assert_array_equal(q, original)
        for bits in product((0, 1), repeat=meta["total_vars"]):
            x = np.array(bits)
            spins = np.append(2 * x - 1, 1)
            for gauge in (1, -1):
                s = gauge * spins
                self.assertAlmostEqual(x @ q @ x + meta["constant"],
                                       -s @ ising @ s + bias + meta["constant"])

    def test_saved_result_verifier_and_corruption_detection(self):
        report = run_candidate_loop(fixture(), POOLS, log=None)
        report["instance_sha256"] = content_hash(report["instance_data"])
        self.assertEqual(verify_report(report)["makespan"], 11)
        mutations = [lambda r: r["best_starts"].__setitem__(0, -1),
                     lambda r: r.__setitem__("best_makespan", 12),
                     lambda r: r["instance_data"]["durations"][0].__setitem__(0, 4),
                     lambda r: r["pool_data"][0][0].reverse(),
                     lambda r: r["best_orders"][0].reverse(),
                     lambda r: r.__setitem__("best_choice", [2, 0, 0])]
        # Normalize immutable pool tuples as they appear in saved JSON.
        import json
        report = json.loads(json.dumps(report))
        for mutate in mutations:
            corrupted = deepcopy(report)
            mutate(corrupted)
            with self.assertRaises(ValueError):
                verify_report(corrupted)


if __name__ == "__main__":
    unittest.main()
