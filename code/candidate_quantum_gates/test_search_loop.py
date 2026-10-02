"""Acceptance tests for the P2 fixed-T quantum-assisted closed loop.

Covers docs/candidate_quantum_improvements_20261002.md P2: the multilayer
circuit changes the sampling distribution on a non-degenerate small example,
every reported schedule passes the independent full graph evaluation, the
outer best makespan never worsens, joint actions are derived from witnesses
only (no optimal-label access, no mechanical first-candidate moves), budgets
are recorded, and sampling failure is never treated as infeasibility.
"""
import unittest

from itertools import product

from circuits import CandidatePool, WitnessSpec
from search_loop import (build_fixed_t_ansatz, classical_joint_search,
                         covering_demo_witnesses, data_probabilities,
                         demo_candidate_pool, demo_instance,
                         graph_witness_specs, propose_joint_actions,
                         run_search_loop, witness_active)

DEMO_POOL = demo_candidate_pool()
DEMO_INST = demo_instance()


def _enumerate_optimum():
    import sys
    from pathlib import Path
    medium = Path(__file__).resolve().parents[1] / "candidate_quantum_medium"
    if str(medium) not in sys.path:
        sys.path.insert(0, str(medium))
    import medium_qjsp as mq
    best = None
    for choice in product(*[range(s) for s in DEMO_POOL.sizes]):
        result = mq.decode(DEMO_INST,
                           [list(DEMO_POOL.candidates[m][a])
                            for m, a in enumerate(choice)])
        if result["feasible"]:
            best = (result["makespan"] if best is None
                    else min(best, result["makespan"]))
    return best


class JointActionTests(unittest.TestCase):
    def test_actions_derived_from_witnesses_only(self):
        raw, specs = covering_demo_witnesses(DEMO_POOL)
        incumbent = [0] * DEMO_POOL.machines
        actions = propose_joint_actions(DEMO_POOL, specs, incumbent)
        self.assertTrue(actions)
        self.assertIsNotNone(_enumerate_optimum())
        for action in actions:
            self.assertLessEqual(len(action.support), 4)
            self.assertGreaterEqual(len(action.support), 2)
            self.assertLess(action.support[0], action.support[1])
            witness = specs[int(action.witness_id[1:])]
            for machine, left, right in zip(action.support, action.left,
                                            action.right):
                self.assertIn(left, witness.allowed[machine])
                self.assertNotIn(right, witness.allowed[machine])
                # left prefers the incumbent label when it lies inside the
                # witness; no other information (e.g. optimal labels) enters.
                if incumbent[machine] in witness.allowed[machine]:
                    self.assertEqual(left, incumbent[machine])

    def test_actions_break_the_triggering_witness(self):
        raw, specs = covering_demo_witnesses(DEMO_POOL)
        incumbent = [0] * DEMO_POOL.machines
        for action in propose_joint_actions(DEMO_POOL, specs, incumbent):
            witness = specs[int(action.witness_id[1:])]
            left_full = list(incumbent)
            moved = list(incumbent)
            for machine, left, right in zip(action.support, action.left,
                                            action.right):
                left_full[machine] = left
                moved[machine] = right
            # This witness triggers on the left endpoint and stops triggering
            # once every support machine moves to its right label at once.
            self.assertTrue(witness_active(witness, tuple(left_full), 15))
            self.assertFalse(witness_active(witness, tuple(moved), 15))


class ClosedLoopTests(unittest.TestCase):
    def test_phase_plus_mixer_changes_sampling_distribution(self):
        raw, specs = covering_demo_witnesses(DEMO_POOL)
        effective = [s for s in specs
                     if any(0 < len(a) < size for a, size in
                            zip(s.allowed, DEMO_POOL.sizes))]
        penalty = float(DEMO_INST.total_duration + 1)
        flat = build_fixed_t_ansatz(DEMO_POOL, effective, (), [], 15, penalty,
                                    mode="uniform")
        mixed = build_fixed_t_ansatz(DEMO_POOL, effective, (),
                                     [(0.31, 0.25, 0.25)], 15, penalty)
        uniform_probs, _ = data_probabilities(flat, DEMO_POOL)
        mixed_probs, _ = data_probabilities(mixed, DEMO_POOL)
        self.assertAlmostEqual(sum(uniform_probs.values()), 1.0, places=8)
        self.assertAlmostEqual(sum(mixed_probs.values()), 1.0, places=8)
        self.assertFalse(
            all(abs(uniform_probs[c] - mixed_probs[c]) < 1e-12
                for c in uniform_probs))

    def test_closed_loop_never_worsens_and_verifies_every_schedule(self):
        report = run_search_loop(DEMO_INST, DEMO_POOL, [0] * 3, mode="xy_joint",
                                 shots=256, seed=7, max_rounds=4)
        self.assertLessEqual(report["best_makespan"], report["initial_makespan"])
        self.assertTrue(report["independent_schedule_verification"])
        self.assertEqual(report["best_makespan"], _enumerate_optimum())
        self.assertGreater(report["evaluation_count"], 0)
        self.assertGreater(report["training_evaluations"], 0)
        self.assertGreater(report["training_seconds"], 0.0)
        for entry in report["trace"]:
            self.assertLessEqual(entry["circuit"]["num_qubits"], 20)
            self.assertIn("illegal_fraction", entry)
        self.assertTrue(any(note.startswith("Sampling failures never prove")
                            for note in report["notes"]))

    def test_uniform_and_xy_modes_run_within_budget(self):
        for mode in ("uniform", "xy"):
            report = run_search_loop(DEMO_INST, DEMO_POOL, [0] * 3, mode=mode,
                                     shots=128, seed=7, max_rounds=2)
            self.assertLessEqual(report["best_makespan"],
                                 report["initial_makespan"])
            self.assertTrue(report["independent_schedule_verification"])
            if mode == "uniform":
                self.assertEqual(report["training_evaluations"], 0)

    def test_sampling_failure_is_not_infeasibility(self):
        report = run_search_loop(DEMO_INST, DEMO_POOL, [0] * 3, mode="xy",
                                 shots=64, seed=7, max_rounds=1)
        joined = " ".join(report["notes"])
        self.assertIn("never prove infeasibility", joined)
        self.assertIn("not promised to decrease monotonically", joined)

    def test_classical_baseline_shares_information_and_actions(self):
        report = classical_joint_search(DEMO_INST, DEMO_POOL, [0] * 3,
                                        seed=7, max_rounds=3)
        self.assertLessEqual(report["best_makespan"], report["initial_makespan"])
        self.assertTrue(report["independent_schedule_verification"])
        self.assertEqual(report["mode"], "classical_joint")

    def test_large_pool_statevector_is_rejected(self):
        from circuits import JointTransition
        big = CandidatePool(tuple(((0, 1), (1, 0)) for _ in range(11)))
        effective = [WitnessSpec("cycle", ((0,),) * 11)]
        ansatz = build_fixed_t_ansatz(big, effective, (),
                                      [(0.1, 0.2, 0.2)], 3, 5.0)
        with self.assertRaises(ValueError):
            data_probabilities(ansatz, big)


if __name__ == "__main__":
    unittest.main()
