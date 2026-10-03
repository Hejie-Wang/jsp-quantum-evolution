#!/usr/bin/env python3
"""Unit tests for the T06 sampling-ablation module.

Run inside the qskit environment:

    python code/candidate_quantum_gates/test_sampling_ablation.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

GATES_DIR = Path(__file__).resolve().parent
if str(GATES_DIR) not in sys.path:
    sys.path.insert(0, str(GATES_DIR))

import data_identity  # noqa: E402
import sampling_ablation as sa  # noqa: E402
from window_diagnostics import build_window  # noqa: E402


def window(seed=2, jobs=3, machines=3, k=3):
    inst = data_identity.build_d0(jobs, machines, seed)
    pool, incumbent, u0 = build_window(inst, k, seed + 1000)
    return inst, pool, incumbent, u0


class SharedSetupTests(unittest.TestCase):
    def test_setup_is_shared_and_frozen(self):
        inst, pool, incumbent, u0 = window()
        setup = sa.window_setup(inst, pool, incumbent, u0, seed=0)
        self.assertEqual(setup["target_t"], u0 - 1)
        self.assertEqual(setup["pool_obj"].sizes, tuple(len(m) for m in pool))
        self.assertGreater(len(setup["specs"]), 0)
        # the same setup object is what every arm consumes
        reference = sa.exact_reference(inst, pool, u0, setup["target_t"],
                                       specs=setup["specs"])
        self.assertEqual(reference["graph_evaluations"], 27)


class ReferenceTests(unittest.TestCase):
    def test_uniform_arm_matches_the_offline_reference(self):
        inst, pool, incumbent, u0 = window()
        setup = sa.window_setup(inst, pool, incumbent, u0, seed=0)
        reference = sa.exact_reference(inst, pool, u0, setup["target_t"],
                                       specs=setup["specs"])
        arm = sa.uniform_arm(pool, inst, u0, setup["target_t"])
        self.assertAlmostEqual(arm["p_hit"], reference["p_hit"], places=12)
        self.assertEqual(arm["graph_evaluations_per_hit"],
                         1.0 / reference["p_hit"])
        self.assertEqual(arm["training_shots"], 0)

    def test_surrogate_alignment_is_counted(self):
        inst, pool, incumbent, u0 = window()
        setup = sa.window_setup(inst, pool, incumbent, u0, seed=0)
        reference = sa.exact_reference(inst, pool, u0, setup["target_t"],
                                       specs=setup["specs"])
        self.assertIsNotNone(reference["improving_with_zero_witness_energy"])
        self.assertLessEqual(reference["improving_with_zero_witness_energy"],
                             reference["improving_combinations"])


class ArmTests(unittest.TestCase):
    def test_variational_arm_respects_the_training_budget(self):
        inst, pool, incumbent, u0 = window()
        setup = sa.window_setup(inst, pool, incumbent, u0, seed=0)
        exact = sa.variational_arm(inst, pool, setup, u0, mode="xy",
                                   training="exact", train_evaluations=6,
                                   shots=64, seed=1)
        self.assertEqual(exact["parameter_evaluations"], 6)
        self.assertEqual(exact["training_shots"], 0)
        self.assertGreaterEqual(exact["p_hit"], 0.0)
        self.assertLessEqual(exact["p_hit"], 1.0)
        shot = sa.variational_arm(inst, pool, setup, u0, mode="xy",
                                  training="shots", train_evaluations=3,
                                  shots=32, seed=1)
        self.assertEqual(shot["parameter_evaluations"], 3)
        self.assertEqual(shot["training_shots"], 3 * 32)

    def test_joint_arm_uses_the_same_protocol_and_reports_both_shot_levels(self):
        inst, pool, incumbent, u0 = window()
        setup = sa.window_setup(inst, pool, incumbent, u0, seed=0)
        arm = sa.variational_arm(inst, pool, setup, u0, mode="xy_joint",
                                 training="exact", train_evaluations=4,
                                 shots=64, seed=2)
        self.assertEqual(arm["arm"], "xy_joint")
        self.assertEqual(arm["shot_estimates"]["main"]["shots"], 64)
        self.assertEqual(arm["shot_estimates"]["secondary"]["shots"],
                         sa.SECONDARY_SHOTS)
        self.assertIn("expected_witness_energy", arm)

    def test_sa_arm_counts_evaluations_and_stays_in_range(self):
        inst, pool, incumbent, u0 = window()
        setup = sa.window_setup(inst, pool, incumbent, u0, seed=0)
        arm = sa.sa_arm(inst, pool, setup, u0, runs=4, steps=25, seed=3)
        self.assertGreaterEqual(arm["p_hit"], 0.0)
        self.assertLessEqual(arm["p_hit"], 1.0)
        self.assertEqual(arm["runs"], 4)
        self.assertGreaterEqual(arm["witness_evaluations_per_run"], 25)
        self.assertEqual(arm["training_shots"], 0)

    def test_negative_control_window_has_no_hits_anywhere(self):
        inst, pool, incumbent, u0 = window(seed=0)
        setup = sa.window_setup(inst, pool, incumbent, u0, seed=0)
        reference = sa.exact_reference(inst, pool, u0, setup["target_t"],
                                       specs=setup["specs"])
        self.assertEqual(reference["improving_combinations"], 0)
        self.assertEqual(sa.uniform_arm(pool, inst, u0, setup["target_t"])["p_hit"], 0.0)
        arm = sa.variational_arm(inst, pool, setup, u0, mode="xy",
                                 training="exact", train_evaluations=4,
                                 shots=32, seed=4)
        self.assertEqual(arm["p_hit"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
