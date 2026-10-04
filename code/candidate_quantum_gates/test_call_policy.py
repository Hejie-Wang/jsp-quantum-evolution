#!/usr/bin/env python3
"""Unit tests for the T07 call-policy module.

Run inside the qskit environment:

    python code/candidate_quantum_gates/test_call_policy.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

GATES_DIR = Path(__file__).resolve().parent
if str(GATES_DIR) not in sys.path:
    sys.path.insert(0, str(GATES_DIR))

import call_policy as cp  # noqa: E402


class PolicyBehaviourTests(unittest.TestCase):
    def test_fixed_frequency_calls_quantum_every_period(self):
        policy = cp.FixedFrequencyPolicy(0.001, 0.5, 0.1, 0.1)
        actions = [policy.next_action(i, 0) for i in range(cp.FIXED_PERIOD * 2)]
        self.assertEqual(actions.count("quantum"), 2)
        self.assertEqual(actions[cp.FIXED_PERIOD - 1], "quantum")

    def test_stagnation_calls_quantum_after_k_misses(self):
        policy = cp.StagnationPolicy(0.001, 0.5, 0.1, 0.1)
        self.assertEqual(policy.next_action(0, cp.STAGNATION_K - 1), "classical")
        self.assertEqual(policy.next_action(0, cp.STAGNATION_K), "quantum")

    def test_disabled_policy_never_calls_quantum(self):
        policy = cp.DisabledPolicy(0.001, 0.5, 0.1, 0.1)
        self.assertEqual({policy.next_action(i, i) for i in range(20)}, {"classical"})

    def test_adaptive_disables_a_useless_module_from_observed_history(self):
        policy = cp.AdaptivePolicy(0.001, 0.5, 0.1, 0.0001, epsilon=0.0)
        # 24 observed quantum calls with a single hit is far below the classical
        # action's hits-per-second, so the module must be disabled
        policy.stats["quantum"] = [2, 25]
        policy.stats["classical"] = [2, 25]
        self.assertEqual(policy.next_action(0, 0), "classical")
        self.assertTrue(policy.disabled)
        self.assertEqual(policy.next_action(0, 0), "classical")

    def test_adaptive_never_calls_a_strictly_worse_module_unless_exploring(self):
        policy = cp.AdaptivePolicy(0.001, 0.5, 0.1, 0.0001, epsilon=0.0)
        actions = set()
        for _ in range(50):
            action = policy.next_action(0, 0)
            actions.add(action)
            policy.record(action, False)
        self.assertEqual(actions, {"classical"})

    def test_adaptive_prefers_the_hitting_action(self):
        policy = cp.AdaptivePolicy(0.001, 0.001, 0.0, 1.0, epsilon=0.0)
        policy.record("quantum", True)
        policy.record("classical", False)
        self.assertEqual(policy.next_action(0, 0), "quantum")


class SimulationTests(unittest.TestCase):
    def test_simulation_stops_at_the_first_hit_and_accounts_cost(self):
        policy = cp.DisabledPolicy(0.01, 1.0, 1.0, 0.0)
        row = cp.simulate(policy, 1.0, 0.0, budget=10.0, seed=0)
        self.assertTrue(row["hit"])
        self.assertAlmostEqual(row["arrival_seconds"], 0.01, places=9)
        self.assertEqual(row["quantum_calls"], 0)
        self.assertEqual(row["quantum_time_share"], 0.0)

    def test_budget_truncation_is_reported_as_failure(self):
        policy = cp.DisabledPolicy(0.01, 1.0, 0.0, 0.0)
        row = cp.simulate(policy, 0.0, 0.0, budget=0.05, seed=0)
        self.assertFalse(row["hit"])
        self.assertIsNone(row["arrival_seconds"])
        self.assertLessEqual(row["elapsed_seconds"], 0.06)

    def test_evaluate_policy_reports_rates_and_shares(self):
        result = cp.evaluate_policy(cp.FixedFrequencyPolicy, 0.005, 0.001, 0.01, 0.2,
                                    repetitions=30, budget=20.0, seed=1)
        self.assertEqual(result["policy"], "fixed_frequency")
        self.assertGreater(result["success_rate"], 0.0)
        self.assertGreater(result["mean_quantum_share"], 0.0)
        self.assertEqual(result["repetitions"], 30)

    def test_negative_control_window_is_skipped_by_the_runner(self):
        # load_t06_arm_probabilities must expose the measured probabilities
        probabilities = cp.load_t06_arm_probabilities(
            str(GATES_DIR / "results_sampling_ablation_20261003"
                / "sampling_ablation.json"), arm="xy_joint", training="exact")
        self.assertIn(("3x3", 2), probabilities)
        entry = probabilities[("3x3", 2)]
        self.assertAlmostEqual(entry["p_classical"], 0.14814814814814814, places=12)
        self.assertGreater(entry["p_quantum"], 0.0)
        self.assertIn(("3x3", 0), probabilities)  # negative control is present


class CostTests(unittest.TestCase):
    def test_paired_bootstrap_returns_interval(self):
        result = cp.paired_bootstrap([1.0, 1.1, 0.9, 1.05], samples=500)
        self.assertLessEqual(result["ci95"][0], result["mean"])
        self.assertGreaterEqual(result["ci95"][1], result["mean"])
        self.assertIsNone(cp.paired_bootstrap([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
