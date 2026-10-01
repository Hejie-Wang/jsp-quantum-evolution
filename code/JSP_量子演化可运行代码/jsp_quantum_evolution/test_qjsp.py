"""Mathematical and end-to-end checks; no large-instance speedup assumption."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from qjsp import (CapacityError, Instance, PermutationModel, demo_instance,
                  evaluate_orders, evolve, sample_schedule, solve, validate_schedule)


class EvolutionTests(unittest.TestCase):
    def test_fixed_orders_and_known_optimum(self):
        model = PermutationModel(demo_instance())
        self.assertEqual(model.dimension, 216)
        self.assertEqual(int((model.costs < model.instance.penalty).sum()), 64)
        self.assertEqual(int(model.costs.min()), 11)
        for i, cost in enumerate(model.costs):
            result = evaluate_orders(model.instance, model.orders_at(i))
            self.assertEqual(result["cost"], cost)
            if result["feasible"]:
                self.assertEqual(validate_schedule(model.instance, result["starts"]), cost)

    def test_swap_involutions_and_ground_state(self):
        model = PermutationModel(demo_instance())
        rng = np.random.default_rng(31)
        psi = rng.normal(size=model.dimension) + 1j*rng.normal(size=model.dimension)
        for swap in model.swaps:
            np.testing.assert_array_equal(model.swap_state(model.swap_state(psi, swap), swap), psi)
        u = model.uniform_state()
        np.testing.assert_allclose(model.apply_hamiltonian(u, 0), -u, atol=1e-14)
        h = model.dense_hamiltonian(0.37)
        np.testing.assert_allclose(h, h.conj().T, atol=1e-14)

    def test_product_layer_against_exact_matrix_exponential(self):
        # 3 jobs on 2 machines has noncommuting swaps within each machine.
        model = PermutationModel(demo_instance(3, 2))
        rng = np.random.default_rng(12)
        psi = rng.normal(size=model.dimension) + 1j*rng.normal(size=model.dimension)
        psi /= np.linalg.norm(psi)
        s = 0.41
        hamiltonian = model.dense_hamiltonian(s)
        errors = []
        for h in (0.002, 0.001):
            actual = model.apply_layer(psi, s, h)
            reference = expm(-1j*h*hamiltonian) @ psi
            self.assertLess(abs(np.linalg.norm(actual)-1), 1e-13)
            errors.append(np.linalg.norm(actual-reference))
        # First-order splitting has O(h^2) local error.
        self.assertGreater(errors[0]/errors[1], 3.8)
        self.assertLess(errors[0]/errors[1], 4.2)

    def test_full_evolution_converges_to_schrodinger_ode(self):
        model = PermutationModel(demo_instance(2, 2))
        tau, final = 2.0, 0.95
        reference = solve_ivp(lambda t, psi: -1j*model.apply_hamiltonian(psi, final*t/tau),
                              (0, tau), model.uniform_state(), method="DOP853",
                              rtol=1e-11, atol=1e-13).y[:, -1]
        errors = []
        for layers in (1000, 2000):
            psi, _ = evolve(model, tau=tau, layers=layers, s_final=final)
            errors.append(np.linalg.norm(psi-reference))
        self.assertLess(errors[1], 0.002)
        self.assertGreater(errors[0]/errors[1], 1.8)
        self.assertLess(errors[0]/errors[1], 2.2)

    def test_capacity_check_precedes_permutation_enumeration(self):
        inst = demo_instance(50, 20)
        with patch("qjsp.permutations", side_effect=AssertionError("must not enumerate")):
            with self.assertRaises(CapacityError):
                PermutationModel(inst)
        with self.assertRaises(CapacityError):
            PermutationModel(demo_instance(), max_memory_mib=0.0001)

    def test_sampling_does_not_return_unmeasured_optimum(self):
        model = PermutationModel(demo_instance())
        feasible_nonoptimal = np.flatnonzero((model.costs < model.instance.penalty) & (model.costs > 11))[0]
        psi = np.zeros(model.dimension, dtype=complex)
        psi[feasible_nonoptimal] = 1
        sampled = sample_schedule(model, psi, shots=17)
        self.assertEqual(sampled["best_measured_schedule"]["basis_index"], feasible_nonoptimal)
        self.assertEqual(sampled["optimal_shots_diagnostic"], 0)
        invalid = int(np.flatnonzero(model.costs == model.instance.penalty)[0])
        psi[:] = 0
        psi[invalid] = 1
        self.assertIsNone(sample_schedule(model, psi)["best_measured_schedule"])

    def test_end_to_end_reproducible_and_valid(self):
        result, psi, model = solve(demo_instance(), tau=8, layers=800, shots=500, seed=3)
        sampled = sample_schedule(model, psi, shots=500, seed=3)
        self.assertEqual(result["sampling"], sampled)
        schedule = sampled["best_measured_schedule"]
        self.assertIsNotNone(schedule)
        self.assertEqual(validate_schedule(model.instance, schedule["starts"]), schedule["makespan"])

    def test_one_job_and_bad_inputs(self):
        inst = Instance(np.array([[2, 3]]), np.array([[0, 1]]))
        result, _, _ = solve(inst, tau=1, layers=4, shots=1)
        self.assertEqual(result["sampling"]["best_measured_schedule"]["makespan"], 5)
        with self.assertRaises(ValueError):
            Instance(np.array([[1, 2]]), np.array([[0, 0]]))
        with self.assertRaises(ValueError):
            Instance(np.array([[1.5]]), np.array([[0]]))
        with self.assertRaises(ValueError):
            evaluate_orders(demo_instance(), [[0], [1], [2]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
