import unittest
import tempfile
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from bounded_qjsp import (Budget, apply_moves, build_circuit, check_compiled_budget,
                          dispatch_initial, fit_angles, hadamard_transform, local_cost_table,
                          phase_model, probabilities, run_request, simulate_search,
                          verify_compiled_probabilities)
from qjsp import demo_instance, evaluate_orders, validate_schedule


class BoundedTests(unittest.TestCase):
    def test_global_decoder_for_every_local_choice(self):
        inst = demo_instance(5, 5)
        orders = dispatch_initial(inst)
        moves = [(0, 0), (0, 2), (1, 1), (2, 0)]
        costs = local_cost_table(inst, orders, moves)
        for x in range(16):
            changed = apply_moves(orders, moves, x)
            result = evaluate_orders(inst, changed)
            self.assertEqual(costs[x], result['cost'])
            if result['feasible']:
                self.assertEqual(validate_schedule(inst, result['starts']), costs[x])
        with self.assertRaises(ValueError):
            apply_moves(orders, [(0,0),(0,1)], 0)

    def test_walsh_exact_reconstruction_and_gate_bound(self):
        rng = np.random.default_rng(4)
        for k in range(1, 7):
            phase = phase_model(rng.integers(1, 100, size=2**k))
            np.testing.assert_allclose(phase['normalized'], phase['approximation'], atol=1e-14)
            self.assertLessEqual(phase['cx_per_layer_before_routing'], (k-2)*2**k+2)

    def test_circuit_matches_independent_amplitudes(self):
        rng = np.random.default_rng(9)
        for k in (1, 2, 4, 6):
            phase = phase_model(rng.integers(1, 30, size=2**k))
            angles = rng.uniform(-2, 2, size=(2, 2))
            circuit = build_circuit(phase, angles)
            actual = Statevector.from_instruction(circuit).probabilities()
            np.testing.assert_allclose(actual, probabilities(phase['approximation'], angles), atol=1e-12)

    def test_truncated_phase_has_valid_probability_error_bound(self):
        rng = np.random.default_rng(3)
        costs = rng.integers(1, 30, size=16)
        exact, reduced = phase_model(costs), phase_model(costs, keep_terms=4)
        angles = np.array([[.11, .4], [.21, -.5]])
        p = probabilities(exact['approximation'], angles)
        q = probabilities(reduced['approximation'], angles)
        bound = np.abs(angles[:,0]).sum()*reduced['sup_error']
        self.assertLessEqual(.5*np.abs(p-q).sum(), bound+1e-12)

    def test_budget_enforcement(self):
        self.assertEqual(Budget(bits=6, max_two_qubit=200).chosen_bits(), 5)
        with self.assertRaises(ValueError): Budget(bits=7).validate()
        q = QuantumCircuit(2); q.cx(0,1); q.cx(0,1)
        with self.assertRaises(ValueError): check_compiled_budget(q, Budget(max_two_qubit=1))
        result, _ = simulate_search(demo_instance(), Budget(max_rounds=50,max_cost_evaluations=20), seed=3)
        self.assertLessEqual(result['local_table_cost_evaluations'], 20)
        self.assertEqual(result['hardware_jobs_submitted'], 0)

    def test_fit_uses_mean_cost_and_bounded_evaluations(self):
        phase = phase_model([31,18,32,31])
        angles, report = fit_angles(phase, evaluations=96)
        self.assertLessEqual(report['ideal_training_evaluations'],96)
        self.assertLessEqual(float(probabilities(phase['approximation'],angles)@phase['normalized']),
                             float(phase['normalized'].mean())+1e-12)

    def test_search_feasible_monotone_and_standalone_compile(self):
        inst=demo_instance()
        result, request=simulate_search(inst,Budget(max_rounds=6,max_cost_evaluations=96),seed=7)
        values=[result['initial_classical_makespan']]+[r['incumbent_after'] for r in result['trace']]
        self.assertTrue(all(b<=a for a,b in zip(values,values[1:])))
        self.assertEqual(validate_schedule(inst,result['schedule']['starts']),result['final_makespan'])
        with tempfile.TemporaryDirectory() as tmp:
            report,_=run_request(request,tmp)
            self.assertTrue(report['post_transpile_ideal_probabilities_checked'])

    def test_audit_prunes_idle_wires_and_respects_measurement_mapping(self):
        q=QuantumCircuit(156,2)
        q.x(39);q.h(102);q.cx(39,102)
        q.measure(39,1);q.measure(102,0)
        expected=np.array([0,0,.5,.5])
        self.assertLess(verify_compiled_probabilities(q,expected),1e-12)


if __name__=='__main__': unittest.main(verbosity=2)
