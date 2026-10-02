import unittest

import numpy as np
from qiskit.quantum_info import Operator, Statevector

from circuits import (CandidatePool, JointTransition, WitnessSpec,
                      build_joint_mixer, build_phase_circuit, build_xy_mixer,
                      circuit_resources, decode_one_hot_counts,
                      ensure_simulation_budget)


POOL = CandidatePool((( (0, 1), (1, 0) ), ((2, 3), (3, 2))))


class GateContractTests(unittest.TestCase):
    def test_pool_identity_and_stale_witness_rejection(self):
        witness = WitnessSpec("cycle", ((0,), (0,)), pool_hash=POOL.content_hash)
        witness.validate(POOL)
        with self.assertRaises(ValueError):
            WitnessSpec("cycle", ((0,), (0,)), pool_hash="stale").validate(POOL)

    def test_fixed_threshold_phase_matches_predicate_and_uncomputes(self):
        witness = WitnessSpec("cycle", ((0,), (0,)), pool_hash=POOL.content_hash)
        circuit = build_phase_circuit(POOL, [witness], phase_angle=0.37,
                                      fixed_t=0)
        self.assertGreater(circuit.metadata["work_qubits"], 0)
        # Legal choices are labels (0,0), (0,1), (1,0), (1,1).  Predicate is
        # true only for (0,0), so its amplitude gets the requested phase.
        for choice in ((0, 0), (0, 1), (1, 0), (1, 1)):
            state = Statevector.from_int(0, 1 << circuit.num_qubits)
            for machine, label in enumerate(choice):
                state = state.evolve(_basis_preparation(circuit, POOL, machine, label))
            result = state.evolve(circuit)
            basis_index = sum(1 << POOL.offset(machine, label)
                              for machine, label in enumerate(choice))
            expected = np.exp(-1j * 0.37) if choice == (0, 0) else 1.0
            self.assertAlmostEqual(abs(result.data[basis_index] - expected), 0.0, places=6)

    def test_path_threshold_register_and_objective_compile(self):
        witness = WitnessSpec("path", ((0,), (0,)), length=3,
                              pool_hash=POOL.content_hash)
        circuit = build_phase_circuit(POOL, [witness], phase_angle=0.2,
                                      time_bits=2, include_objective=True,
                                      penalty=9)
        self.assertEqual(circuit.metadata["time_qubits"], 2)
        self.assertGreater(circuit.count_ops().get("p", 0), 0)
        self.assertEqual(circuit.metadata["hardware_jobs_submitted"], 0)

    def test_register_time_threshold_boundaries(self):
        witness = WitnessSpec("path", ((0,), (0,)), length=3,
                              pool_hash=POOL.content_hash)
        circuit = build_phase_circuit(POOL, [witness], phase_angle=0.2,
                                      time_bits=2)
        for time_value, active in ((2, True), (3, False)):
            preparation = _basis_preparation(circuit, POOL, 0, 0)
            preparation.x(POOL.offset(1, 0))
            for bit in range(2):
                if time_value & (1 << bit):
                    preparation.x(POOL.data_qubits + bit)
            result = Statevector.from_int(0, 1 << circuit.num_qubits)
            result = result.evolve(preparation).evolve(circuit)
            index = ((1 << POOL.offset(0, 0)) | (1 << POOL.offset(1, 0))
                     | (time_value << POOL.data_qubits))
            expected = np.exp(-1j * 0.2) if active else 1.0
            self.assertAlmostEqual(abs(result.data[index] - expected), 0.0, places=6)

    def test_simulation_budget_rejects_medium_circuit(self):
        circuit = build_xy_mixer(CandidatePool(
            tuple(((0,), (1,)) for _ in range(11))
        ))
        with self.assertRaises(ValueError):
            ensure_simulation_budget(circuit, max_qubits=20)

    def test_mixers_are_unitary_and_resource_metadata_is_offline(self):
        xy = build_xy_mixer(POOL, ring=True)
        self.assertTrue(np.allclose(Operator(xy).data.conj().T @ Operator(xy).data,
                                    np.eye(2 ** POOL.data_qubits), atol=1e-8))
        transition = JointTransition((0, 1), (0, 0), (1, 1), "w0")
        joint = build_joint_mixer(POOL, [transition], theta=0.11)
        self.assertTrue(np.allclose(Operator(joint).data.conj().T @ Operator(joint).data,
                                    np.eye(2 ** POOL.data_qubits), atol=1e-8))
        self.assertEqual(circuit_resources(joint)["hardware_jobs_submitted"], 0)

    def test_joint_transition_only_moves_two_legal_endpoints(self):
        transition = JointTransition((0, 1), (0, 0), (1, 1), "w0")
        circuit = build_joint_mixer(POOL, [transition], theta=0.4)
        before = np.zeros(1 << POOL.data_qubits, dtype=complex)
        left = (1 << POOL.offset(0, 0)) | (1 << POOL.offset(1, 0))
        right = (1 << POOL.offset(0, 1)) | (1 << POOL.offset(1, 1))
        before[left] = 1.0
        out = Statevector(before).evolve(circuit).data
        self.assertAlmostEqual(abs(out[left]), np.cos(0.4), places=6)
        self.assertAlmostEqual(abs(out[right]), np.sin(0.4), places=6)

    def test_count_decoder_rejects_illegal_without_repair(self):
        # Qiskit count strings are displayed most-significant bit first.
        result = decode_one_hot_counts({"0101": 2, "0011": 3}, POOL)
        self.assertEqual(result["illegal_count"], 3)
        self.assertEqual(result["legal"][( (0, 0), 0)], 2)


def _basis_preparation(circuit, pool, machine, label):
    from qiskit import QuantumCircuit
    preparation = QuantumCircuit(circuit.num_qubits)
    preparation.x(pool.offset(machine, label))
    return preparation


if __name__ == "__main__":
    unittest.main()
