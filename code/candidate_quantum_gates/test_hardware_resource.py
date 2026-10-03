#!/usr/bin/env python3
"""Unit tests for the T09 hardware resource / noise report module.

Run inside the qskit environment:

    python code/candidate_quantum_gates/test_hardware_resource.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

GATES_DIR = Path(__file__).resolve().parent
if str(GATES_DIR) not in sys.path:
    sys.path.insert(0, str(GATES_DIR))

import data_identity  # noqa: E402
import hardware_resource as hr  # noqa: E402
from window_diagnostics import build_window  # noqa: E402
from witness_surrogate import collect_witnesses  # noqa: E402


def window(seed=0, k=3):
    inst = data_identity.build_d0(3, 3, seed)
    pool, incumbent, u0 = build_window(inst, k, seed + 1000)
    witnesses, _stats = collect_witnesses(inst, pool, incumbent, n_probe=8, seed=seed)
    return inst, pool, u0, witnesses


class SelectionTests(unittest.TestCase):
    def test_dq_proxy_has_six_slots_and_uses_classical_profile_only(self):
        seeds = range(6)
        selected, strata = hr.select_dq_proxy(seeds, pool_k=3, per_stratum=2)
        self.assertEqual(len(selected), 6)
        self.assertEqual(sum(strata.values()), 6 * len(data_identity.D0_SIZES))
        for entry in selected:
            self.assertIn("rho", entry["profile"])
            self.assertIsNotNone(entry["profile"]["combinations"])

    def test_stratification_rule_is_respected_for_available_strata(self):
        selected, _strata = hr.select_dq_proxy(range(6), pool_k=3, per_stratum=2)
        no_improvement = [e for e in selected if e["profile"]["improving"] == 0]
        self.assertGreaterEqual(len(no_improvement), 1)


class CircuitTests(unittest.TestCase):
    def test_sampling_circuit_layout_and_uniform_ideal_distribution(self):
        inst, pool, u0, witnesses = window()
        circuit, pool_obj = hr.build_sampling_circuit(pool, witnesses, u0 - 1)
        sizes = [len(machine) for machine in pool]
        self.assertEqual(pool_obj.sizes, tuple(sizes))
        self.assertEqual(circuit.metadata["data_qubits"], sum(sizes))
        self.assertEqual(circuit.metadata["work_qubits"], len(pool) + 3)
        self.assertTrue(circuit.metadata["work_register_reused"])
        probabilities = hr.ideal_probabilities(circuit, pool_obj.data_qubits)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=9)
        legal = np.prod(sizes)
        for bits, probability in enumerate(probabilities):
            valid = True
            for machine, size in enumerate(sizes):
                offset = sum(sizes[:machine])
                if sum((bits >> (offset + a)) & 1 for a in range(size)) != 1:
                    valid = False
                    break
            self.assertAlmostEqual(probability, 1.0 / legal if valid else 0.0, places=9)

    def test_choice_metrics_reproduce_the_pool_coverage(self):
        inst, pool, u0, witnesses = window(seed=2)
        circuit, pool_obj = hr.build_sampling_circuit(pool, witnesses, u0 - 1)
        probabilities = hr.ideal_probabilities(circuit, pool_obj.data_qubits)
        metrics = hr.choice_metrics(probabilities, pool, inst, u0, u0 - 1)
        profile = hr.structural_profile(inst, pool, u0)
        self.assertAlmostEqual(metrics["p_imp"], profile["rho"], places=9)
        self.assertAlmostEqual(metrics["illegal_one_hot_mass"], 0.0, places=9)
        self.assertAlmostEqual(metrics["legal_mass"], 1.0, places=9)


class NoiseTests(unittest.TestCase):
    def test_noise_only_degrades_and_leaks_illegal_mass(self):
        inst, pool, u0, witnesses = window(seed=2)
        circuit, pool_obj = hr.build_sampling_circuit(pool, witnesses, u0 - 1)
        data_bits = pool_obj.data_qubits
        ideal = hr.ideal_probabilities(circuit, data_bits)
        model, description = hr.gate_depolarizing_model(5e-3)
        counts = hr.sample_counts(circuit, data_bits, model, shots=4000, seed=3)
        noisy = hr.counts_to_probabilities(counts, data_bits)
        metrics = hr.choice_metrics(noisy, pool, inst, u0, u0 - 1)
        self.assertEqual(description["kind"], "gate_depolarizing")
        self.assertGreater(metrics["illegal_one_hot_mass"], 0.0)
        self.assertLessEqual(metrics["p_imp"], hr.choice_metrics(
            ideal, pool, inst, u0, u0 - 1)["p_imp"] + 1e-9)
        self.assertGreater(hr.total_variation(ideal, noisy), 0.0)

    def test_noise_models_are_labelled_separately(self):
        _model, gate = hr.gate_depolarizing_model(1e-3)
        _model2, thermal = hr.thermal_damping_model(1e-3)
        self.assertEqual(gate["kind"], "gate_depolarizing")
        self.assertEqual(thermal["kind"], "thermal_damping")
        self.assertNotEqual(gate["note"], thermal["note"])


class HardwareProtocolTests(unittest.TestCase):
    def test_protocol_is_preregistered_and_marks_every_job_field_unexecuted(self):
        selected, _strata = hr.select_dq_proxy(range(3), pool_k=3, per_stratum=2)
        protocol = hr.hardware_protocol(selected)
        self.assertEqual(protocol["status"], "not_executed")
        self.assertEqual(protocol["hardware_jobs_submitted"], 0)
        self.assertTrue(protocol["gate_vs_thermal_separated"])
        for slot in protocol["slots"]:
            self.assertEqual(slot["status"], "not_executed")
            self.assertEqual(slot["preset"]["batches"], 5)
            self.assertEqual(slot["preset"]["shots_per_batch"], 1024)
            self.assertAlmostEqual(slot["preset"]["mechanism_threshold"], 0.80)
            self.assertIsNone(slot["job"]["job_id"])
            self.assertIsNone(slot["job"]["counts_path"])


class PhysicalMappingTests(unittest.TestCase):
    def test_measured_mapping_reports_data_measurements_only(self):
        inst, pool, u0, witnesses = window()
        circuit, pool_obj = hr.build_sampling_circuit(pool, witnesses, u0 - 1)
        mapping = hr.physical_mapping(circuit, pool_obj.data_qubits)
        self.assertEqual(mapping["measurements"], pool_obj.data_qubits)
        self.assertGreaterEqual(mapping["physical_qubits"], circuit.num_qubits)
        self.assertGreaterEqual(mapping["cx"], 0)
        # the uniform legal preparation expands into reset-based state prep
        self.assertGreaterEqual(mapping["resets"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
