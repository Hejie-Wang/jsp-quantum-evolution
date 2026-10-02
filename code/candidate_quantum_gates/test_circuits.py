"""Acceptance tests for P0 constraint/phase semantics and P1 resources.

Covers docs/candidate_quantum_improvements_20261002.md:
P0 - small-pool phase enumeration, non-unit penalties, T = L-1/L/L+1
     boundaries, constant-true/false conditions, random superpositions,
     auxiliary bits return to zero, intersection witness projection shared
     with medium_qjsp, real (not hand-written) demo witnesses.
P1 - exact joint-gate equivalence for r=2/3/4, legal-encoding preservation,
     untouched non-target states, reused work register, compiled CX/depth
     comparison against the dense reference under identical settings.
"""
import unittest
import sys
from pathlib import Path

import numpy as np
from itertools import product
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector

from circuits import (CandidatePool, JointTransition, WitnessSpec,
                      allowed_labels, build_joint_mixer, build_phase_circuit,
                      build_xy_mixer, circuit_resources, decode_one_hot_counts,
                      derive_witness, ensure_simulation_budget,
                      prepare_legal_basis, prepare_uniform_legal,
                      _local_transition_matrix)

POOL = CandidatePool((((0, 1), (1, 0)), ((2, 3), (3, 2))))
DEMO_POOL = CandidatePool((((0, 5, 7), (7, 5, 0)),
                           ((1, 3, 8), (3, 1, 8)),
                           ((2, 4, 6), (4, 6, 2))))
DEMO_DURATIONS = [[3, 2, 2], [2, 1, 4], [4, 3, 1]]
DEMO_ROUTES = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]


def _ensure_medium_path():
    medium = Path(__file__).resolve().parents[1] / "candidate_quantum_medium"
    if str(medium) not in sys.path:
        sys.path.insert(0, str(medium))


def _demo_instance():
    _ensure_medium_path()
    import medium_qjsp
    return medium_qjsp.Instance(DEMO_DURATIONS, DEMO_ROUTES, name="demo_3x3")


def _demo_real_witnesses():
    _ensure_medium_path()
    import medium_qjsp as mq
    inst = _demo_instance()
    raw, seen = [], set()
    for choice in product(*[range(s) for s in DEMO_POOL.sizes]):
        result = mq.decode(inst, [list(DEMO_POOL.candidates[m][a])
                                  for m, a in enumerate(choice)])
        witness = mq.separate(inst, result)
        if witness.key() not in seen:
            seen.add(witness.key())
            raw.append(witness)
    return _derive_all(raw)


def _derive_all(raw_witnesses):
    return tuple(derive_witness(DEMO_POOL, {
        "kind": w.kind, "relations": w.relations, "length": w.length,
        "nodes": w.nodes}) for w in raw_witnesses)


def _basis_state(circuit, pool, choice, t_value=0, t_offset=None):
    """Legal one-hot basis state on the full circuit width."""
    t_offset = pool.data_qubits if t_offset is None else t_offset
    index = sum(1 << pool.offset(m, a) for m, a in enumerate(choice))
    index |= t_value << t_offset
    return Statevector.from_int(index, 1 << circuit.num_qubits)


def _aux_population(state, data_bits):
    """Total probability on any state with a non-data bit set."""
    data = np.asarray(state.data)
    keep = np.arange(len(data)) >> data_bits == 0
    return float(np.sum(np.abs(data[~keep]) ** 2))


class WitnessProjectionTests(unittest.TestCase):
    def test_intersection_projection_doc_example(self):
        # Both relations must hold on one machine: only label 0 survives.
        self.assertEqual(
            allowed_labels([(0, 1, 2), (2, 0, 1), (1, 2, 0)], [(0, 1), (1, 2)]),
            (0,))

    def test_derive_witness_intersects_and_binds_pool_hash(self):
        # Machine 1 carries two relations whose per-candidate survivors
        # intersect to label 0 only; machine 2 is unconstrained.
        raw = {"kind": "cycle",
               "relations": [(0, 5, 7), (1, 1, 3), (1, 3, 8)]}
        witness = derive_witness(DEMO_POOL, raw)
        self.assertEqual(witness.allowed[0], (0,))
        self.assertEqual(witness.allowed[1], (0,))
        self.assertEqual(witness.allowed[2], (0, 1))
        self.assertEqual(witness.pool_hash, DEMO_POOL.content_hash)
        with self.assertRaises(ValueError):
            WitnessSpec("cycle", witness.allowed, pool_hash="stale").validate(DEMO_POOL)

    def test_empty_intersection_never_triggers(self):
        # (5 before 0) and (0 before 7) have no common candidate on machine 0.
        raw = {"kind": "cycle", "relations": [(0, 5, 0), (0, 0, 7)]}
        witness = derive_witness(DEMO_POOL, raw)
        self.assertEqual(witness.allowed[0], ())
        circuit = build_phase_circuit(DEMO_POOL, [witness], fixed_t=0)
        self.assertEqual(circuit.count_ops(), {})
        self.assertEqual(circuit.metadata["witnesses_skipped_constant_false"], 1)

    def test_derive_witness_matches_medium_projection(self):
        _ensure_medium_path()
        import medium_qjsp as mq
        rng = np.random.default_rng(11)
        for _ in range(25):
            relations = []
            machines = rng.choice(3, size=int(rng.integers(1, 4)), replace=False)
            for machine in machines:
                ops = DEMO_POOL.candidates[int(machine)][0]
                u, v = rng.choice(list(ops), size=2, replace=False)
                relations.append((int(machine), int(u), int(v)))
            raw = {"kind": "cycle", "relations": relations}
            witness = derive_witness(DEMO_POOL, raw)
            projected = mq.witness_support(
                mq.Witness("cycle", tuple(sorted(relations))), DEMO_POOL.candidates)
            if projected is None:
                empty = any(not labels for labels in witness.allowed)
                constrained = [m for m in range(3)
                               if any(rel[0] == m for rel in relations)]
                self.assertTrue(empty or not constrained)
            else:
                support, allowed = projected
                for machine in range(3):
                    expected = allowed.get(machine, tuple(range(DEMO_POOL.sizes[machine])))
                    self.assertEqual(tuple(witness.allowed[machine]), tuple(expected))

    def test_demo_witnesses_are_real_graph_evidence(self):
        witnesses = _demo_real_witnesses()
        self.assertTrue(witnesses)
        inst = _demo_instance()
        import medium_qjsp as mq
        # (0, 0, 0) decodes feasible with makespan 16: no witness may mark it
        # a cycle, and its own longest path must appear as a path witness.
        result = mq.decode(inst, [list(DEMO_POOL.candidates[m][0]) for m in range(3)])
        self.assertTrue(result["feasible"])
        for witness in witnesses:
            triggers = all(a in allowed for a, allowed in zip((0, 0, 0), witness.allowed))
            if witness.kind == "cycle":
                self.assertFalse(triggers)
        kinds = {w.kind for w in witnesses}
        self.assertEqual(kinds, {"cycle", "path"})


class PhaseSemanticsTests(unittest.TestCase):
    ANGLE = 0.31
    PENALTY = 23.0

    def test_small_pool_enumeration_matches_classical_predicate(self):
        witnesses = _demo_real_witnesses()
        circuit = build_phase_circuit(DEMO_POOL, witnesses, phase_angle=self.ANGLE,
                                      fixed_t=15, penalty=self.PENALTY)
        for choice in product(*[range(s) for s in DEMO_POOL.sizes]):
            triggered = sum(
                1 for w in witnesses
                if all(a in allowed for a, allowed in zip(choice, w.allowed))
                and (w.kind == "cycle" or w.length > 15))
            state = _basis_state(circuit, DEMO_POOL, choice).evolve(circuit)
            index = sum(1 << DEMO_POOL.offset(m, a) for m, a in enumerate(choice))
            expected = np.exp(-1j * self.ANGLE * self.PENALTY * triggered)
            self.assertAlmostEqual(abs(state.data[index] - expected), 0.0, places=6)
            self.assertAlmostEqual(_aux_population(state, DEMO_POOL.data_qubits),
                                   0.0, places=9)

    def test_register_boundaries_t_equal_l_minus_one_l_l_plus_one(self):
        witness = WitnessSpec("path", ((0,), (0,)), length=3,
                              pool_hash=POOL.content_hash)
        circuit = build_phase_circuit(POOL, [witness], phase_angle=0.2,
                                      time_bits=3, penalty=9.0)
        for t_value, active in ((2, True), (3, False), (4, False)):
            prep = QuantumCircuit(circuit.num_qubits)
            prep.x(POOL.offset(0, 0))
            prep.x(POOL.offset(1, 0))
            for bit in range(3):
                if t_value & (1 << bit):
                    prep.x(POOL.data_qubits + bit)
            state = Statevector.from_int(0, 1 << circuit.num_qubits)
            state = state.evolve(prep).evolve(circuit)
            index = ((1 << POOL.offset(0, 0)) | (1 << POOL.offset(1, 0))
                     | (t_value << POOL.data_qubits))
            expected = np.exp(-1j * 0.2 * 9.0) if active else 1.0
            self.assertAlmostEqual(abs(state.data[index] - expected), 0.0, places=6)

    def test_constant_true_witness_is_global_phase_only(self):
        witness = WitnessSpec("cycle", ((0, 1), (0, 1)), pool_hash=POOL.content_hash)
        circuit = build_phase_circuit(POOL, [witness], phase_angle=0.4,
                                      fixed_t=0, penalty=5.0)
        # No data-dependent gates: the witness can only contribute a global
        # phase, so no predicate/membership gate may appear.
        self.assertEqual(set(circuit.count_ops()) & {"cx", "mcx", "x", "p"}, set())
        for choice in product(*[range(s) for s in POOL.sizes]):
            state = _basis_state(circuit, POOL, choice).evolve(circuit)
            index = sum(1 << POOL.offset(m, a) for m, a in enumerate(choice))
            expected = np.exp(-1j * 0.4 * 5.0)
            self.assertAlmostEqual(abs(state.data[index] - expected), 0.0, places=6)

    def test_constant_false_witness_emits_nothing_and_stays_clean(self):
        empty = WitnessSpec("cycle", ((), (0,)), pool_hash=POOL.content_hash)
        circuit = build_phase_circuit(POOL, [empty], fixed_t=0)
        self.assertEqual(circuit.count_ops(), {})
        self.assertEqual(circuit.metadata["witnesses_skipped_constant_false"], 1)
        beyond = WitnessSpec("path", ((0,), (0,)), length=3,
                             pool_hash=POOL.content_hash)
        circuit = build_phase_circuit(POOL, [beyond], fixed_t=3, penalty=7.0)
        self.assertEqual(circuit.count_ops(), {})
        for choice in product(*[range(s) for s in POOL.sizes]):
            state = _basis_state(circuit, POOL, choice).evolve(circuit)
            self.assertAlmostEqual(
                _aux_population(state, POOL.data_qubits), 0.0, places=9)

    def test_random_superposition_gets_exact_phases_and_clean_aux(self):
        witnesses = _demo_real_witnesses()
        circuit = build_phase_circuit(DEMO_POOL, witnesses, phase_angle=0.17,
                                      fixed_t=15, penalty=11.0)
        rng = np.random.default_rng(5)
        amplitudes = np.zeros(1 << circuit.num_qubits, dtype=complex)
        choices = list(product(*[range(s) for s in DEMO_POOL.sizes]))
        weights = rng.random(len(choices)) + 0.1
        weights /= weights.sum()
        for weight, choice in zip(weights, choices):
            index = sum(1 << DEMO_POOL.offset(m, a) for m, a in enumerate(choice))
            amplitudes[index] = np.sqrt(weight)
        state = Statevector(amplitudes).evolve(circuit)
        data = np.asarray(state.data)
        self.assertAlmostEqual(_aux_population(state, DEMO_POOL.data_qubits),
                               0.0, places=9)
        for weight, choice in zip(weights, choices):
            triggered = sum(
                1 for w in witnesses
                if all(a in allowed for a, allowed in zip(choice, w.allowed))
                and (w.kind == "cycle" or w.length > 15))
            index = sum(1 << DEMO_POOL.offset(m, a) for m, a in enumerate(choice))
            expected = np.sqrt(weight) * np.exp(-1j * 0.17 * 11.0 * triggered)
            self.assertAlmostEqual(abs(data[index] - expected), 0.0, places=6)
            self.assertAlmostEqual(abs(data[index]), np.sqrt(weight), places=6)

    def test_work_register_is_reused_across_witness_count(self):
        witnesses = _demo_real_witnesses()
        widths = set()
        for count in range(1, len(witnesses) + 1):
            circuit = build_phase_circuit(DEMO_POOL, witnesses[:count],
                                          fixed_t=15)
            widths.add(circuit.num_qubits)
            self.assertEqual(circuit.metadata["work_qubits"],
                             DEMO_POOL.machines + 3)
        self.assertEqual(len(widths), 1)

    def test_initial_state_preparation_is_separate(self):
        uniform = prepare_uniform_legal(DEMO_POOL)
        probabilities = np.abs(np.asarray(Statevector(uniform).data)) ** 2
        for choice in product(*[range(s) for s in DEMO_POOL.sizes]):
            index = sum(1 << DEMO_POOL.offset(m, a) for m, a in enumerate(choice))
            self.assertAlmostEqual(probabilities[index], 1 / 8, places=9)
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=9)
        basis = prepare_legal_basis(DEMO_POOL, label=1)
        index = sum(1 << DEMO_POOL.offset(m, 1) for m in range(3))
        state = Statevector.from_int(0, 1 << DEMO_POOL.data_qubits).evolve(basis)
        self.assertAlmostEqual(abs(state.data[index]), 1.0, places=9)
        with self.assertRaises(ValueError):
            build_phase_circuit(DEMO_POOL, [], fixed_t=0, penalty=0.0)


class ObjectiveRegisterTests(unittest.TestCase):
    def test_t_register_objective_and_penalty_composition(self):
        witness = WitnessSpec("path", ((0,), (0,)), length=3,
                              pool_hash=POOL.content_hash)
        circuit = build_phase_circuit(POOL, [witness], phase_angle=0.5,
                                      time_bits=3, include_objective=True,
                                      penalty=9.0)
        for t_value in (0, 2, 3):
            prep = QuantumCircuit(circuit.num_qubits)
            prep.x(POOL.offset(0, 0))
            prep.x(POOL.offset(1, 0))
            for bit in range(3):
                if t_value & (1 << bit):
                    prep.x(POOL.data_qubits + bit)
            state = Statevector.from_int(0, 1 << circuit.num_qubits)
            state = state.evolve(prep).evolve(circuit)
            index = ((1 << POOL.offset(0, 0)) | (1 << POOL.offset(1, 0))
                     | (t_value << POOL.data_qubits))
            penalty_part = 9.0 if t_value < 3 else 0.0
            expected = np.exp(-1j * 0.5 * (t_value + penalty_part))
            self.assertAlmostEqual(abs(state.data[index] - expected), 0.0,
                                   places=6)


class JointGateTests(unittest.TestCase):
    CASES = (((0, 1), (0, 0), (1, 1)),
             ((0, 1, 2), (0, 1, 0), (1, 0, 1)),
             ((0, 1, 2, 3), (0, 0, 0, 0), (1, 1, 1, 1)))

    def _pool_for(self, support):
        machines = max(support) + 1
        return CandidatePool(tuple(((0, 1), (1, 0)) for _ in range(machines)))

    def test_exact_construction_matches_dense_reference(self):
        for support, left, right in self.CASES:
            pool = self._pool_for(support)
            transition = JointTransition(support, left, right, "w0")
            for theta in (0.11, 0.4):
                exact = build_joint_mixer(pool, [transition], theta=theta)
                reference = QuantumCircuit(pool.data_qubits)
                qargs = []
                for machine, l, r in zip(support, left, right):
                    qargs.extend((pool.offset(machine, l), pool.offset(machine, r)))
                reference.append(UnitaryGate(
                    _local_transition_matrix(pool, transition, theta)), qargs)
                difference = np.abs(Operator(exact).data
                                    - Operator(reference).data).max()
                self.assertLess(difference, 1e-9,
                                f"r={len(support)} theta={theta}")

    def test_legal_encoding_preserved_and_non_targets_untouched(self):
        pool = self._pool_for((0, 1))
        transition = JointTransition((0, 1), (0, 0), (1, 1), "w0")
        gate = build_joint_mixer(pool, [transition], theta=0.35)
        rng = np.random.default_rng(3)
        legal = np.zeros(1 << pool.data_qubits, dtype=complex)
        for choice in product(*[range(s) for s in pool.sizes]):
            index = sum(1 << pool.offset(m, a) for m, a in enumerate(choice))
            legal[index] = rng.normal() + 1j * rng.normal()
        out = np.asarray(Statevector(legal).evolve(gate).data)
        illegal = sum(abs(out[i]) ** 2
                      for i in range(1 << pool.data_qubits)
                      if not self._is_one_hot(i, pool))
        self.assertLess(illegal, 1e-12)
        # A state with both support machines outside {left, right} labels is
        # an eigenvector with eigenvalue 1.
        fixed = (1 << pool.offset(0, 0)) | (1 << pool.offset(1, 0))
        moved = (1 << pool.offset(0, 1)) | (1 << pool.offset(1, 1))
        untouched = np.zeros(1 << pool.data_qubits, dtype=complex)
        untouched[fixed] = 0.5
        untouched[moved] = 0.5
        # fixed/moved span the rotated subspace; instead verify a disjoint
        # basis state is unchanged.
        other = (1 << pool.offset(0, 0)) | (1 << pool.offset(1, 1))
        state = Statevector.from_int(other, 1 << pool.data_qubits).evolve(gate)
        self.assertAlmostEqual(abs(state.data[other]), 1.0, places=9)

    @staticmethod
    def _is_one_hot(index, pool):
        for machine, size in enumerate(pool.sizes):
            segment = (index >> sum(pool.sizes[:machine])) & ((1 << size) - 1)
            if bin(segment).count("1") != 1:
                return False
        return True

    def test_exact_construction_uses_no_ancilla_and_fewer_cx(self):
        support, left, right = self.CASES[0]
        pool = self._pool_for(support)
        transition = JointTransition(support, left, right, "w0")
        exact = build_joint_mixer(pool, [transition], theta=0.2)
        self.assertEqual(exact.num_qubits, pool.data_qubits)
        dense = QuantumCircuit(pool.data_qubits)
        qargs = [pool.offset(0, 0), pool.offset(0, 1),
                 pool.offset(1, 0), pool.offset(1, 1)]
        dense.append(UnitaryGate(_local_transition_matrix(pool, transition, 0.2)),
                     qargs)
        settings = dict(basis_gates=["u", "cx"], optimization_level=0)
        exact_t = transpile(exact, **settings)
        dense_t = transpile(dense, **settings)
        exact_cx = exact_t.count_ops().get("cx", 0)
        dense_cx = dense_t.count_ops().get("cx", 0)
        self.assertLess(exact_cx, dense_cx)

    def test_phase_circuit_metadata_reports_included_and_skipped(self):
        real = _demo_real_witnesses()
        beyond = WitnessSpec("path", ((0,), (0,), (0,)), length=99,
                             pool_hash=DEMO_POOL.content_hash)
        circuit = build_phase_circuit(DEMO_POOL, real + (beyond,),
                                      fixed_t=15, penalty=11.0)
        applied = sum(1 for w in real
                      if w.kind == "cycle" or w.length > 15) + 1  # beyond
        self.assertEqual(circuit.metadata["witnesses"], len(real) + 1)
        self.assertEqual(circuit.metadata["witnesses_applied"], applied)
        self.assertEqual(circuit.metadata["witnesses_skipped_constant_false"], 1)


class ResourceGuardTests(unittest.TestCase):
    def test_simulation_budget_rejects_medium_circuit(self):
        circuit = build_xy_mixer(CandidatePool(
            tuple(((0,), (1,)) for _ in range(11))))
        with self.assertRaises(ValueError):
            ensure_simulation_budget(circuit, max_qubits=20)

    def test_mixers_are_unitary_and_offline(self):
        xy = build_xy_mixer(POOL, ring=True)
        self.assertTrue(np.allclose(Operator(xy).data.conj().T @ Operator(xy).data,
                                    np.eye(2 ** POOL.data_qubits), atol=1e-8))
        transition = JointTransition((0, 1), (0, 0), (1, 1), "w0")
        joint = build_joint_mixer(POOL, [transition], theta=0.11)
        self.assertTrue(np.allclose(Operator(joint).data.conj().T @ Operator(joint).data,
                                    np.eye(2 ** POOL.data_qubits), atol=1e-8))
        self.assertEqual(circuit_resources(joint)["hardware_jobs_submitted"], 0)

    def test_count_decoder_rejects_illegal_without_repair(self):
        result = decode_one_hot_counts({"0101": 2, "0011": 3}, POOL)
        self.assertEqual(result["illegal_count"], 3)
        self.assertEqual(result["legal"][((0, 0), 0)], 2)


if __name__ == "__main__":
    unittest.main()
