"""Local-only gate builders for the machine-candidate JSP formulation.

The module deliberately has no provider, account, or backend imports.  It uses
Qiskit circuit objects only; callers choose a local simulator or offline
transpilation target.  Membership flags are exact on the legal one-hot
subspace, which is the encoded search space used by the candidate model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from math import ceil, log2
from typing import Iterable, Mapping, Sequence


def _hash_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidatePool:
    """Ordered machine candidates with an immutable content identity."""

    candidates: tuple[tuple[tuple[int, ...], ...], ...]
    version: str = "pool-v1"
    content_hash: str = field(init=False)

    def __post_init__(self):
        normalized = tuple(tuple(tuple(int(v) for v in order) for order in machine)
                           for machine in self.candidates)
        if any(not machine for machine in normalized):
            raise ValueError("every machine must have at least one candidate")
        if any(not order for machine in normalized for order in machine):
            raise ValueError("candidate orders must be nonempty")
        object.__setattr__(self, "candidates", normalized)
        object.__setattr__(self, "content_hash", _hash_json({
            "version": self.version, "candidates": normalized,
        }))

    @property
    def machines(self) -> int:
        return len(self.candidates)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(len(machine) for machine in self.candidates)

    @property
    def data_qubits(self) -> int:
        return sum(self.sizes)

    def offset(self, machine: int, candidate: int) -> int:
        if not 0 <= machine < self.machines:
            raise IndexError("machine index out of range")
        if not 0 <= candidate < self.sizes[machine]:
            raise IndexError("candidate index out of range")
        return sum(self.sizes[:machine]) + candidate


@dataclass(frozen=True)
class WitnessSpec:
    """Raw witness evidence plus derived per-machine allowed labels."""

    kind: str
    allowed: tuple[tuple[int, ...], ...]
    length: int = 0
    pool_hash: str | None = None
    nodes: tuple[int, ...] = ()

    def validate(self, pool: CandidatePool):
        if self.kind not in {"cycle", "path"}:
            raise ValueError("witness kind must be cycle or path")
        if self.pool_hash not in {None, pool.content_hash}:
            raise ValueError("witness belongs to a different candidate pool")
        if len(self.allowed) != pool.machines:
            raise ValueError("one allowed set is required per machine")
        for machine, labels in enumerate(self.allowed):
            if len(set(labels)) != len(labels):
                raise ValueError("allowed labels must be unique")
            if any(not 0 <= label < pool.sizes[machine] for label in labels):
                raise ValueError("allowed label out of range")
        if self.kind == "path" and self.length < 0:
            raise ValueError("path length must be nonnegative")


@dataclass(frozen=True)
class JointTransition:
    """A witness-derived transition between two legal candidate endpoints."""

    support: tuple[int, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    witness_id: str = ""
    weight: float = 1.0

    def validate(self, pool: CandidatePool):
        if not 2 <= len(self.support) <= 4:
            raise ValueError("joint support must contain 2 to 4 machines")
        if len(self.left) != len(self.support) or len(self.right) != len(self.support):
            raise ValueError("joint endpoint dimensions do not match support")
        if len(set(self.support)) != len(self.support):
            raise ValueError("joint support must not repeat machines")
        for machine, left, right in zip(self.support, self.left, self.right):
            if not 0 <= machine < pool.machines:
                raise ValueError("joint machine out of range")
            if not 0 <= left < pool.sizes[machine] or not 0 <= right < pool.sizes[machine]:
                raise ValueError("joint candidate out of range")
            if left == right:
                raise ValueError("joint endpoints must differ on every support machine")
        if self.weight <= 0:
            raise ValueError("joint weight must be positive")


def _require_qiskit():
    try:
        from qiskit import QuantumCircuit, QuantumRegister
    except ImportError as exc:  # pragma: no cover - exercised outside qskit env
        raise RuntimeError("run this module in the conda qskit environment") from exc
    return QuantumCircuit, QuantumRegister


def _apply_membership(circuit, data, pool: CandidatePool, machine: int,
                      allowed: Sequence[int], flag):
    """XOR allowed one-hot bits into a flag; exact for legal one-hot states."""
    if not allowed:
        return False
    if len(allowed) == pool.sizes[machine]:
        return True
    for label in allowed:
        circuit.cx(data[pool.offset(machine, label)], flag)
    return False


def _apply_predicate(circuit, controls, target):
    if not controls:
        circuit.x(target)
    elif len(controls) == 1:
        circuit.cx(controls[0], target)
    else:
        circuit.mcx(list(controls), target)


def _append_comparator(circuit, time_bits, compare, work, value):
    from qiskit.circuit.library import IntegerComparator
    if not time_bits:
        circuit.x(compare)
        return None
    gate = IntegerComparator(len(time_bits), int(value), geq=False)
    circuit.append(gate, list(time_bits) + [compare] + list(work))
    return gate


def build_phase_circuit(
    pool: CandidatePool,
    witnesses: Iterable[WitnessSpec],
    *,
    phase_angle: float = 0.25,
    fixed_t: int | None = None,
    time_bits: int | None = None,
    include_objective: bool = False,
    penalty: float | None = None,
    initialize_uniform: bool = False,
):
    """Build witness phase gates with reversible flag/comparator cleanup.

    `fixed_t` selects the fixed-T feasibility form.  Otherwise `time_bits`
    creates an explicit little-endian T register and path predicates use
    `T < witness.length`.  The returned circuit has named registers so tests
    and result writers can decode resource ownership without guessing.
    """
    QuantumCircuit, QuantumRegister = _require_qiskit()
    if fixed_t is not None and fixed_t < 0:
        raise ValueError("fixed_t must be nonnegative")
    if fixed_t is not None and time_bits is not None:
        raise ValueError("choose fixed_t or time_bits, not both")
    witnesses = tuple(witnesses)
    for witness in witnesses:
        witness.validate(pool)
    if penalty is None:
        penalty = 1.0
    data = QuantumRegister(pool.data_qubits, "candidate")
    t_count = 0 if fixed_t is not None else int(time_bits or 0)
    if t_count < 0:
        raise ValueError("time_bits must be nonnegative")
    t_reg = QuantumRegister(t_count, "T") if t_count else None
    # Predicate, comparator output, and their conjunction are separate.  The
    # comparator's internal work register is reused between witnesses.
    comparator_work = max(0, t_count - 1)
    witness_stride = pool.machines + 3
    ancilla_count = max(1, witness_stride * len(witnesses) + comparator_work)
    anc = QuantumRegister(ancilla_count, "work")
    circuit = QuantumCircuit(data, *(tuple([t_reg]) if t_reg else ()), anc)
    if initialize_uniform:
        for machine, size in enumerate(pool.sizes):
            # H on one-hot basis is not a legal uniform state for K>2.  Prepare
            # a deterministic legal basis and let XY mixers create support.
            circuit.x(data[pool.offset(machine, 0)])
    witness_flag = 0
    comparator_offset = witness_stride * len(witnesses)
    for witness in witnesses:
        witness_base = witness_flag
        supports = []
        constants_true = True
        for machine, allowed in enumerate(witness.allowed):
            if not allowed:
                constants_true = False
                break
            if len(allowed) < pool.sizes[machine]:
                supports.append((machine, allowed, anc[witness_base + machine]))
        if not constants_true:
            continue
        predicate = anc[witness_base + pool.machines]
        compare = anc[witness_base + pool.machines + 1]
        active = anc[witness_base + pool.machines + 2]
        witness_flag += witness_stride
        controls = []
        for machine, allowed, flag in supports:
            _apply_membership(circuit, data, pool, machine, allowed, flag)
            controls.append(flag)
        if supports:
            _apply_predicate(circuit, controls, predicate)
        else:
            # Every machine allows every label: this witness is a constant
            # predicate in the current pool, so materialize it briefly.
            circuit.x(predicate)
        comparator_used = False
        if witness.kind == "path" and fixed_t is None:
            if t_count == 0:
                raise ValueError("path witness needs fixed_t or time_bits")
            comparator_used = True
            _append_comparator(
                circuit, list(t_reg), compare,
                list(anc[comparator_offset:comparator_offset + comparator_work]),
                witness.length,
            )
            _apply_predicate(circuit, [predicate, compare], active)
        elif witness.kind == "path" and fixed_t is not None:
            if fixed_t >= witness.length:
                # The path predicate is false at this fixed threshold.
                _apply_predicate(circuit, controls, predicate)
                for _, allowed, flag in reversed(supports):
                    _apply_membership(circuit, data, pool, _, allowed, flag)
                continue
        phase_target = active if comparator_used else predicate
        circuit.p(-phase_angle * (penalty if witness.kind == "cycle" else 1.0), phase_target)
        if comparator_used:
            _apply_predicate(circuit, [predicate, compare], active)
        if comparator_used:
            _append_comparator(
                circuit, list(t_reg), compare,
                list(anc[comparator_offset:comparator_offset + comparator_work]),
                witness.length,
            )
        if supports:
            _apply_predicate(circuit, controls, predicate)
        else:
            circuit.x(predicate)
        for machine, allowed, flag in reversed(supports):
            _apply_membership(circuit, data, pool, machine, allowed, flag)
    if include_objective and t_reg is not None:
        for bit, qubit in enumerate(t_reg):
            circuit.p(-float(phase_angle) * (1 << bit), qubit)
    circuit.metadata = {
        "execution_mode": "local_simulator_or_offline_resource_compile",
        "hardware_jobs_submitted": 0,
        "pool_hash": pool.content_hash,
        "data_qubits": pool.data_qubits,
        "time_qubits": t_count,
        "work_qubits": ancilla_count,
        "witnesses": len(witnesses),
        "fixed_t": fixed_t,
    }
    return circuit


def build_xy_mixer(pool: CandidatePool, theta: float = 0.2, *, ring: bool = False):
    """One layer of legal one-hot-preserving XY exchanges."""
    QuantumCircuit, QuantumRegister = _require_qiskit()
    data = QuantumRegister(pool.data_qubits, "candidate")
    circuit = QuantumCircuit(data)
    from qiskit.circuit.library import XXPlusYYGate
    for machine, size in enumerate(pool.sizes):
        pairs = [(a, (a + 1) % size) for a in range(size)] if ring else [
            (a, b) for a in range(size) for b in range(a + 1, size)]
        if size < 2:
            continue
        for left, right in pairs:
            circuit.append(XXPlusYYGate(2 * theta), [
                data[pool.offset(machine, left)], data[pool.offset(machine, right)]
            ])
    circuit.metadata = {"mixer": "xy", "ring": ring, "hardware_jobs_submitted": 0}
    return circuit


def _local_transition_matrix(pool: CandidatePool, transition: JointTransition, theta: float):
    import numpy as np
    support = transition.support
    local_qubits = 2 * len(support)
    dimension = 1 << local_qubits
    matrix = np.eye(dimension, dtype=complex)
    left_index = 0
    right_index = 0
    for i, (left, right) in enumerate(zip(transition.left, transition.right)):
        left_index |= 1 << (2 * i)
        right_index |= 1 << (2 * i + 1)
    c, s = np.cos(theta), -1j * np.sin(theta)
    matrix[left_index, left_index] = c
    matrix[right_index, right_index] = c
    matrix[left_index, right_index] = s
    matrix[right_index, left_index] = s
    return matrix


def build_joint_mixer(pool: CandidatePool, transitions: Iterable[JointTransition],
                      theta: float = 0.2):
    """Compile support-local two-level rotations, never a full-space matrix."""
    QuantumCircuit, QuantumRegister = _require_qiskit()
    data = QuantumRegister(pool.data_qubits, "candidate")
    circuit = QuantumCircuit(data)
    from qiskit.circuit.library import UnitaryGate
    transitions = tuple(transitions)
    for transition in transitions:
        transition.validate(pool)
        matrix = _local_transition_matrix(pool, transition, theta * transition.weight)
        qargs = []
        for machine, left, right in zip(transition.support,
                                        transition.left, transition.right):
            qargs.extend((data[pool.offset(machine, left)],
                          data[pool.offset(machine, right)]))
        circuit.append(UnitaryGate(matrix, label="joint-local"), qargs)
    circuit.metadata = {
        "mixer": "witness_joint_local_support",
        "transitions": len(transitions),
        "hardware_jobs_submitted": 0,
        "support_matrix_qubits": [2 * len(t.support) for t in transitions],
    }
    return circuit


def circuit_resources(circuit) -> dict:
    """Return resource metadata without selecting a backend or simulating."""
    counts = circuit.count_ops()
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "count_ops": {str(k): int(v) for k, v in counts.items()},
        "metadata": dict(circuit.metadata or {}),
        "hardware_jobs_submitted": 0,
    }


def ensure_simulation_budget(circuit, *, max_qubits: int = 20,
                             memory_bytes: int = 512 * 1024 * 1024):
    """Reject statevector requests that exceed the local validation budget."""
    if circuit.num_qubits > max_qubits:
        raise ValueError(
            f"statevector simulation disabled above {max_qubits} qubits "
            f"(requested {circuit.num_qubits})"
        )
    required = (1 << circuit.num_qubits) * 16
    if required > memory_bytes:
        raise ValueError(
            f"statevector estimate {required} bytes exceeds {memory_bytes} byte budget"
        )


def decode_one_hot_counts(counts: Mapping[str, int], pool: CandidatePool,
                          *, t_bits: int = 0) -> dict:
    """Decode counts and explicitly retain illegal one-hot samples."""
    legal, illegal = {}, 0
    for bitstring, frequency in counts.items():
        bits = [int(ch) for ch in reversed(bitstring.replace(" ", ""))]
        if len(bits) < pool.data_qubits + t_bits:
            raise ValueError("count bitstring is shorter than requested registers")
        choices = []
        valid = True
        for machine, size in enumerate(pool.sizes):
            segment = bits[sum(pool.sizes[:machine]):sum(pool.sizes[:machine + 1])]
            active = [i for i, value in enumerate(segment) if value]
            if len(active) != 1:
                valid = False
                break
            choices.append(active[0])
        if not valid:
            illegal += int(frequency)
            continue
        t_value = sum(bits[pool.data_qubits + bit] << bit for bit in range(t_bits))
        legal[(tuple(choices), t_value)] = legal.get((tuple(choices), t_value), 0) + int(frequency)
    total = sum(int(v) for v in counts.values())
    return {
        "legal": legal,
        "illegal_count": illegal,
        "total_count": total,
        "illegal_fraction": illegal / total if total else 0.0,
    }
