#!/usr/bin/env python3
"""Run local gate checks and offline resource compilation.

This entry point intentionally has no hardware/provider option.  The medium
case is compiled only; no statevector or cloud job is created for it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter

from circuits import (CandidatePool, JointTransition, WitnessSpec,
                      build_joint_mixer, build_phase_circuit, build_xy_mixer,
                      circuit_resources, decode_one_hot_counts)


def package_versions():
    names = ("qiskit", "qiskit-aer", "numpy", "scipy")
    return {name: importlib.metadata.version(name)
            if _has_package(name) else None for name in names}


def _has_package(name):
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def demo_pool_and_witnesses():
    # Same fixed fixture as candidate_quantum_demo, expressed as immutable data.
    pool = CandidatePool((((0, 5, 7), (7, 5, 0)),
                          ((1, 3, 8), (3, 1, 8)),
                          ((2, 4, 6), (4, 6, 2))))
    witnesses = (
        WitnessSpec("cycle", ((0,), (0,), (0,)), pool_hash=pool.content_hash),
        WitnessSpec("path", ((0,), (0,), (0,)), length=11,
                     pool_hash=pool.content_hash),
    )
    return pool, witnesses


def run_demo_sampling(shots=256, seed=7):
    from qiskit import ClassicalRegister, QuantumCircuit
    from qiskit_aer import AerSimulator
    import numpy as np

    pool, witnesses = demo_pool_and_witnesses()
    phase = build_phase_circuit(pool, witnesses[:1], fixed_t=10,
                                phase_angle=0.31, penalty=23)
    circuit = QuantumCircuit(*phase.qregs, ClassicalRegister(pool.data_qubits, "readout"))
    for machine, size in enumerate(pool.sizes):
        if size != 2:
            raise ValueError("demo sampling fixture expects two candidates per machine")
        q0, q1 = pool.offset(machine, 0), pool.offset(machine, 1)
        circuit.initialize([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], [q0, q1])
    circuit.compose(phase, inplace=True)
    circuit.measure(list(circuit.qregs[0]), list(circuit.cregs[0]))
    simulator = AerSimulator()
    result = simulator.run(circuit, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()
    decoded = decode_one_hot_counts(counts, pool)
    decoded["legal"] = {f"{choice}|T={t}": count
                         for (choice, t), count in decoded["legal"].items()}
    return {
        "mode": "local_aer_simulator",
        "hardware_jobs_submitted": 0,
        "shots": shots,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "decoded": decoded,
        "resources": circuit_resources(phase),
    }


def derive_witness(pool, raw):
    allowed = [[] for _ in range(pool.machines)]
    for machine, u, v in raw.get("relations", []):
        for label, order in enumerate(pool.candidates[machine]):
            if order.index(u) < order.index(v) and label not in allowed[machine]:
                allowed[machine].append(label)
    # Machines absent from the raw witness are unconstrained.
    for machine in range(pool.machines):
        if not allowed[machine] and not any(machine == rel[0] for rel in raw.get("relations", [])):
            allowed[machine] = list(range(pool.sizes[machine]))
    return WitnessSpec(raw["kind"], tuple(tuple(labels) for labels in allowed),
                       length=int(raw.get("length", 0)), pool_hash=pool.content_hash,
                       nodes=tuple(raw.get("nodes", ())))


def compile_medium(path, witness_limit=2, joint_limit=2, timeout=60.0):
    from qiskit import transpile

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    pool = CandidatePool(tuple(tuple(tuple(item["value"] if isinstance(item, dict) else item)
                                      for item in machine)
                          for machine in report["pool_data"]))
    witnesses = tuple(derive_witness(pool, raw)
                      for raw in report.get("witness_data", [])[:witness_limit])
    start = perf_counter()
    total_duration = sum(sum(int(v) for v in row)
                         for row in report["instance_data"]["durations"])
    phase = build_phase_circuit(pool, witnesses, fixed_t=int(report["final_target_T"]),
                                phase_angle=0.01, penalty=total_duration + 1)
    phase_logical = circuit_resources(phase)
    phase_transpiled = transpile(phase, basis_gates=["u", "cx"], optimization_level=0)
    phase_elapsed = perf_counter() - start
    transitions = []
    for index, witness in enumerate(witnesses[:joint_limit]):
        support = tuple(machine for machine, labels in enumerate(witness.allowed)
                        if 0 < len(labels) < pool.sizes[machine])[:4]
        if len(support) < 2:
            continue
        left = tuple(witness.allowed[machine][0] for machine in support)
        right = tuple(next(label for label in range(pool.sizes[machine])
                           if label not in witness.allowed[machine]) for machine in support)
        transitions.append(JointTransition(support, left, right, f"w{index}"))
    joint = build_joint_mixer(pool, transitions, theta=0.05)
    joint_transpiled = transpile(joint, basis_gates=["u", "cx"], optimization_level=0)
    return {
        "mode": "offline_resource_compile",
        "hardware_jobs_submitted": 0,
        "source_result": str(Path(path).resolve()),
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "candidate_data_qubits": pool.data_qubits,
        "time_qubits": 0,
        "witnesses_included": len(witnesses),
        "witnesses_available": len(report.get("witness_data", [])),
        "witnesses_omitted": max(0, len(report.get("witness_data", [])) - len(witnesses)),
        "joint_terms_included": len(transitions),
        "phase_logical": phase_logical,
        "phase_transpiled": circuit_resources(phase_transpiled),
        "joint_logical": circuit_resources(joint),
        "joint_transpiled": circuit_resources(joint_transpiled),
        "compile_seconds": phase_elapsed,
        "statevector_simulation": False,
        "note": "Derived 15x20 pool; partial witness layer, no medium solution claim.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medium-result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shots", type=int, default=256)
    args = parser.parse_args(argv)
    result = {"environment": {"python": sys.executable,
                               "python_version": sys.version,
                               "packages": package_versions()},
              "small_demo": run_demo_sampling(args.shots)}
    if args.medium_result:
        result["medium_resources"] = compile_medium(args.medium_result)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
