#!/usr/bin/env python3
"""Run local gate checks and offline resource compilation.

This entry point intentionally has no hardware/provider option.  The medium
case is compiled only; no statevector or cloud job is created for it.

P0/P1 updates (docs/candidate_quantum_improvements_20261002.md): witnesses for
the demo fixture come from real graph evaluation instead of hand-written fake
evidence, ``derive_witness`` shares the intersection semantics of
``medium_qjsp.witness_support``, and the medium compile includes every
available witness on one reused work register.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter

from circuits import (CandidatePool, JointTransition,
                      build_joint_mixer, build_phase_circuit,
                      circuit_resources, decode_one_hot_counts, derive_witness,
                      prepare_uniform_legal)

_MEDIUM_DIR = Path(__file__).resolve().parents[1] / "candidate_quantum_medium"
if str(_MEDIUM_DIR) not in sys.path:
    sys.path.insert(0, str(_MEDIUM_DIR))


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
    """Same fixed pool fixture as candidate_quantum_demo with REAL witnesses.

    Every witness below is produced by full graph evaluation (medium_qjsp
    decode/separate) over all eight pool combinations, then projected onto
    candidate labels with the intersection rule.  Choice (0, 0, 0) decodes to
    a feasible schedule with makespan 16, so nothing may label it a cycle.
    """
    from search_loop import covering_demo_witnesses, demo_candidate_pool
    pool = demo_candidate_pool()
    _raw, witnesses = covering_demo_witnesses(pool)
    for witness in witnesses:
        witness.validate(pool)
    return pool, witnesses


def run_demo_sampling(shots=256, seed=7):
    """Small-circuit sampling demo: initial state -> [phase -> mixers] -> readout.

    A bare uniform superposition followed by the phase layer alone would leave
    measurement probabilities untouched; the mixers are what turn phases into
    a changed distribution.  This demo reports both distributions as evidence.
    """
    from qiskit import ClassicalRegister, QuantumCircuit
    from qiskit_aer import AerSimulator

    from search_loop import (build_fixed_t_ansatz, demo_candidate_pool,
                             demo_instance, graph_witness_specs)

    pool = demo_candidate_pool()
    inst = demo_instance()
    _raw, witnesses = demo_pool_and_witnesses()
    effective = [s for s in witnesses
                 if any(0 < len(a) < size for a, size in
                        zip(s.allowed, pool.sizes))]
    fixed_t = 15  # below the verified 16 initial makespan of this pool
    penalty = float(inst.total_duration + 1)
    phase_only = build_phase_circuit(pool, effective, phase_angle=0.31,
                                     fixed_t=fixed_t, penalty=penalty)
    ansatz = build_fixed_t_ansatz(pool, effective, (), [(0.31, 0.25, 0.25)],
                                  fixed_t, penalty)

    def sample(circuit):
        measured = QuantumCircuit(*circuit.qregs,
                                  ClassicalRegister(pool.data_qubits, "readout"))
        measured.compose(circuit, qubits=list(range(circuit.num_qubits)),
                         inplace=True)
        measured.measure(list(range(pool.data_qubits)),
                         list(range(pool.data_qubits)))
        from qiskit import transpile
        measured = transpile(measured, basis_gates=["u", "cx"],
                             optimization_level=0, seed_transpiler=seed)
        counts = AerSimulator().run(measured, shots=shots,
                                    seed_simulator=seed).result().get_counts()
        decoded = decode_one_hot_counts(counts, pool)
        return {f"{choice}|T={t}": count
                for (choice, t), count in decoded["legal"].items()}, decoded

    # uniform-only reference distribution (initial state, no phase, no mixer)
    bare = QuantumCircuit(*phase_only.qregs)
    bare.compose(prepare_uniform_legal(pool),
                 qubits=list(range(pool.data_qubits)), inplace=True)
    uniform_counts, uniform_decoded = sample(bare)
    mixed_counts, mixed_decoded = sample(ansatz)
    return {
        "mode": "local_aer_simulator",
        "hardware_jobs_submitted": 0,
        "shots": shots,
        "witnesses": len(effective),
        "fixed_t": fixed_t,
        "penalty": penalty,
        "uniform_legal_counts": uniform_counts,
        "phase_plus_mixer_counts": mixed_counts,
        "distributions_differ": uniform_counts != mixed_counts,
        "illegal_fraction": {"uniform": uniform_decoded["illegal_fraction"],
                             "phase_plus_mixer": mixed_decoded["illegal_fraction"]},
        "resources": {"phase_only": circuit_resources(phase_only),
                      "ansatz": circuit_resources(ansatz)},
    }


def compile_medium(path, witness_limit=None, joint_limit=2, timeout=600.0):
    """Offline resource compile of a saved medium report with ALL witnesses.

    The report must embed ``pool_data``/``witness_data`` (kaiwu-style).  The
    phase circuit reuses one work register across witnesses; the joint mixer
    uses the exact CX/X/MC-Rx construction with zero ancillas.  Dense joint
    matrices are kept as unit-test references only.
    """
    from qiskit import transpile

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    pool = CandidatePool(tuple(tuple(tuple(item["value"] if isinstance(item, dict) else item)
                                      for item in machine)
                              for machine in report["pool_data"]))
    raw_witnesses = tuple(report.get("witness_data", []))
    included = raw_witnesses if witness_limit is None else raw_witnesses[:witness_limit]
    witnesses = tuple(derive_witness(pool, raw) for raw in included)
    total_duration = sum(sum(int(v) for v in row)
                         for row in report["instance_data"]["durations"])
    start = perf_counter()
    phase = build_phase_circuit(pool, witnesses, fixed_t=int(report["final_target_T"]),
                                phase_angle=0.01, penalty=total_duration + 1)
    phase_logical = circuit_resources(phase)
    phase_transpiled_res = None
    if pool.data_qubits + phase.num_qubits - pool.data_qubits <= 200:
        phase_transpiled = transpile(phase, basis_gates=["u", "cx"],
                                     optimization_level=0)
        phase_transpiled_res = circuit_resources(phase_transpiled)
    phase_elapsed = perf_counter() - start
    transitions = []
    for index, witness in enumerate(witnesses):
        if len(transitions) >= joint_limit:
            break
        support = tuple(machine for machine, labels in enumerate(witness.allowed)
                        if 0 < len(labels) < pool.sizes[machine])[:4]
        if len(support) < 2:
            continue
        left = tuple(witness.allowed[machine][0] for machine in support)
        right = tuple(next(label for label in range(pool.sizes[machine])
                           if label not in witness.allowed[machine]) for machine in support)
        transitions.append(JointTransition(support, left, right, f"w{index}"))
    joint = build_joint_mixer(pool, transitions, theta=0.05)
    joint_transpiled = transpile(joint, basis_gates=["u", "cx"],
                                 optimization_level=0)
    return {
        "mode": "offline_resource_compile",
        "hardware_jobs_submitted": 0,
        "source_result": str(Path(path).resolve()),
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "candidate_data_qubits": pool.data_qubits,
        "time_qubits": 0,
        "witnesses_included": len(witnesses),
        "witnesses_available": len(raw_witnesses),
        "witnesses_omitted": len(raw_witnesses) - len(witnesses),
        "joint_terms_included": len(transitions),
        "phase_logical": phase_logical,
        "phase_transpiled": phase_transpiled_res,
        "joint_logical": circuit_resources(joint),
        "joint_transpiled": circuit_resources(joint_transpiled),
        "compile_seconds": phase_elapsed,
        "statevector_simulation": False,
        "note": ("One shared work register across all witnesses; joint gates "
                 "use the exact CX/X/MC-Rx construction with zero ancillas; "
                 "no medium solution claim."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medium-result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--witness-limit", type=int, default=None)
    args = parser.parse_args(argv)
    result = {"environment": {"python": sys.executable,
                               "python_version": sys.version,
                               "packages": package_versions()},
              "small_demo": run_demo_sampling(args.shots)}
    if args.medium_result:
        result["medium_resources"] = compile_medium(args.medium_result,
                                                    witness_limit=args.witness_limit)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
