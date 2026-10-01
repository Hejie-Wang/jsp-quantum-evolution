#!/usr/bin/env python3
"""2x2 JSP permutation-space evolution as a Qiskit circuit, per docs/算法性能分析.md.

Encoding (compact per-machine rank encoding, 2 qubits for 2x2):
  qubit m = permutation rank of machine m (2! = 2 permutations -> 1 qubit each).
  qjsp numpy basis index n = rank0*2 + rank1 (machine 0 is the high digit);
  Qiskit statevector index     = rank0*1 + rank1*2 (qubit 0 is the LSB).

One evolution layer of H(s) = -(1-s)/d * sum_m S_m + s * diag(F - LB), d = 2:
  cost phase  exp(-i*s*h*diag(potential))  -> Rz(q0), Rz(q1), Rzz(q0,q1)  [exact]
  swap mixer  exp(i*beta*X_m), beta=(1-s)*h/d -> RX(-2*beta) on qubit m   [exact]

Usage:
  python qjsp_hardware.py --layers 1                  # ideal statevector check
  python qjsp_hardware.py --layers 1 --submit         # real QPU, 1024 shots
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

QJSP_DIR = Path(__file__).resolve().parent.parent / "JSP_量子演化可运行代码" / "jsp_quantum_evolution"
sys.path.insert(0, str(QJSP_DIR))

import qjsp  # noqa: E402

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402


def build_circuit(model: qjsp.PermutationModel, tau: float, layers: int, s_final: float) -> QuantumCircuit:
    """Qiskit circuit reproducing qjsp.evolve for a 2-machine x 2-job instance."""
    if model.shape != (2, 2) or model.d != 2:
        raise ValueError("this circuit builder supports exactly 2 machines x 2 jobs")
    potential = model.potential  # numpy index n = rank0*2 + rank1
    h = tau / layers
    qc = QuantumCircuit(2, name="qjsp_2x2")
    qc.h(0)
    qc.h(1)
    qc.barrier()
    for ell in range(layers):
        s = s_final * (ell + 0.5) / layers  # midpoint schedule, same as qjsp.evolve
        # Phases per basis state; numpy index n = 2*rank0 + rank1.
        def phase(r0, r1):
            return -s * h * float(potential[2 * r0 + r1])
        p00, p10, p01, p11 = phase(0, 0), phase(1, 0), phase(0, 1), phase(1, 1)
        b = (p00 - p10 + p01 - p11) / 4  # Z on qubit 0
        c = (p00 + p10 - p01 - p11) / 4  # Z on qubit 1
        e = (p00 - p10 - p01 + p11) / 4  # Z0 Z1
        qc.rz(-2 * b, 0)
        qc.rz(-2 * c, 1)
        qc.rzz(-2 * e, 0, 1)
        beta = (1 - s) * h / model.d
        qc.rx(-2 * beta, 0)  # exp(i*beta*S_m), S_m = X on a 2-permutation qubit
        qc.rx(-2 * beta, 1)
        qc.barrier()
    return qc


def decode_counts(model: qjsp.PermutationModel, counts: dict[str, int]) -> dict:
    """Map measured bitstrings (key = 'q1 q0') to schedules and costs."""
    shots = sum(counts.values())
    inst = model.instance
    rows = []
    for key, n_shots in sorted(counts.items()):
        key = key.replace(" ", "")
        rank0, rank1 = int(key[-1]), int(key[-2])
        n = 2 * rank0 + rank1
        cost = int(model.costs[n])
        rows.append({
            "bitstring": key, "shots": n_shots, "frequency": n_shots / shots,
            "numpy_basis_index": n, "orders_zero_based": model.orders_at(n),
            "cost": cost, "feasible": bool(cost < inst.penalty),
        })
    best = min((r for r in rows if r["feasible"]), key=lambda r: r["cost"], default=None)
    return {"shots": shots, "outcomes": rows,
            "p_feasible": sum(r["frequency"] for r in rows if r["feasible"]),
            "p_optimal": sum(r["frequency"] for r in rows if r["cost"] == int(model.costs.min())),
            "best_measured_cost": None if best is None else best["cost"]}


def ideal_probabilities(model, tau, layers, s_final):
    """Exact probabilities from the qjsp numpy evolution (classical reference)."""
    psi, _ = qjsp.evolve(model, tau=tau, layers=layers, s_final=s_final)
    probs = np.abs(psi) ** 2
    probs /= probs.sum()
    return probs  # numpy index order


def tv_distance_from_ideal(model, counts, ideal_probs):
    freq = np.zeros(model.dimension)
    total = sum(counts.values())
    for key, n in counts.items():
        key = key.replace(" ", "")
        freq[2 * int(key[-1]) + int(key[-2])] = n / total
    return 0.5 * float(np.abs(freq - ideal_probs).sum()), freq


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", default=str(QJSP_DIR / "data" / "demo_2x2.json"))
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--s-final", type=float, default=0.95)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--submit", action="store_true", help="run on a real IBM QPU")
    ap.add_argument("--backend", default=None, help="backend name; default = least busy")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    inst = qjsp.Instance.read(args.instance)
    model = qjsp.PermutationModel(inst)
    qc = build_circuit(model, args.tau, args.layers, args.s_final)

    ideal = ideal_probabilities(model, args.tau, args.layers, args.s_final)
    report = {
        "instance": inst.as_dict(), "tau": args.tau, "layers": args.layers,
        "s_final": args.s_final, "shots": args.shots,
        "cost_offset_LB": inst.lower_bound,
        "enumerated_optimum_diagnostic": int(model.costs.min()),
        "cost_table_numpy_index_order": [int(c) for c in model.costs],
        "ideal_probabilities_numpy_index_order": [float(p) for p in ideal],
        "ideal_p_optimal": float(ideal[model.costs == model.costs.min()].sum()),
        "ideal_p_feasible": float(ideal[model.costs < inst.penalty].sum()),
        "circuit": {"qubits": qc.num_qubits, "depth": qc.depth(),
                    "gate_counts": {k: int(v) for k, v in qc.count_ops().items()}},
    }

    # Ideal statevector sanity check: circuit must reproduce the numpy evolution.
    from qiskit.quantum_info import Statevector
    sv = np.asarray(Statevector.from_instruction(qc).data)  # qiskit index order
    sv_numpy_order = np.empty(4, dtype=complex)
    for r0 in (0, 1):
        for r1 in (0, 1):
            sv_numpy_order[2 * r0 + r1] = sv[r0 + 2 * r1]
    circ_probs = np.abs(sv_numpy_order) ** 2
    report["statevector_vs_numpy_max_prob_diff"] = float(np.abs(circ_probs - ideal).max())

    if not args.submit:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
        return 0

    # --- Real hardware path ---
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

    service = QiskitRuntimeService()
    backend = (service.least_busy(simulator=False, operational=True)
               if args.backend is None else service.backend(args.backend))
    report["backend"] = backend.name

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1,
                                      scheduling_method="alap")
    meas = qc.copy()
    meas.measure_all()
    isa = pm.run(meas)
    ops = isa.count_ops()
    two_q = sum(int(v) for k, v in ops.items() if k in ("cx", "ecr", "cz"))
    duration_s = None
    if isa.duration is not None and backend.dt:
        duration_s = isa.duration * backend.dt
    report["transpiled"] = {
        "depth": isa.depth(), "gate_counts": {k: int(v) for k, v in ops.items()},
        "two_qubit_gates": two_q, "scheduled_duration_seconds": duration_s,
    }
    print(json.dumps({"backend": backend.name, "transpiled": report["transpiled"]},
                     ensure_ascii=False, indent=2), flush=True)

    sampler = Sampler(mode=backend)
    job = sampler.run([isa], shots=args.shots)
    report["job_id"] = job.job_id()
    print(f"Job ID: {job.job_id()}", flush=True)
    counts = job.result()[0].data.meas.get_counts()

    tv, _freq = tv_distance_from_ideal(model, counts, ideal)
    decoded = decode_counts(model, counts)
    report["hardware"] = {"counts": counts, "tv_distance_vs_ideal": tv, **decoded}
    report["limitations"] = [
        "Few-layer circuit: validates the hardware implementation, does not reproduce "
        "the 2000-layer classical solution quality.",
        "Every hardware shot reprepares and re-evolves the state; unlike the CPU sampler.",
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
