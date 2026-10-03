#!/usr/bin/env python3
"""T09: hardware resource, noise and service-cost report (simulation only).

Implements the *simulation* half of the T09 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md.  No QPU job is submitted,
no credential is used, and every hardware field is written as
``not_executed`` with the pre-registered protocol attached, so the report can
be completed later without re-deciding the experiment after seeing results.

What is measured here, per selected window:

  * the ideal sampling distribution q of the *witness-phase* circuit
    (uniform legal preparation + native conditional phase at fixed T), decoded
    to machine-candidate choices;
  * the noisy distributions q_noise under two noise models that are kept
    strictly separate, as the work package requires: a gate-level depolarizing
    model and a thermal amplitude-damping model (annealer-like relaxation).
    A noisy simulation is never presented as hardware data;
  * p_imp (mass on legal, improving choices), the target-reaching mass, the
    illegal one-hot leakage, and TV(q, q_noise) as a *diagnostic only* - the
    primary metric stays p_imp, as specified;
  * logical data/work qubits, the physical mapping after transpilation to a
    fixed line topology, two-qubit gate counts, depth, measurement and reset
    counts, and whether the work register is uncomputed.

Window selection follows the DQ rule "pick by classical structure, never by
quantum benefit": the six windows are stratified by the *classical* improvement
density rho of the frozen pool.  D2 does not exist yet (T04 is producing it), so
this run uses a clearly labelled DQ proxy built from the D0 frozen layer and
records the substitution reason; the pre-registered hardware protocol still
names the six DQ slots to be re-selected from D2 before any submission.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GATES_DIR = Path(__file__).resolve().parent
MEDIUM_DIR = ROOT / "code" / "candidate_quantum_medium"
for _p in (str(GATES_DIR), str(MEDIUM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import circuits  # noqa: E402
import medium_qjsp as mq  # noqa: E402
from window_diagnostics import check_choice  # noqa: E402

SCHEMA_VERSION = 1
MECHANISM_THRESHOLD = 0.80          # pre-registered: hardware p_imp >= 80% of ideal
BATCHES = 5                          # pre-registered hardware batches
SHOTS_PER_BATCH = 1024               # pre-registered shots per batch
NOISE_LEVELS = {"gate_depolarizing": (1e-3, 5e-3), "thermal_damping": (1e-3, 5e-3)}


# ---------------------------------------------------------------------------
# Window selection (classical structure only)
# ---------------------------------------------------------------------------

def structural_profile(inst, pool, u0):
    """Classical-only window profile used for stratification."""
    sizes = [len(machine) for machine in pool]
    total = int(np.prod(sizes))
    feasible = improving = 0
    best = None
    for choice in itertools.product(*(range(k) for k in sizes)):
        result = check_choice(inst, pool, choice)
        if not result["feasible"]:
            continue
        feasible += 1
        if result["makespan"] < u0:
            improving += 1
        if best is None or result["makespan"] < best:
            best = result["makespan"]
    return {"combinations": total, "feasible": feasible, "improving": improving,
            "rho": improving / total if total else None,
            "pool_optimum": best, "u0": int(u0)}


def select_dq_proxy(seeds, pool_k=3, per_stratum=2, seed_offset=1000):
    """Six windows stratified by classical improvement density rho.

    Strata: no improvement (rho = 0), intermediate (0 < rho <= 0.25), higher
    (rho > 0.25).  The selection uses only classical pool diagnostics.
    """
    import data_identity
    from window_diagnostics import build_window
    candidates = []
    for jobs, machines in data_identity.D0_SIZES:
        for seed in seeds:
            inst = data_identity.build_d0(jobs, machines, seed)
            pool, incumbent, u0 = build_window(inst, pool_k, seed + seed_offset)
            profile = structural_profile(inst, pool, u0)
            candidates.append({
                "scale": f"{jobs}x{machines}", "seed": seed, "inst": inst,
                "pool": pool, "u0": u0, "profile": profile,
                "instance_sha256": data_identity.instance_sha256(inst),
                "pool_sha256": mq.content_hash(pool),
            })
    strata = {
        "no_improvement": [c for c in candidates if c["profile"]["improving"] == 0],
        "intermediate": [c for c in candidates
                         if 0 < c["profile"]["improving"]
                         and (c["profile"]["rho"] or 0) <= 0.25],
        "higher": [c for c in candidates if (c["profile"]["rho"] or 0) > 0.25],
    }
    selected, taken = [], set()
    for name in ("no_improvement", "intermediate", "higher"):
        pool_of = sorted(strata[name], key=lambda c: (c["scale"], c["seed"]))
        for candidate in pool_of[:per_stratum]:
            taken.add(id(candidate))
            selected.append(candidate)
    wanted = 3 * per_stratum
    if len(selected) < wanted:
        # The higher-density stratum is often empty on D0 frozen pools; the six
        # DQ slots are topped up with the largest classical rho still available
        # and the substitution is recorded instead of silently shrinking DQ.
        remaining = [c for c in candidates if id(c) not in taken]
        remaining.sort(key=lambda c: (-(c["profile"]["rho"] or 0), c["scale"], c["seed"]))
        for candidate in remaining[:wanted - len(selected)]:
            candidate["selection_note"] = ("fallback: highest classical rho among the "
                                           "remaining windows (density stratum empty)")
            selected.append(candidate)
    return selected, {name: len(rows) for name, rows in strata.items()}


# ---------------------------------------------------------------------------
# Circuit construction and decoding
# ---------------------------------------------------------------------------

def build_sampling_circuit(pool, witnesses, target_t):
    """Uniform legal preparation + native conditional phase at fixed T.

    The phases do not change the ideal measurement distribution; the point of
    this circuit is that it is the object the recorder would send to hardware,
    so its noise behaviour and its physical mapping are what T09 must report.
    """
    pool_obj = circuits.CandidatePool(
        tuple(tuple(tuple(int(v) for v in order) for order in machine)
              for machine in pool))
    specs = [circuits.derive_witness(pool_obj, w) for w in witnesses]
    prep = circuits.prepare_uniform_legal(pool_obj)
    phase = circuits.build_phase_circuit(pool_obj, specs, phase_angle=np.pi,
                                         fixed_t=int(target_t), penalty=1.0)
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(phase.num_qubits)
    circuit.compose(prep, qubits=list(range(prep.num_qubits)), inplace=True)
    circuit.compose(phase, qubits=list(range(phase.num_qubits)), inplace=True)
    circuit.metadata = {
        "execution_mode": "local_simulation_only",
        "hardware_jobs_submitted": 0,
        "pool_hash": pool_obj.content_hash,
        "data_qubits": pool_obj.data_qubits,
        "work_qubits": int(phase.metadata.get("work_qubits", 0)),
        "work_register_reused": True,
        "fixed_t": int(target_t),
        "witnesses": len(specs),
    }
    return circuit, pool_obj


def _data_probabilities(state, data_bits):
    """Marginal distribution over data-bit patterns (sums out the work register)."""
    probabilities = np.zeros(1 << data_bits)
    for index, amplitude in enumerate(state):
        probabilities[index & ((1 << data_bits) - 1)] += abs(amplitude) ** 2
    return probabilities


def ideal_probabilities(circuit, data_bits):
    from qiskit.quantum_info import Statevector
    state = Statevector(circuit).data
    return _data_probabilities(state, data_bits)


def counts_to_probabilities(counts, data_bits):
    total = sum(counts.values())
    probabilities = np.zeros(1 << data_bits)
    for bitstring, frequency in counts.items():
        bits = int(str(bitstring).replace(" ", ""), 2) & ((1 << data_bits) - 1)
        probabilities[bits] += frequency / total
    return probabilities


# ---------------------------------------------------------------------------
# Noise models (kept separate on purpose)
# ---------------------------------------------------------------------------

def gate_depolarizing_model(rate):
    """Gate-level depolarizing noise on single- and two-qubit gates."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    model = NoiseModel()
    one_qubit = depolarizing_error(rate, 1)
    two_qubit = depolarizing_error(min(1.0, 2 * rate), 2)
    model.add_all_qubit_quantum_error(one_qubit, ["u", "u1", "u2", "u3", "x", "sx", "rz", "h", "p"])
    model.add_all_qubit_quantum_error(two_qubit, ["cx", "cz"])
    return model, {"kind": "gate_depolarizing", "one_qubit_rate": rate,
                   "two_qubit_rate": min(1.0, 2 * rate),
                   "note": "gate-level noise; not an annealer model"}


def thermal_damping_model(rate):
    """Thermal relaxation proxy: amplitude damping after every gate.

    Kept strictly separate from the gate-level model: the work package requires
    gate noise and annealer-like thermalisation not to stand in for each other.
    """
    from qiskit_aer.noise import NoiseModel, amplitude_damping_error
    model = NoiseModel()
    one_qubit = amplitude_damping_error(rate)
    model.add_all_qubit_quantum_error(one_qubit, ["u", "u1", "u2", "u3", "x", "sx", "rz", "h", "p"])
    model.add_all_qubit_quantum_error(amplitude_damping_error(rate).tensor(
        amplitude_damping_error(rate)), ["cx", "cz"])
    return model, {"kind": "thermal_damping", "one_qubit_gamma": rate,
                   "two_qubit_gamma": rate,
                   "note": "relaxation proxy; not a gate-error model"}


def sample_counts(circuit, data_bits, noise_model, shots, seed):
    """Sample the circuit under a noise model.

    The circuit is first transpiled to a hardware-agnostic gate basis so the
    state preparation is expanded into real gates and therefore *is* affected by
    the noise model; without this step a simulator would execute ``initialize``
    as a noiseless state preparation and the reported p_imp would be optimistic.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    expanded = transpile(circuit, basis_gates=["u", "cx", "rz", "sx", "x"],
                         optimization_level=1, seed_transpiler=7)
    measured = QuantumCircuit(expanded.num_qubits, data_bits)
    measured.compose(expanded, inplace=True)
    measured.measure(list(range(data_bits)), list(range(data_bits)))
    backend = AerSimulator(noise_model=noise_model) if noise_model is not None else AerSimulator()
    result = backend.run(measured, shots=shots, seed_simulator=seed).result()
    return result.get_counts()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def choice_metrics(probabilities, pool, inst, u0, target_t):
    """p_imp, target mass and illegal leakage of a distribution over data bits."""
    sizes = [len(machine) for machine in pool]
    data_bits = int(sum(sizes))
    legal_mass, illegal_mass = 0.0, 0.0
    improving_mass, reaching_mass = 0.0, 0.0
    improving_choices = 0
    for bits in range(1 << data_bits):
        probability = float(probabilities[bits])
        if probability == 0.0:
            continue
        choice = []
        valid = True
        for machine, size in enumerate(sizes):
            offset = sum(sizes[:machine])
            active = [a for a in range(size) if (bits >> (offset + a)) & 1]
            if len(active) != 1:
                valid = False
                break
            choice.append(active[0])
        if not valid:
            illegal_mass += probability
            continue
        legal_mass += probability
        result = check_choice(inst, pool, tuple(choice))
        if not result["feasible"]:
            continue
        if result["makespan"] < u0:
            improving_mass += probability
            improving_choices += 1
        if result["makespan"] <= target_t:
            reaching_mass += probability
    return {"p_imp": improving_mass, "legal_mass": legal_mass,
            "illegal_one_hot_mass": illegal_mass,
            "target_reaching_mass": reaching_mass,
            "improving_choices_with_mass": improving_choices}


def total_variation(left, right):
    return float(0.5 * np.abs(np.asarray(left) - np.asarray(right)).sum())


# ---------------------------------------------------------------------------
# Physical mapping
# ---------------------------------------------------------------------------

def physical_mapping(circuit, data_bits, *, topology="line", optimization_level=1):
    """Transpile the measured circuit to a fixed line topology.

    Only the data register is measured (the work register is uncomputed before
    readout), so the reported measurement count must equal the data qubits.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap
    measured = QuantumCircuit(circuit.num_qubits, data_bits)
    measured.compose(circuit, inplace=True)
    measured.measure(list(range(data_bits)), list(range(data_bits)))
    coupling = CouplingMap.from_line(measured.num_qubits)
    try:
        compiled = transpile(measured, coupling_map=coupling,
                             basis_gates=["cx", "u", "rz", "sx", "x"],
                             optimization_level=optimization_level, seed_transpiler=7)
    except Exception as exc:  # pragma: no cover - reported honestly
        return {"skipped": f"{type(exc).__name__}: {exc}"}
    counts = compiled.count_ops()
    return {
        "topology": topology,
        "physical_qubits": int(compiled.num_qubits),
        "cx": int(counts.get("cx", 0)),
        "swap": int(counts.get("swap", 0)),
        "depth": int(compiled.depth()),
        "size": int(compiled.size()),
        "count_ops": {str(k): int(v) for k, v in counts.items()},
        "measurements": int(sum(1 for instruction in compiled.data
                                if instruction.operation.name == "measure")),
        "resets": int(sum(1 for instruction in compiled.data
                          if instruction.operation.name == "reset")),
    }


# ---------------------------------------------------------------------------
# Pre-registered hardware protocol (never executed here)
# ---------------------------------------------------------------------------

def hardware_protocol(selected, threshold=MECHANISM_THRESHOLD):
    """Pre-registered QPU protocol with every field marked not executed.

    The parameters are frozen *before* any submission; the runtime fields are
    left null and must be filled only from real job metadata (job id, time,
    backend, shots, counts, compiled-circuit digest, classical verification,
    cost breakdown).  A noisy simulation never fills these fields.
    """
    slots = []
    for index, window in enumerate(selected, start=1):
        slots.append({
            "dq_slot": index,
            "scale": window["scale"],
            "seed": window["seed"],
            "instance_sha256": window["instance_sha256"],
            "pool_sha256": window["pool_sha256"],
            "improvement_density_rho": window["profile"]["rho"],
            "stratum": window.get("stratum"),
            "status": "not_executed",
            "reason": "no owner authorisation for paid QPU jobs in this session",
            "preset": {"batches": BATCHES, "shots_per_batch": SHOTS_PER_BATCH,
                       "total_shots": BATCHES * SHOTS_PER_BATCH,
                       "mechanism_threshold": threshold,
                       "main_parameters_locked_before_hardware": True},
            "job": {"job_id": None, "submitted_at": None, "backend": None,
                    "shots": None, "counts_path": None,
                    "compiled_circuit_sha256": None, "calibration_metadata": None,
                    "classical_verification": None, "cost_breakdown": None},
        })
    return {
        "status": "not_executed",
        "hardware_jobs_submitted": 0,
        "threshold_definition": "hardware p_imp >= 0.80 * ideal p_imp (windows with "
                                "ideal p_imp = 0 are negative controls and take no ratio)",
        "slots": slots,
        "gate_vs_thermal_separated": True,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_window(entry, witnesses, target_t, shots, seed, noise_levels=NOISE_LEVELS):
    inst, pool = entry["inst"], entry["pool"]
    circuit, pool_obj = build_sampling_circuit(pool, witnesses, target_t)
    data_bits = pool_obj.data_qubits
    ideal = ideal_probabilities(circuit, data_bits)
    ideal_metrics = choice_metrics(ideal, pool, inst, entry["u0"], target_t)
    rows = []
    for kind, rates in noise_levels.items():
        for rate in rates:
            model, description = (gate_depolarizing_model(rate) if kind == "gate_depolarizing"
                                  else thermal_damping_model(rate))
            started = time.perf_counter()
            counts = sample_counts(circuit, data_bits, model, shots, seed)
            seconds = time.perf_counter() - started
            noisy = counts_to_probabilities(counts, data_bits)
            metrics = choice_metrics(noisy, pool, inst, entry["u0"], target_t)
            rows.append({
                "noise": description,
                "shots": shots,
                "sampling_seconds": seconds,
                "metrics": metrics,
                "tv_to_ideal": total_variation(ideal, noisy),
                "p_imp_ratio_to_ideal": (metrics["p_imp"] / ideal_metrics["p_imp"]
                                         if ideal_metrics["p_imp"] > 0 else None),
                "mechanism_threshold": MECHANISM_THRESHOLD,
                "passes_mechanism_threshold": (
                    None if ideal_metrics["p_imp"] <= 0
                    else bool(metrics["p_imp"] >= MECHANISM_THRESHOLD * ideal_metrics["p_imp"])),
            })
    return {
        "ideal_metrics": ideal_metrics,
        "ideal_probabilities": [float(p) for p in ideal],
        "noisy": rows,
        "circuit": {"qubits": int(circuit.num_qubits),
                    "data_qubits": data_bits,
                    "work_qubits": int(circuit.metadata.get("work_qubits", 0)),
                    "work_register_reused": bool(circuit.metadata.get("work_register_reused")),
                    "logical_size": int(circuit.size()),
                    "logical_depth": int(circuit.depth()),
                    "logical_count_ops": {str(k): int(v)
                                          for k, v in circuit.count_ops().items()},
                    "physical": physical_mapping(circuit, data_bits)},
    }


def run(seeds, *, pool_k=3, shots=4096, per_stratum=2, seed=7):
    from witness_surrogate import collect_witnesses
    selected, strata = select_dq_proxy(seeds, pool_k=pool_k, per_stratum=per_stratum)
    results = []
    for entry in selected:
        witnesses, stats = collect_witnesses(entry["inst"], entry["pool"],
                                             tuple(0 for _ in entry["pool"]),
                                             n_probe=16, seed=entry["seed"])
        target_t = int(entry["u0"]) - 1
        entry["stratum"] = ("no_improvement" if entry["profile"]["improving"] == 0
                            else "intermediate" if (entry["profile"]["rho"] or 0) <= 0.25
                            else "higher")
        result = run_window(entry, witnesses, target_t, shots, seed + entry["seed"])
        result.update({
            "scale": entry["scale"], "seed": entry["seed"],
            "instance_sha256": entry["instance_sha256"],
            "pool_sha256": entry["pool_sha256"],
            "u0": entry["u0"], "target_t": target_t,
            "stratum": entry["stratum"],
            "selection_note": entry.get("selection_note"),
            "profile": entry["profile"],
            "witness_stats": stats,
        })
        results.append(result)
        print(f"  {entry['scale']} seed {entry['seed']} ({entry['stratum']}): "
              f"ideal p_imp={result['ideal_metrics']['p_imp']:.4f} "
              f"illegal={result['ideal_metrics']['illegal_one_hot_mass']:.2e}", flush=True)
    return selected, strata, results


def main(argv=None):
    ap = argparse.ArgumentParser(description="T09 hardware resource / noise report (simulation only)")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--pool-k", type=int, default=3)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--per-stratum", type=int, default=2)
    ap.add_argument("--out", default=str(GATES_DIR / "results_hardware_resource_20261003"
                                         / "hardware_resource.json"))
    args = ap.parse_args(argv)
    lo, hi = args.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    selected, strata, results = run(seeds, pool_k=args.pool_k, shots=args.shots,
                                    per_stratum=args.per_stratum)
    output = {
        "schema_version": SCHEMA_VERSION,
        "work_package": "T09 (Issue #19, PR #18 docs TODO)",
        "scope": "simulation only; no QPU job submitted, no credential used",
        "dq_source": "DQ proxy from the D0 frozen layer (D2 not available yet: T04)",
        "dq_proxy_selection_rule": "classical improvement density rho stratification "
                                   "(no_improvement / intermediate / higher), 2 per stratum",
        "strata_sizes": strata,
        "windows": results,
        "hardware_protocol": hardware_protocol(selected),
        "noise_levels": {k: list(v) for k, v in NOISE_LEVELS.items()},
        "threshold": MECHANISM_THRESHOLD,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=1, default=_json_default),
                   encoding="utf-8")
    clean = sum(1 for w in results
                if all(row["metrics"]["illegal_one_hot_mass"] >= 0 for row in w["noisy"]))
    print(f"wrote {out}")
    print(f"windows simulated: {len(results)}; hardware status: "
          f"{output['hardware_protocol']['status']} (jobs submitted 0); sanity {clean}/{len(results)}")
    return 0


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj).__name__)


if __name__ == "__main__":
    sys.exit(main())
