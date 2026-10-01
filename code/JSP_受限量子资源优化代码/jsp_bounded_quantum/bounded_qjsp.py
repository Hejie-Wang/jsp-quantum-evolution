#!/usr/bin/env python3
"""Bounded local quantum circuit experiments for JSP (not a global exact solver).

Each qubit selects one of k disjoint adjacent swaps of an incumbent schedule.
All 2**k choices are evaluated CLASSICALLY, then compiled as an exact diagonal
phase. This exponential cost is capped in k, not hidden in a free oracle.
No quantum advantage is claimed: exact minimization of the same small table
is included as a baseline. Default execution is local ideal simulation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import math
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from qjsp import Instance, demo_instance, evaluate_orders, validate_schedule, export_schedule, write_json


@dataclass
class Budget:
    bits: int = 4
    layers: int = 2
    max_two_qubit: int = 200
    max_depth: int = 600
    max_duration_us: float = 30.0
    max_cost_evaluations: int = 320
    max_rounds: int = 20
    shots: int = 1024
    train_evaluations: int = 96

    def validate(self):
        if not 1 <= self.bits <= 6:
            raise ValueError("this prototype caps the local register at 1..6 qubits")
        if not 1 <= self.layers <= 8:
            raise ValueError("layers must lie in 1..8")
        if any(x < 1 for x in (self.max_two_qubit, self.max_depth, self.max_cost_evaluations,
                               self.max_rounds, self.shots, self.train_evaluations)):
            raise ValueError("budgets must be positive")
        if not math.isfinite(self.max_duration_us) or self.max_duration_us <= 0:
            raise ValueError("max_duration_us must be positive")

    def chosen_bits(self):
        self.validate()
        k = self.bits
        # Exact parity-network phase worst-case, before hardware routing.
        while k > 1 and self.layers*((k-2)*2**k+2) > self.max_two_qubit:
            k -= 1
        return k


def dispatch_initial(inst):
    """Classical feasible initializer; no quantum optimization occurs here."""
    next_op = [0]*inst.jobs
    ready_job, ready_machine = [0]*inst.jobs, [0]*inst.machines
    orders = [[] for _ in range(inst.machines)]
    for _ in range(inst.operations):
        choices = []
        for j, k in enumerate(next_op):
            if k < inst.machines:
                m = int(inst.routes[j, k]); p = int(inst.durations[j, k])
                finish = max(ready_job[j], ready_machine[m])+p
                choices.append((finish, p, j, k, m))
        finish, _, j, k, m = min(choices)
        orders[m].append(j*inst.machines+k)
        ready_job[j] = ready_machine[m] = finish
        next_op[j] += 1
    result = evaluate_orders(inst, orders)
    if not result["feasible"]:
        raise RuntimeError("initializer produced a cycle")
    validate_schedule(inst, result["starts"])
    return orders


def critical_moves(inst, orders):
    result = evaluate_orders(inst, orders)
    if not result["feasible"]:
        raise ValueError("incumbent must be feasible")
    starts = result["starts"]
    predecessor = [None]*inst.operations
    position = {}
    for m, row in enumerate(orders):
        for k, v in enumerate(row):
            position[v] = (m, k)
            if k:
                predecessor[v] = row[k-1]
    v = max(range(inst.operations), key=lambda v: starts[v]+inst.processing[v])
    moves = []
    while starts[v] > 0:
        pred_machine = predecessor[v]
        if pred_machine is not None and starts[pred_machine]+inst.processing[pred_machine] == starts[v]:
            m, k = position[pred_machine]
            moves.append((m, k)); v = pred_machine
        elif v % inst.machines and starts[v-1]+inst.processing[v-1] == starts[v]:
            v -= 1
        else:
            raise RuntimeError("cannot trace critical predecessor")
    return moves


def select_moves(inst, orders, bits, rng):
    critical = list(dict.fromkeys(critical_moves(inst, orders)))
    candidates = [(m, k) for m, row in enumerate(orders) for k in range(len(row)-1)]
    rng.shuffle(critical); rng.shuffle(candidates)
    # Occasionally explore a random block instead of only critical-path moves.
    sequence = critical+candidates if rng.random() < 0.8 else candidates
    selected, used = [], set()
    for m, k in sequence:
        positions = {(m, k), (m, k+1)}
        if used.isdisjoint(positions):
            selected.append((m, k)); used.update(positions)
            if len(selected) == bits:
                break
    return selected


def apply_moves(orders, moves, bit_index):
    """Qiskit convention: move i is controlled by bit i (least-significant first)."""
    if bit_index < 0 or bit_index >= 2**len(moves):
        raise ValueError("bit index outside local basis")
    result = [list(row) for row in orders]
    used = set()
    for i, (m, k) in enumerate(moves):
        if not 0 <= m < len(result) or not 0 <= k < len(result[m])-1:
            raise ValueError("invalid adjacent swap")
        positions = {(m, k), (m, k+1)}
        if not used.isdisjoint(positions):
            raise ValueError("local moves must have disjoint positions")
        used.update(positions)
        if (bit_index >> i) & 1:
            result[m][k], result[m][k+1] = result[m][k+1], result[m][k]
    return result


def local_cost_table(inst, orders, moves):
    if not 1 <= len(moves) <= 6:
        raise ValueError("local table limited to 1..6 qubits")
    costs = [evaluate_orders(inst, apply_moves(orders, moves, x))["cost"]
             for x in range(2**len(moves))]
    return np.array(costs, dtype=np.int64)


def hadamard_transform(a):
    out = np.asarray(a, dtype=float).copy()
    if not len(out) or len(out) & (len(out)-1):
        raise ValueError("table length must be a power of two")
    step = 1
    while step < len(out):
        view = out.reshape(-1, 2*step)
        left, right = view[:, :step].copy(), view[:, step:].copy()
        view[:, :step], view[:, step:] = left+right, left-right
        step *= 2
    return out


def phase_model(costs, keep_terms=None):
    costs = np.asarray(costs, dtype=float)
    k = (len(costs)-1).bit_length()
    if not 1 <= k <= 6 or len(costs) != 2**k or not np.all(np.isfinite(costs)):
        raise ValueError("expected a finite 2**k cost table with 1 <= k <= 6")
    low, scale = float(costs.min()), float(max(1, costs.max()-costs.min()))
    normalized = (costs-low)/scale
    coefficients = hadamard_transform(normalized)/len(costs)
    masks = [i for i in range(1, len(costs)) if abs(coefficients[i]) > 1e-13]
    if keep_terms is not None:
        if keep_terms < 0:
            raise ValueError("keep_terms must be nonnegative")
        masks = sorted(masks, key=lambda i: -abs(coefficients[i]))[:keep_terms]
    selected = np.zeros_like(coefficients)
    selected[0] = coefficients[0]
    for mask in masks:
        selected[mask] = coefficients[mask]
    approximation = hadamard_transform(selected)
    terms = [(int(i), float(selected[i])) for i in sorted(masks)]
    return {"bits": k, "offset": low, "scale": scale,
            "constant": float(selected[0]), "normalized": normalized,
            "approximation": approximation, "terms": terms,
            "sup_error": float(np.max(np.abs(normalized-approximation))),
            "cx_per_layer_before_routing": sum(2*(mask.bit_count()-1) for mask, _ in terms)}


def probabilities(potential, angles):
    """Exact ideal simulation of the specified SHALLOW variational circuit."""
    potential = np.asarray(potential)
    psi = np.full(len(potential), 1/np.sqrt(len(potential)), dtype=complex)
    k = (len(potential)-1).bit_length()
    indices = np.arange(len(potential))
    for gamma, beta in np.asarray(angles).reshape(-1, 2):
        psi *= np.exp(-1j*gamma*potential)
        for bit in range(k):
            psi = math.cos(beta)*psi+1j*math.sin(beta)*psi[indices ^ (1 << bit)]
    p = np.abs(psi)**2
    if abs(float(p.sum())-1) > 1e-10:
        raise RuntimeError("norm drift")
    return p/p.sum()


def fit_angles(phase, layers=2, evaluations=96, seed=7):
    """Bounded classical parameter fitting using a <=64-state ideal simulator.

    Minimize expected ORIGINAL normalized cost, not optimum-hit probability.
    This training does not spend hardware shots and is counted separately.
    """
    if layers < 1 or evaluations < 1:
        raise ValueError("positive layer and evaluation budgets required")
    rng = np.random.default_rng(seed)
    best_value, best_angles, count = float("inf"), None, 0
    def objective(x):
        nonlocal best_value, best_angles, count
        if count >= evaluations:
            return best_value
        p = probabilities(phase["approximation"], x)
        value = float(p @ phase["normalized"])
        count += 1
        if value < best_value:
            best_value, best_angles = value, np.asarray(x).copy()
        return value
    objective(np.zeros(2*layers))
    seeds = min(max(1, evaluations//4), 24)
    bounds = [(-2*np.pi, 2*np.pi), (-np.pi/2, np.pi/2)]*layers
    for _ in range(min(seeds, evaluations-count)):
        objective(np.array([rng.uniform(a, b) for a, b in bounds]))
    if count < evaluations:
        minimize(objective, best_angles, method="Powell", bounds=bounds,
                 options={"maxfev": evaluations-count, "xtol": 1e-5, "ftol": 1e-7})
    angles = best_angles.reshape(layers, 2)
    return angles, {"ideal_training_evaluations": count, "mean_normalized_cost": best_value,
                    "max_probability_event_error_from_phase_truncation_bound":
                    min(1.0, float(np.abs(angles[:, 0]).sum())*phase["sup_error"])}


def build_circuit(phase, angles, measure=False):
    from qiskit import QuantumCircuit
    k = phase["bits"]
    qc = QuantumCircuit(k, name="bounded_jsp")
    qc.h(range(k))
    for gamma, beta in np.asarray(angles).reshape(-1, 2):
        # All Z strings commute, so this cost phase has NO Trotter error.
        for mask, coefficient in phase["terms"]:
            support = [i for i in range(k) if (mask >> i) & 1]
            target = support[-1]
            for control in support[:-1]:
                qc.cx(control, target)
            qc.rz(2*float(gamma)*coefficient, target)
            for control in reversed(support[:-1]):
                qc.cx(control, target)
        for bit in range(k):
            qc.rx(-2*float(beta), bit)
    if measure:
        qc.measure_all()
    return qc


def check_compiled_budget(circuit, budget, target=None):
    """A hard gate/depth gate; duration is enforced when calibration is present."""
    two = sum(i.operation.num_qubits == 2 for i in circuit.data
              if i.operation.name not in ("barrier", "delay"))
    depth = circuit.depth(filter_function=lambda i: i.operation.name not in ("barrier", "delay"))
    used = {q for i in circuit.data if i.operation.name not in ("barrier", "delay") for q in i.qubits}
    duration = None
    if target is not None:
        duration = float(circuit.estimate_duration(target, unit="s"))
    if two > budget.max_two_qubit or depth > budget.max_depth:
        raise ValueError(f"compiled circuit over budget: 2q={two}, depth={depth}")
    if duration is not None and duration*1e6 > budget.max_duration_us:
        raise ValueError(f"compiled duration {duration*1e6:.3f} us exceeds budget")
    return {"two_qubit_gates": int(two), "depth_excluding_delays": int(depth),
            "active_instruction_qubits": len(used), "scheduled_duration_seconds": duration,
            "gate_counts": {k: int(v) for k, v in circuit.count_ops().items()}}


def simulate_search(inst, budget=None, seed=7, mode="ideal_quantum", initial_orders=None):
    budget = budget or Budget()
    bits = budget.chosen_bits()
    if mode not in ("ideal_quantum", "uniform", "exact_local"):
        raise ValueError("unknown local engine")
    rng = np.random.default_rng(seed)
    start = perf_counter()
    orders = dispatch_initial(inst) if initial_orders is None else [list(row) for row in initial_orders]
    schedule = evaluate_orders(inst, orders)
    if not schedule["feasible"]:
        raise ValueError("initial orders must be feasible")
    initial = schedule["makespan"]
    trace, cost_evaluations, parameter_evaluations, last_request = [], 0, 0, None
    for round_index in range(budget.max_rounds):
        moves = select_moves(inst, orders, bits, rng)
        if not moves or cost_evaluations+2**len(moves) > budget.max_cost_evaluations:
            break
        costs = local_cost_table(inst, orders, moves)
        cost_evaluations += len(costs)
        phase = phase_model(costs)
        before = schedule["makespan"]
        if mode == "ideal_quantum":
            angles, training = fit_angles(phase, budget.layers, budget.train_evaluations,
                                         seed+1009*round_index)
            parameter_evaluations += training["ideal_training_evaluations"]
            p = probabilities(phase["approximation"], angles)
            measured = rng.choice(len(costs), size=budget.shots, p=p)
            candidate = int(measured[np.argmin(costs[measured])])
            last_request = {"instance": inst.as_dict(), "orders_zero_based": orders,
                            "moves": [list(x) for x in moves], "costs": costs.tolist(),
                            "angles": angles.tolist(), "budgets": budget.__dict__,
                            "note": "Full global JSP costs for a restricted local move table; no global-optimum claim."}
        elif mode == "uniform":
            p = np.full(len(costs), 1/len(costs))
            measured = rng.choice(len(costs), size=budget.shots, p=p)
            candidate = int(measured[np.argmin(costs[measured])])
        else:
            # Randomize equal optima so the baseline can move across plateaus,
            # instead of systematically choosing the incumbent (index zero).
            candidate = int(rng.choice(np.flatnonzero(costs == costs.min())))
            p = np.eye(1, len(costs), candidate).ravel()
        if costs[candidate] <= before and costs[candidate] < inst.penalty:
            # Quantum/uniform engines select only MEASURED candidates. The
            # incumbent is always retained if every measured candidate is worse.
            orders = apply_moves(orders, moves, candidate)
            schedule = evaluate_orders(inst, orders)
            validate_schedule(inst, schedule["starts"])
        trace.append({"round": round_index+1, "bits": len(moves), "states": len(costs),
                      "incumbent_before": before, "incumbent_after": schedule["makespan"],
                      "exact_local_best_diagnostic": int(costs.min()),
                      "ideal_probability_of_strict_improvement": float(p[costs < before].sum()),
                      "cx_before_routing": phase["cx_per_layer_before_routing"]*budget.layers,
                      "cost_evaluations_cumulative": cost_evaluations})
    result = {"instance": inst.name, "mode": mode, "seed": seed,
              "requested_budget": budget.__dict__, "chosen_bits": bits,
              "initial_classical_makespan": initial, "final_makespan": schedule["makespan"],
              "final_orders_zero_based": orders, "schedule": schedule,
              "local_table_cost_evaluations": cost_evaluations,
              "ideal_parameter_objective_evaluations": parameter_evaluations,
              "seconds": perf_counter()-start, "hardware_jobs_submitted": 0,
              "trace": trace,
              "limitations": ["Local-search heuristic; no global-optimum guarantee.",
                              "All local costs were already evaluated classically; exact_local is the necessary baseline.",
                              "An ideal sampled search is not a quantum-hardware performance measurement."]}
    return result, last_request


def request_circuit(request):
    inst = Instance(np.array(request["instance"]["durations"]),
                    np.array(request["instance"]["machines_1_based"])-1)
    costs = local_cost_table(inst, request["orders_zero_based"], request["moves"])
    if costs.tolist() != request["costs"]:
        raise ValueError("request cost table does not match independently decoded global JSP")
    phase = phase_model(costs)
    angles = np.asarray(request["angles"], dtype=float)
    if not np.all(np.isfinite(angles)) or angles.ndim != 2 or angles.shape[1] != 2:
        raise ValueError("invalid angles")
    budget = Budget(**request["budgets"]); budget.validate()
    if len(request["moves"]) > budget.bits or len(angles) != budget.layers:
        raise ValueError("request dimensions exceed declared budget")
    return inst, phase, build_circuit(phase, angles), budget


def run_request(request, output_dir, backend=None):
    """Compile and export one circuit. Never submits a quantum job."""
    from qiskit import qpy
    from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
    from qiskit.quantum_info import Statevector
    inst, phase, circuit, budget = request_circuit(request)
    expected = probabilities(phase["approximation"], request["angles"])
    error = float(np.max(np.abs(Statevector.from_instruction(circuit).probabilities()-expected)))
    if error > 1e-10:
        raise RuntimeError("Qiskit circuit disagrees with independent amplitude simulation")
    measured = circuit.copy(); measured.measure_all()
    if backend is None:
        # Synthetic connected line: gate counts only, no device time prediction.
        k = phase["bits"]
        kwargs = {"basis_gates": ["rz", "sx", "x", "cx"],
                  "coupling_map": CouplingMap.from_line(k) if k > 1 else None}
        manager = generate_preset_pass_manager(optimization_level=3, seed_transpiler=7, **kwargs)
        compiled = manager.run(measured)
        stats = check_compiled_budget(compiled, budget)
        # For a synthetic k-wire target, remove measurements and map the final
        # physical axes back to classical output bits. This checks routing too.
        verify_compiled_probabilities(compiled, expected)
    else:
        manager = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                               seed_transpiler=7, scheduling_method="alap")
        compiled = manager.run(measured)
        stats = check_compiled_budget(compiled, budget, backend.target)
        verify_compiled_probabilities(compiled, expected)
    path = Path(output_dir); path.mkdir(parents=True, exist_ok=True)
    with (path/"circuit.qpy").open("wb") as f:
        qpy.dump(compiled, f)
    report = {"backend": "synthetic_line_no_noise_or_timing" if backend is None else backend.name,
              "independent_ideal_probability_max_error": error,
              "post_transpile_ideal_probabilities_checked": True,
              "resources": stats, "hardware_jobs_submitted": 0,
              "shots_requested": budget.shots, "request": request}
    write_json(path/"compiled_report.json", report)
    return report, compiled


def verify_compiled_probabilities(compiled, expected, max_active=10):
    """Check logical output probabilities AFTER layout/routing, ignoring noise.

    Prunes idle backend wires and delays; reads the actual measurement map.
    No simulation of all 156 allocated backend wires is attempted.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    active = sorted({compiled.find_bit(q).index for item in compiled.data
                     if item.operation.name not in ("barrier", "delay", "measure")
                     for q in item.qubits} |
                    {compiled.find_bit(item.qubits[0]).index for item in compiled.data
                     if item.operation.name == "measure"})
    if len(active) > max_active:
        raise ValueError("too many active physical wires for post-transpile audit")
    rank = {q: i for i, q in enumerate(active)}
    small = QuantumCircuit(len(active)); mapping = {}; measured_wires = set()
    for item in compiled.data:
        name = item.operation.name
        if name in ("barrier", "delay"):
            continue
        if name == "measure":
            q = rank[compiled.find_bit(item.qubits[0]).index]
            c = compiled.find_bit(item.clbits[0]).index
            mapping[c] = q
            measured_wires.add(q)
        else:
            wires = [rank[compiled.find_bit(q).index] for q in item.qubits]
            if measured_wires.intersection(wires) or name == "reset" or item.clbits:
                raise ValueError("audit supports unitary circuits with terminal measurements")
            small.append(item.operation, wires)
    if set(mapping) != set(range((len(expected)-1).bit_length())):
        raise ValueError("unexpected measurement mapping")
    actual = np.zeros(len(expected))
    for physical_index, probability in enumerate(Statevector.from_instruction(small).probabilities()):
        logical_index = sum(((physical_index >> q) & 1) << c for c, q in mapping.items())
        actual[logical_index] += probability
    error = float(np.max(np.abs(actual-expected)))
    if error > 1e-9:
        raise RuntimeError(f"post-transpile distribution mismatch {error}")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search")
    p.add_argument("--instance")
    p.add_argument("--initial-orders", help="JSON containing orders_zero_based (e.g. hardware_once output)")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--train-evaluations", type=int, default=96)
    p.add_argument("--max-cost-evaluations", type=int, default=320)
    p.add_argument("--max-two-qubit", type=int, default=200)
    p.add_argument("--mode", choices=["ideal_quantum", "uniform", "exact_local"], default="ideal_quantum")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", default="my_result")
    p = sub.add_parser("compile")
    p.add_argument("--request", required=True)
    p.add_argument("--output-dir", default="compiled")
    p.add_argument("--backend", help="optional IBM device; reads calibration but submits no jobs")
    p.add_argument("--ibm-instance", help="explicit IBM instance, required for real backend access")
    args = parser.parse_args()
    if args.command == "search":
        inst = Instance.read(args.instance) if args.instance else demo_instance()
        budget = Budget(bits=args.bits, layers=args.layers, max_rounds=args.rounds,
                        max_two_qubit=args.max_two_qubit, max_cost_evaluations=args.max_cost_evaluations,
                        shots=args.shots, train_evaluations=args.train_evaluations)
        initial = None
        if args.initial_orders:
            initial = json.loads(Path(args.initial_orders).read_text())["orders_zero_based"]
        result, request = simulate_search(inst, budget, args.seed, args.mode, initial)
        out = Path(args.output_dir)
        write_json(out/"search.json", result)
        export_schedule(inst, result["schedule"], out/"schedule.csv")
        if request is not None:
            write_json(out/"last_request.json", request)
        print(json.dumps({k: result[k] for k in ("mode", "initial_classical_makespan", "final_makespan",
                                                "local_table_cost_evaluations", "seconds")}, indent=2))
    else:
        request = json.loads(Path(args.request).read_text())
        backend = None
        if args.backend:
            if not args.ibm_instance:
                parser.error("--ibm-instance is required; no automatic paid-instance selection")
            from qiskit_ibm_runtime import QiskitRuntimeService
            backend = QiskitRuntimeService(instance=args.ibm_instance).backend(args.backend)
        result, _ = run_request(request, args.output_dir, backend)
        print(json.dumps(result["resources"], indent=2))


if __name__ == "__main__":
    main()
