#!/usr/bin/env python3
"""Permutation-space JSP quantum evolution, simulated on a classical CPU.

H(s) = -(1-s)/d sum(S_mk) + s diag(F-LB).
This implements the prescribed unitary, not a classical scheduling heuristic.
Full cost enumeration and state vectors restrict simulation to small instances.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import permutations, product
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


class CapacityError(ValueError):
    """Requested full simulation exceeds a limit before allocating its basis."""


@dataclass
class Instance:
    durations: np.ndarray
    routes: np.ndarray  # zero-based machine IDs
    name: str = "instance"

    def __post_init__(self):
        p, r = np.asarray(self.durations), np.asarray(self.routes)
        if p.ndim != 2 or p.shape != r.shape or not all(p.shape):
            raise ValueError("durations and routes must be nonempty equal 2D arrays")
        if p.dtype.kind not in "iu" or r.dtype.kind not in "iu":
            raise ValueError("durations and routes must contain integers")
        if np.any(p <= 0):
            raise ValueError("durations must be positive")
        self.jobs, self.machines = p.shape
        if any(sorted(map(int, row)) != list(range(self.machines)) for row in r):
            raise ValueError("each job must visit every machine exactly once")
        # Python integers below prevent overflow in graph arithmetic.
        self.total_duration = sum(map(int, p.flat))
        if self.total_duration + 1 >= 2**53:
            raise ValueError("costs must stay below 2**53 for this float64 simulator")
        self.durations = np.array(p, dtype=np.int64, copy=True)
        self.routes = np.array(r, dtype=np.int64, copy=True)
        self.operations = self.jobs * self.machines
        self.processing = tuple(map(int, self.durations.flat))
        self.groups = tuple(tuple(v for v in range(self.operations)
                                  if int(self.routes.flat[v]) == m)
                            for m in range(self.machines))
        self.job_arcs = tuple((j*self.machines+k, j*self.machines+k+1)
                             for j in range(self.jobs)
                             for k in range(self.machines-1))
        self.penalty = self.total_duration + 1
        self.lower_bound = max(
            max(sum(map(int, row)) for row in self.durations),
            max(sum(self.processing[v] for v in g) for g in self.groups))

    @classmethod
    def read(cls, path):
        path = Path(path)
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            # JSON machine IDs are explicitly one-based, as in the text input.
            return cls(np.asarray(obj["durations"]),
                       np.asarray(obj["machines_1_based"])-1,
                       obj.get("name", path.stem))
        a = np.loadtxt(path, dtype=np.int64, ndmin=2)
        if len(a) % 2:
            raise ValueError("text input: J duration rows followed by J route rows")
        return cls(a[:len(a)//2], a[len(a)//2:]-1, path.stem)

    def as_dict(self):
        return {"name": self.name, "durations": self.durations.tolist(),
                "machines_1_based": (self.routes+1).tolist()}


def demo_instance(jobs=3, machines=3, seed=7):
    if (jobs, machines) == (3, 3):
        return Instance(np.array([[3, 2, 2], [2, 1, 4], [4, 3, 1]]),
                        np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]), "demo_3x3")
    rng = np.random.default_rng(seed)
    return Instance(rng.integers(1, 10, size=(jobs, machines)),
                    np.array([rng.permutation(machines) for _ in range(jobs)]),
                    f"demo_{jobs}x{machines}_seed{seed}")


def evaluate_orders(inst, orders, validate=True):
    """Return the exact earliest schedule for fixed machine orders, or a cycle.

    Operation ID = job * number_of_machines + position_in_job (zero based).
    This graph decoder does not search for a better order.
    """
    if validate:
        if len(orders) != inst.machines:
            raise ValueError("expected one order for each machine")
        for row, group in zip(orders, inst.groups):
            if any(not isinstance(x, (int, np.integer)) for x in row):
                raise ValueError("operation IDs must be integers")
            if sorted(row) != list(group):
                raise ValueError("each order must permute exactly its machine's operations")
    adjacent = [[] for _ in range(inst.operations)]
    degree = [0] * inst.operations
    for u, v in inst.job_arcs:
        adjacent[u].append(v)
        degree[v] += 1
    for row in orders:
        for u, v in zip(row, row[1:]):
            adjacent[u].append(v)
            degree[v] += 1
    queue = [v for v in range(inst.operations) if degree[v] == 0]
    starts = [0] * inst.operations
    for u in queue:
        finish = starts[u] + inst.processing[u]
        for v in adjacent[u]:
            if finish > starts[v]:
                starts[v] = finish
            degree[v] -= 1
            if degree[v] == 0:
                queue.append(v)
    if len(queue) != inst.operations:
        return {"feasible": False, "cost": inst.penalty, "reason": "directed_cycle"}
    makespan = max(s+p for s, p in zip(starts, inst.processing))
    return {"feasible": True, "cost": makespan, "makespan": makespan, "starts": starts}


def validate_schedule(inst, starts):
    """Independent direct precedence and non-overlap check for output schedules."""
    if len(starts) != inst.operations or any(
            not isinstance(s, (int, np.integer)) or s < 0 for s in starts):
        raise ValueError("invalid start times")
    for u, v in inst.job_arcs:
        if starts[u] + inst.processing[u] > starts[v]:
            raise ValueError("job precedence violated")
    for group in inst.groups:
        row = sorted(group, key=lambda v: starts[v])
        for u, v in zip(row, row[1:]):
            if starts[u] + inst.processing[u] > starts[v]:
                raise ValueError("machine overlap")
    return max(s+p for s, p in zip(starts, inst.processing))


def estimate_resources(inst, layers=2000):
    f = math.factorial(inst.jobs)
    dimension = f ** inst.machines
    d = inst.machines * (inst.jobs-1)
    # Budget, not an allocator measurement: cost, phases, state, work arrays,
    # sampling buffers, and a conservative allowance for local Python tables.
    planned_bytes = 128*dimension + inst.machines*f*(128+64*inst.jobs+8*(inst.jobs-1))
    log_dim = math.log10(dimension)
    return {
        "name": inst.name, "jobs": inst.jobs, "machines": inst.machines,
        "operations": inst.operations,
        "basis_states_exact": str(dimension) if log_dim < 100 else None,
        "log10_basis_states": log_dim,
        "log10_complex128_state_bytes": log_dim+math.log10(16),
        "simulator_planned_peak_bytes": planned_bytes if planned_bytes < 2**53 else None,
        "log10_simulator_planned_peak_bytes": math.log10(planned_bytes),
        "compact_per_machine_rank_qubits": inst.machines*(f-1).bit_length(),
        "position_list_qubits": inst.machines*inst.jobs*(inst.jobs-1).bit_length(),
        "cost_bits": inst.penalty.bit_length(), "adjacent_swap_generators": d,
        "total_duration": inst.total_duration, "cycle_penalty": inst.penalty,
        "simple_lower_bound": inst.lower_bound,
        "layers": layers,
        "log10_amplitude_swap_updates": math.log10(layers)+log_dim+math.log10(d) if d else None,
        "logical_qubits_exclude_oracle_and_error_correction": True,
        "wall_time_estimate": None,
        "wall_time_note": "Use measured small-instance benchmarks; quantum hardware time is not specified."
    }


def check_capacity(inst, max_states=200_000, max_memory_mib=512):
    if max_states < 1 or not math.isfinite(max_memory_mib) or max_memory_mib <= 0:
        raise ValueError("capacity limits must be positive")
    resource = estimate_resources(inst)
    dim = math.factorial(inst.jobs) ** inst.machines
    if dim > max_states:
        raise CapacityError(f"Full simulation needs about 10^{resource['log10_basis_states']:.3f} "
                            f"states; max_states={max_states}. Use 'estimate' or 'decode'.")
    if resource["log10_simulator_planned_peak_bytes"] > math.log10(max_memory_mib*2**20):
        raise CapacityError("Estimated simulator working memory exceeds --max-memory-mib")
    return resource


class PermutationModel:
    """Explicit small-instance basis. Capacity is checked before enumeration."""
    def __init__(self, inst, max_states=200_000, max_memory_mib=512):
        self.resources = check_capacity(inst, max_states, max_memory_mib)
        self.instance = inst
        self.local = [list(permutations(g)) for g in inst.groups]
        self.shape = tuple(len(a) for a in self.local)
        self.dimension = math.prod(self.shape)
        self.swaps = []  # (tensor axis, local permutation-index map)
        for m, rows in enumerate(self.local):
            rank = {p: i for i, p in enumerate(rows)}
            for k in range(inst.jobs-1):
                mapping = np.empty(len(rows), dtype=np.int64)
                for i, row in enumerate(rows):
                    p = list(row)
                    p[k], p[k+1] = p[k+1], p[k]
                    mapping[i] = rank[tuple(p)]
                self.swaps.append((m, mapping))
        self.d = len(self.swaps)
        costs = np.empty(self.dimension, dtype=np.int64)
        for i, orders in enumerate(product(*self.local)):
            costs[i] = evaluate_orders(inst, orders, validate=False)["cost"]
        self.costs = costs
        # Only a valid, cheaply computed lower bound enters the Hamiltonian.
        # The exact optimum of this enumerated table is used ONLY for diagnostics.
        self.potential = (costs-inst.lower_bound).astype(np.float64)

    def uniform_state(self):
        return np.full(self.dimension, 1/math.sqrt(self.dimension), dtype=np.complex128)

    def orders_at(self, index):
        if not 0 <= int(index) < self.dimension:
            raise ValueError("basis index out of range")
        indices = np.unravel_index(int(index), self.shape)
        return [list(self.local[m][rank]) for m, rank in enumerate(indices)]

    def swap_state(self, state, swap):
        axis, mapping = swap
        return np.take(state.reshape(self.shape), mapping, axis=axis).reshape(-1)

    def apply_layer(self, state, s, h):
        """Cost phase, then exact involution rotations in ascending (m,k) order.

        This is a first-order product formula, not an exact time-ordered
        exponential. No normalization is applied to hide integration errors.
        """
        psi = state * np.exp((-1j*s*h)*self.potential)
        if self.d:
            beta = (1-s)*h/self.d
            cosine, sine = math.cos(beta), 1j*math.sin(beta)
            for swap in self.swaps:
                psi = cosine*psi + sine*self.swap_state(psi, swap)
        return psi

    def apply_hamiltonian(self, state, s):
        out = s*self.potential*state
        if self.d:
            for swap in self.swaps:
                out = out - ((1-s)/self.d)*self.swap_state(state, swap)
        return out

    def dense_hamiltonian(self, s, max_dimension=256):
        if self.dimension > max_dimension:
            raise CapacityError(f"Dense spectrum restricted to {max_dimension} states")
        eye = np.eye(self.dimension, dtype=np.complex128)
        return np.column_stack([self.apply_hamiltonian(eye[:, i], s)
                                for i in range(self.dimension)])


def state_statistics(model, state):
    probability = np.abs(state)**2
    norm = float(probability.sum())
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError("invalid state norm")
    probability /= norm  # measurement convention; evolution is not renormalized
    feasible = model.costs < model.instance.penalty
    optimum = int(model.costs.min())
    best = model.costs == optimum
    p_feasible, p_best = float(probability[feasible].sum()), float(probability[best].sum())
    return {
        "norm_squared": norm,
        "feasible_probability": p_feasible,
        "optimal_probability_diagnostic": p_best,
        "enumerated_optimum_diagnostic": optimum,
        "optimum_degeneracy_diagnostic": int(best.sum()),
        "uniform_optimal_probability_diagnostic": float(best.mean()),
        "expected_cost_including_cycle_penalty": float(probability @ model.costs),
        "conditional_feasible_mean_makespan": float(probability[feasible] @ model.costs[feasible]/p_feasible)
        if p_feasible else None,
    }


def evolve(model, tau=20.0, layers=2000, s_final=0.95, progress=False):
    if not math.isfinite(tau) or tau <= 0 or layers < 1:
        raise ValueError("tau and layers must be positive")
    if not math.isfinite(s_final) or not 0 < s_final <= 1:
        raise ValueError("s_final must lie in (0,1]")
    psi = model.uniform_state()
    trace, max_norm_error = [], 0.0
    stride = max(1, layers//10)
    h = tau/layers
    for ell in range(layers):
        s = s_final*(ell+0.5)/layers
        psi = model.apply_layer(psi, s, h)
        if (ell+1) % stride == 0 or ell+1 == layers:
            stats = state_statistics(model, psi)
            max_norm_error = max(max_norm_error, abs(stats["norm_squared"]-1))
            trace.append({"completed_layers": ell+1, "s_boundary": s_final*(ell+1)/layers,
                          "feasible_probability": stats["feasible_probability"],
                          "optimal_probability_diagnostic": stats["optimal_probability_diagnostic"],
                          "norm_squared": stats["norm_squared"]})
            if progress:
                print(f"evolution {ell+1}/{layers}; p(feasible)={stats['feasible_probability']:.5f}; "
                      f"p(optimal, diagnostic)={stats['optimal_probability_diagnostic']:.5f}", file=sys.stderr)
    return psi, {"tau_dimensionless": tau, "layers": layers, "s_final": s_final,
                 "step_size": h, "method": "midpoint schedule + first-order cost-then-swap product",
                 "max_recorded_norm_squared_error": max_norm_error, "trace": trace}


def sample_schedule(model, state, shots=1000, seed=7):
    if shots < 1:
        raise ValueError("shots must be positive")
    probabilities = np.abs(state)**2
    norm = float(probabilities.sum())
    if not math.isfinite(norm) or abs(norm-1) > 1e-8:
        raise FloatingPointError("state norm drift exceeds 1e-8; refusing to sample")
    probabilities /= norm
    rng = np.random.default_rng(seed)
    indices = rng.choice(model.dimension, size=shots, p=probabilities)
    sampled_costs = model.costs[indices]
    feasible = sampled_costs < model.instance.penalty
    best_result = None
    if np.any(feasible):
        # Selection is over MEASURED outcomes only, never over the full table.
        index = int(indices[int(np.argmin(sampled_costs))])
        orders = model.orders_at(index)
        schedule = evaluate_orders(model.instance, orders)
        if validate_schedule(model.instance, schedule["starts"]) != schedule["makespan"]:
            raise RuntimeError("independent schedule validation failed")
        best_result = {"basis_index": index, "orders_zero_based": orders, **schedule}
    optimum = int(model.costs.min())
    p = float(probabilities[model.costs == optimum].sum())
    miss_probability = 0.0 if p >= 1 else math.exp(shots*math.log1p(-p))
    return {"shots": shots, "seed": seed, "feasible_shots": int(feasible.sum()),
            "optimal_shots_diagnostic": int(np.sum(sampled_costs == optimum)),
            "at_least_one_optimal_probability_diagnostic": 1-miss_probability,
            "best_measured_schedule": best_result,
            "sampling_note": "CPU reuses one simulated probability distribution; actual hardware must reprepare for each shot."}


def solve(inst, tau=20.0, layers=2000, s_final=0.95, shots=1000, seed=7,
          max_states=200_000, max_memory_mib=512, progress=False):
    start = perf_counter()
    model = PermutationModel(inst, max_states, max_memory_mib)
    built = perf_counter()
    psi, evolution = evolve(model, tau, layers, s_final, progress)
    evolved = perf_counter()
    sampling = sample_schedule(model, psi, shots, seed)
    sampled = perf_counter()
    result = {
        "implementation": "classical full-state simulation of permutation-space quantum evolution",
        "instance": inst.as_dict(), "resources": estimate_resources(inst, layers),
        "cost_offset": inst.lower_bound, "evolution": evolution,
        "final_state_diagnostics": state_statistics(model, psi), "sampling": sampling,
        "timings_seconds": {"basis_and_cost_table": built-start, "unitary_evolution": evolved-built,
                            "sampling_and_decode": sampled-evolved, "total": sampled-start},
        "limitations": ["Cost table enumerates every basis state; no large-instance quantum speedup is claimed.",
                        "tau is dimensionless evolution time, not seconds of hardware time.",
                        "Enumerated optimum is a diagnostic and never enters the state preparation or evolution.",
                        "Finite tau and layers do not guarantee global-optimum sampling."]
    }
    return result, psi, model


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def export_schedule(inst, schedule, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["job", "operation", "machine", "start", "finish", "duration"])
        for v, start in enumerate(schedule["starts"]):
            j, k = divmod(v, inst.machines)
            out.writerow([j+1, k+1, int(inst.routes[j,k])+1, start,
                          start+inst.processing[v], inst.processing[v]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("simulate", "estimate", "spectrum", "decode"):
        p = sub.add_parser(name)
        p.add_argument("--instance", help="text: duration rows then 1-based routes; or JSON")
        p.add_argument("--jobs", type=int, default=3, help="generated demo size when no input file")
        p.add_argument("--machines", type=int, default=3)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--output", help="JSON report file (also prints JSON to stdout)")
        if name in ("simulate", "estimate"):
            p.add_argument("--layers", type=int, default=2000)
        if name in ("simulate", "spectrum"):
            p.add_argument("--max-states", type=int, default=200_000 if name == "simulate" else 256)
            p.add_argument("--max-memory-mib", type=float, default=512)
            p.add_argument("--s-final", type=float, default=0.95)
        if name == "simulate":
            p.add_argument("--tau", type=float, default=20)
            p.add_argument("--shots", type=int, default=1000)
            p.add_argument("--schedule-csv")
            p.add_argument("--state-out", help="optional .npy file of complex amplitudes")
            p.add_argument("--progress", action="store_true")
        elif name == "spectrum":
            p.add_argument("--points", type=int, default=21)
        elif name == "decode":
            p.add_argument("--orders", required=True, help="JSON with orders_zero_based operation IDs")
            p.add_argument("--schedule-csv")
    args = parser.parse_args(argv)
    try:
        if args.jobs < 1 or args.machines < 1:
            raise ValueError("jobs and machines must be positive")
        inst = Instance.read(args.instance) if args.instance else demo_instance(args.jobs, args.machines, args.seed)
        if args.command == "estimate":
            if args.layers < 1:
                raise ValueError("layers must be positive")
            result = estimate_resources(inst, args.layers)
        elif args.command == "decode":
            obj = json.loads(Path(args.orders).read_text(encoding="utf-8"))
            result = evaluate_orders(inst, obj["orders_zero_based"])
            if result["feasible"]:
                validate_schedule(inst, result["starts"])
                if args.schedule_csv:
                    export_schedule(inst, result, args.schedule_csv)
        elif args.command == "spectrum":
            if args.points < 2 or not 0 < args.s_final <= 1:
                raise ValueError("points >= 2 and 0 < s_final <= 1 required")
            check_capacity(inst, min(args.max_states, 256), args.max_memory_mib)
            model = PermutationModel(inst, min(args.max_states, 256), args.max_memory_mib)
            rows = []
            for s in np.linspace(0, args.s_final, args.points):
                eig = np.linalg.eigvalsh(model.dense_hamiltonian(float(s)))
                rows.append({"s": float(s), "ground_energy": float(eig[0]),
                             "gap": float(eig[1]-eig[0]) if len(eig) > 1 else None})
            finite = [row for row in rows if row["gap"] is not None]
            result = {"instance": inst.name, "grid": rows,
                      "smallest_sampled_gap": min((r["gap"] for r in finite), default=None),
                      "note": "Grid samples are NOT a certified lower bound on the continuous minimum gap."}
        else:
            result, psi, model = solve(inst, args.tau, args.layers, args.s_final, args.shots,
                                       args.seed, args.max_states, args.max_memory_mib, args.progress)
            schedule = result["sampling"]["best_measured_schedule"]
            if args.schedule_csv and schedule is not None:
                export_schedule(inst, schedule, args.schedule_csv)
            if args.state_out:
                path = Path(args.state_out)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, psi)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, FloatingPointError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
