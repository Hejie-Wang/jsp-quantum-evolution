#!/usr/bin/env python3
"""Machine-candidate quantum-search master loop for medium JSP instances.

Extends the 3x3 mathematical demo (code/candidate_quantum_demo) to instances
that cannot be enumerated. The candidate-combination master is posed in the
fixed-T feasibility form of docs/quantum_candidate_design.md section 4.1:
one-hot machine-candidate variables, one linear cut per witness
(sum of per-machine membership indicators <= |S|-1), T lowered only after an
independently verified feasible schedule achieves it.

Two master backends are provided:
  milp   exact classical master (scipy HiGHS) - gives pool lower-bound
         certificates when infeasible; reference backend, timed separately.
  kaiwu  QUBO simulated annealing (kaiwu SDK) - classical sampler;
         failure to reach zero violation proves nothing.

No full candidate enumeration, no dense Hamiltonian, no quantum speedup claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

class Instance:
    """Operation id = job * machines + position_in_job (zero based)."""

    def __init__(self, durations, routes, name="instance"):
        p, r = np.asarray(durations), np.asarray(routes)
        if p.ndim != 2 or p.shape != r.shape or not all(p.shape):
            raise ValueError("durations and routes must be nonempty equal 2D arrays")
        if p.dtype.kind not in "iu" or r.dtype.kind not in "iu":
            raise ValueError("durations and routes must contain integers")
        if np.any(p <= 0):
            raise ValueError("durations must be positive")
        self.jobs, self.machines = p.shape
        for row in r:
            if sorted(map(int, row)) != list(range(self.machines)):
                raise ValueError("each job must visit every machine exactly once")
        self.durations = np.array(p, dtype=np.int64)
        self.routes = np.array(r, dtype=np.int64)
        self.name = name
        self.operations = self.jobs * self.machines
        self.processing = tuple(map(int, self.durations.flat))
        self.machine_of = tuple(map(int, self.routes.flat))
        self.groups = tuple(tuple(v for v in range(self.operations)
                                  if self.machine_of[v] == m)
                            for m in range(self.machines))
        self.job_arcs = tuple((j * self.machines + k, j * self.machines + k + 1)
                              for j in range(self.jobs)
                              for k in range(self.machines - 1))
        self.job_arc_set = frozenset(self.job_arcs)
        self.total_duration = sum(self.processing)
        self.lower_bound = max(
            max(sum(map(int, row)) for row in self.durations),
            max(sum(self.processing[v] for v in g) for g in self.groups))

    @classmethod
    def read(cls, path):
        """Text format: J duration rows followed by J 1-based route rows."""
        path = Path(path)
        a = np.loadtxt(path, dtype=np.int64, ndmin=2)
        if len(a) % 2:
            raise ValueError("text input: J duration rows followed by J route rows")
        return cls(a[:len(a) // 2], a[len(a) // 2:] - 1, path.stem)


def validate_schedule(inst, starts):
    """Independent precedence and non-overlap check; returns makespan."""
    if (len(starts) != inst.operations
            or any(not isinstance(s, (int, np.integer)) or s < 0 for s in starts)):
        raise ValueError("invalid start times")
    for u, v in inst.job_arcs:
        if starts[u] + inst.processing[u] > starts[v]:
            raise ValueError("job precedence violated")
    for group in inst.groups:
        row = sorted(group, key=lambda v: starts[v])
        for u, v in zip(row, row[1:]):
            if starts[u] + inst.processing[u] > starts[v]:
                raise ValueError("machine overlap")
    return max(s + p for s, p in zip(starts, inst.processing))


def validate_pool(inst, pool):
    """Reject malformed pools before allocating a master problem."""
    if len(pool) != inst.machines:
        raise ValueError("expected one candidate pool per machine")
    for machine, candidates in enumerate(pool):
        if not candidates:
            raise ValueError("machine pools must be nonempty")
        seen = set()
        for order in candidates:
            if (any(not isinstance(v, (int, np.integer)) for v in order)
                    or sorted(order) != list(inst.groups[machine])):
                raise ValueError("candidate must permute its machine's operations")
            if tuple(order) in seen:
                raise ValueError("duplicate machine candidate")
            seen.add(tuple(order))


def content_hash(data):
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"),
                         default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Graph decoder: fixed machine orders -> earliest schedule, or one cycle
# ---------------------------------------------------------------------------

def decode(inst, orders, validate=True):
    if validate:
        if len(orders) != inst.machines:
            raise ValueError("expected one order per machine")
        for row, group in zip(orders, inst.groups):
            if sorted(map(int, row)) != list(group):
                raise ValueError("each order must permute exactly its machine's operations")
    adj = [[] for _ in range(inst.operations)]
    degree = [0] * inst.operations
    for u, v in inst.job_arcs:
        adj[u].append(v)
        degree[v] += 1
    for row in orders:
        for u, v in zip(row, row[1:]):
            adj[u].append(v)
            degree[v] += 1
    queue = [v for v in range(inst.operations) if degree[v] == 0]
    starts = [0] * inst.operations
    parent = [-1] * inst.operations
    for u in queue:
        finish = starts[u] + inst.processing[u]
        for v in adj[u]:
            if finish > starts[v]:
                starts[v] = finish
                parent[v] = u
            degree[v] -= 1
            if degree[v] == 0:
                queue.append(v)
    if len(queue) == inst.operations:
        last = max(range(inst.operations), key=lambda u: starts[u] + inst.processing[u])
        path = [last]
        while parent[path[-1]] != -1:
            path.append(parent[path[-1]])
        return {"feasible": True, "makespan": starts[last] + inst.processing[last],
                "starts": starts, "nodes": tuple(reversed(path))}
    # Iterative DFS; the remaining unprocessed subgraph must contain a cycle.
    color = [0] * inst.operations
    for root in range(inst.operations):
        if color[root]:
            continue
        stack, path = [(root, iter(adj[root]))], [root]
        color[root] = 1
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if color[v] == 1:
                    cycle = path[path.index(v):] + [v]
                    return {"feasible": False, "nodes": tuple(cycle)}
                if color[v] == 0:
                    color[v] = 1
                    stack.append((v, iter(adj[v])))
                    path.append(v)
                    advanced = True
                    break
            if not advanced:
                color[u] = 2
                stack.pop()
                path.pop()
    raise AssertionError("an unfinished topological sort implies a cycle")


# ---------------------------------------------------------------------------
# Witnesses: cycle or longest path -> per-machine candidate membership sets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Witness:
    kind: str                       # "cycle" or "path"
    relations: tuple                # sorted tuple of (machine, u, v); u before v
    length: int = 0                 # sum of node durations; paths only
    nodes: tuple = ()               # original graph evidence, not pool labels

    def key(self):
        return (self.kind, self.relations, self.length)


def separate(inst, decoded):
    """Compile one decode result into a witness with machine precedence pairs."""
    nodes = decoded["nodes"]
    requirements = []
    for u, v in zip(nodes, nodes[1:]):
        if (u, v) in inst.job_arc_set:
            continue
        mu, mv = inst.machine_of[u], inst.machine_of[v]
        if mu != mv:
            raise AssertionError("non-job edge in witness must stay on one machine")
        requirements.append((mu, u, v))
    length = sum(inst.processing[u] for u in dict.fromkeys(nodes))
    return Witness("path" if decoded["feasible"] else "cycle",
                   tuple(sorted(set(requirements))), length, tuple(nodes))


def witness_support(witness, pool):
    """Per-machine allowed-candidate tuples; support = non-trivial machines.

    Returns (support, allowed) or None when the witness can never activate
    inside this pool (some machine satisfies no required relation).
    """
    per_machine = {}
    for m, u, v in witness.relations:
        per_machine.setdefault(m, []).append((u, v))
    allowed = {}
    for m, reqs in per_machine.items():
        ok = []
        for a, order in enumerate(pool[m]):
            pos = {v: i for i, v in enumerate(order)}
            if all(pos[u] < pos[v] for u, v in reqs):
                ok.append(a)
        if not ok:
            return None
        allowed[m] = tuple(ok)
    support = tuple(m for m in allowed if len(allowed[m]) < len(pool[m]))
    return support, allowed


# ---------------------------------------------------------------------------
# Candidate pool: priority-rule serial SGS schedules -> per-machine orders
# ---------------------------------------------------------------------------

def serial_sgs(inst, rule, rng):
    """Append-only serial dispatch schedule; returns (starts, makespan).

    No insertion into machine idle gaps: this is not an active-SGS guarantee.
    """
    proc = inst.processing
    remaining = [sum(inst.durations[j, k:])
                 for j in range(inst.jobs) for k in range(inst.machines)]
    next_pos = [0] * inst.jobs
    job_ready = [0] * inst.jobs
    machine_free = [0] * inst.machines
    starts = [0] * inst.operations
    done = 0
    while done < inst.operations:
        ready = [(j * inst.machines + next_pos[j], j)
                 for j in range(inst.jobs) if next_pos[j] < inst.machines]
        if rule == "random":
            op, j = ready[rng.integers(len(ready))]
        else:
            keyed = []
            for op, j in ready:
                est = max(job_ready[j], machine_free[inst.machine_of[op]])
                if rule == "spt":
                    key = (proc[op], est)
                elif rule == "lpt":
                    key = (-proc[op], est)
                elif rule == "mwr":
                    key = (-remaining[op], est)
                elif rule == "est":
                    key = (est, proc[op])
                else:
                    raise ValueError(f"unknown rule {rule!r}")
                keyed.append((key, rng.random(), op, j))
            _, _, op, j = min(keyed)
        start = max(job_ready[j], machine_free[inst.machine_of[op]])
        starts[op] = start
        finish = start + proc[op]
        machine_free[inst.machine_of[op]] = finish
        job_ready[j] = finish
        next_pos[j] += 1
        done += 1
    return starts, max(s + p for s, p in zip(starts, proc))


def build_pool(inst, candidates_per_machine=8, n_schedules=240, seed=7):
    """Generate heuristic schedules; keep up to K distinct orders per machine.

    The overall best schedule's per-machine orders are always included, so the
    pool contains at least one complete feasible combination.
    """
    if candidates_per_machine < 1 or n_schedules < 1:
        raise ValueError("candidate and schedule counts must be positive")
    rng = np.random.default_rng(seed)
    rules = ("spt", "lpt", "mwr", "est", "random")
    schedules = []
    for i in range(n_schedules):
        starts, makespan = serial_sgs(inst, rules[i % len(rules)], rng)
        orders = [tuple(sorted(inst.groups[m], key=lambda v: starts[v]))
                  for m in range(inst.machines)]
        schedules.append({"makespan": makespan, "starts": starts, "orders": orders})
    schedules.sort(key=lambda s: s["makespan"])
    best = schedules[0]
    if validate_schedule(inst, best["starts"]) != best["makespan"]:
        raise RuntimeError("SGS schedule failed independent validation")
    pool = []
    for m in range(inst.machines):
        seen = {best["orders"][m]: best["makespan"]}
        for s in schedules:
            seen.setdefault(s["orders"][m], s["makespan"])
        ranked = sorted(seen, key=lambda o: (seen[o], o))
        keep = [best["orders"][m]]
        keep += [o for o in ranked if o != best["orders"][m]]
        pool.append(tuple(keep[:candidates_per_machine]))
    validate_pool(inst, pool)
    return pool, {"makespan": best["makespan"], "orders": [list(o) for o in best["orders"]],
                  "schedules_generated": n_schedules,
                  "distinct_per_machine": [len({tuple(sorted(inst.groups[m], key=lambda v: s['starts'][v]))
                                                for s in schedules})
                                           for m in range(inst.machines)]}


# ---------------------------------------------------------------------------
# QUBO master (kaiwu simulated annealing backend)
# ---------------------------------------------------------------------------

def build_qubo(pool, cuts, max_vars=2048):
    """Upper-triangular Q; x.T @ Q @ x + constant = squared residuals.

    Variables: x_{m,a} at m*K+a, then per-cut binary slack bits.
    One-hot: (sum_a x_{m,a} - 1)^2 per machine.
    Cut with support S and slack bits 2^b: (sum_m in S l_m + sum_b 2^b s_b
    - (|S|-1))^2, l_m = sum_{a in allowed} x_{m,a}.
    Zero energy iff every machine is one-hot and every cut holds.
    """
    sizes = [len(p) for p in pool]
    base = sum(sizes)
    slack_bits = [max(0, math.ceil(math.log2(len(c["support"])))) if len(c["support"]) > 1 else 0
                  for c in cuts]
    total = base + sum(slack_bits)
    if total > max_vars:
        raise ValueError(f"QUBO requires {total} variables, cap is {max_vars}")
    q = np.zeros((total, total))
    constant = float(len(pool))
    # c vector of one affine expression, expanded as E^2 with constant d.
    def add_square(coeffs, d):
        idx = sorted(coeffs)
        for i in idx:
            ci = coeffs[i]
            q[i, i] += ci * ci + 2 * d * ci
        for pos, i in enumerate(idx):
            for j in idx[pos + 1:]:
                q[i, j] += 2 * coeffs[i] * coeffs[j]
    for m, size in enumerate(sizes):
        add_square({_flat(sizes, m, a): 1.0 for a in range(size)}, -1.0)
    offset = 0
    for c, bits in zip(cuts, slack_bits):
        coeffs = {}
        for m in c["support"]:
            for a in c["allowed"][m]:
                i = _flat(sizes, m, a)
                coeffs[i] = coeffs.get(i, 0.0) + 1.0
        for b in range(bits):
            coeffs[base + offset + b] = float(1 << b)
        offset += bits
        add_square(coeffs, -(len(c["support"]) - 1.0))
        constant += (len(c["support"]) - 1.0) ** 2
    return q, {"base_vars": base, "slack_bits": slack_bits,
               "total_vars": total, "constant": constant}


def _flat(sizes, m, a):
    return sum(sizes[:m]) + a


def qubo_violation(pool, cuts, x):
    """(one-hot violations, violated cut count) of a binary configuration."""
    sizes = [len(p) for p in pool]
    oh = sum(abs(int(sum(x[_flat(sizes, m, a)] for a in range(size))) - 1)
             for m, size in enumerate(sizes))
    bad = 0
    for c in cuts:
        active = sum(1 for m in c["support"]
                     if any(x[_flat(sizes, m, a)] for a in c["allowed"][m]))
        if active == len(c["support"]):
            bad += 1
    return oh, bad


def solve_qubo_kaiwu(q, meta, seed=7, initial_temperature=100.0, alpha=0.98,
                     cutoff_temperature=0.01, iterations_per_t=100,
                     size_limit=50):
    """kaiwu classical SA on the Ising form; returns samples sorted by energy."""
    from kaiwu import SimulatedAnnealingOptimizer, qubo_matrix_to_ising_matrix
    ising, _ = qubo_matrix_to_ising_matrix(q)
    optimizer = SimulatedAnnealingOptimizer(
        initial_temperature=initial_temperature, alpha=alpha,
        cutoff_temperature=cutoff_temperature,
        iterations_per_t=iterations_per_t, size_limit=size_limit,
        rand_seed=seed)
    # SDK 1.0.7's second argument is negtail_flip, NOT an initial state.
    raw = optimizer.solve(ising, negtail_flip=True, sort_solutions=True)
    return decode_ising_samples(raw, q, meta)


def decode_ising_samples(raw, q, meta):
    """Remove the auxiliary field spin with gauge-invariant decoding."""
    n = meta["total_vars"]
    if raw is None:
        return []
    raw = np.asarray(raw)
    if raw.size == 0:
        return []
    raw = np.atleast_2d(raw)
    if raw.shape[1] != n + 1 or not np.all(np.isin(raw, (-1, 1))):
        raise ValueError("SDK must return n+1 Ising spins per sample")
    samples, seen = [], set()
    for row in raw:
        x = tuple(int(v > 0) for v in row[:n] * row[-1])
        if x in seen:
            continue
        seen.add(x)
        arr = np.asarray(x, dtype=float)
        samples.append((float(arr @ q @ arr + meta["constant"]), x))
    samples.sort(key=lambda t: t[0])
    return samples


# ---------------------------------------------------------------------------
# MILP master (exact classical reference backend)
# ---------------------------------------------------------------------------

def solve_master_milp(pool, cuts, rank_bias=None, time_limit=120.0):
    """Exact feasibility master. Returns (status, choice list or None).

    Only "infeasible" is a proof. A limit may carry a checked incumbent;
    "unknown" and "error" never imply infeasibility.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    if time_limit <= 0:
        return "unknown", None
    sizes = [len(p) for p in pool]
    n = sum(sizes)
    rows, lb, ub = [], [], []
    for m, size in enumerate(sizes):
        row = lil_matrix((1, n))
        for a in range(size):
            row[0, _flat(sizes, m, a)] = 1.0
        rows.append(row)
        lb.append(1.0)
        ub.append(1.0)
    for c in cuts:
        row = lil_matrix((1, n))
        for m in c["support"]:
            for a in c["allowed"][m]:
                row[0, _flat(sizes, m, a)] = 1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(len(c["support"]) - 1.0)
    from scipy.sparse import vstack
    a_mat = vstack(rows).tocsr()
    c_obj = np.zeros(n)
    if rank_bias is not None:
        for m, size in enumerate(sizes):
            for a in range(size):
                c_obj[_flat(sizes, m, a)] = 1e-3 * a
    res = milp(c_obj, constraints=LinearConstraint(a_mat, lb, ub),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if res.status == 2:
        return "infeasible", None
    if res.status not in (0, 1):
        return "error", None
    if res.x is None:
        return "unknown", None
    values = np.asarray(res.x)
    if (values.shape != (n,) or not np.all(np.isfinite(values))
            or np.any(np.abs(values - np.rint(values)) > 1e-6)
            or np.any(values < -1e-6) or np.any(values > 1 + 1e-6)):
        return "unknown", None
    x = np.rint(values).astype(int)
    if qubo_violation(pool, cuts, x) != (0, 0):
        return "unknown", None
    choice = one_hot_choice(pool, x)
    return ("optimal" if res.status == 0 else "limit_with_incumbent"), choice


def one_hot_choice(pool, x):
    """Decode a legal one-hot sample; never repair an illegal reading."""
    sizes = [len(p) for p in pool]
    values = np.asarray(x)
    if len(values) < sum(sizes) or not np.all(np.isin(values, (0, 1))):
        raise ValueError("expected binary candidate bits")
    if qubo_violation(pool, [], values)[0]:
        raise ValueError("illegal one-hot encoding")
    return [next(a for a in range(size) if values[_flat(sizes, m, a)])
            for m, size in enumerate(sizes)]


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------

def compile_cuts(pool, witnesses, target_t=None):
    """Project witnesses to the pool; drop never-active ones; report drops."""
    cuts, dropped = [], 0
    for w in witnesses:
        if w.kind == "path" and target_t is not None and w.length <= target_t:
            continue
        projected = witness_support(w, pool)
        if projected is None:
            dropped += 1
            continue
        support, allowed = projected
        cuts.append({"support": support, "allowed": allowed, "kind": w.kind,
                     "length": w.length})
    return cuts, dropped


def choice_orders(pool, choice):
    if (len(choice) != len(pool)
            or any(not isinstance(a, (int, np.integer)) or not 0 <= a < len(pool[m])
                   for m, a in enumerate(choice))):
        raise ValueError("invalid machine candidate choice")
    return [list(pool[m][a]) for m, a in enumerate(choice)]


def run_candidate_loop(inst, pool, master="milp", max_iterations=60,
                       time_limit=600.0, seed=7, samples_per_iter=4,
                       sa_options=None, log=print, initial_choice=None,
                       max_qubo_vars=2048):
    """Fixed-T feasibility cut loop over the candidate pool.

    T starts at the best pool schedule's makespan - 1 and is lowered only by
    independently verified feasible decodes. MILP infeasibility at T certifies
    pool optimum >= T+1. SA failure proves nothing; the loop then just stops.
    """
    if master not in ("milp", "kaiwu"):
        raise ValueError(f"unknown master {master!r}")
    if (max_iterations < 0 or samples_per_iter < 1 or max_qubo_vars < 1
            or not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("invalid iteration, sample, variable or time budget")
    started = perf_counter()
    validate_pool(inst, pool)
    lb = inst.lower_bound
    witnesses, seen_keys, trace = [], set(), []
    sa_options = dict(sa_options or {})
    initial_choice = [0] * len(pool) if initial_choice is None else list(initial_choice)
    initial = decode(inst, choice_orders(pool, initial_choice))
    best = _verified_best(inst, initial_choice, initial) if initial["feasible"] else None
    t_target = best["makespan"] - 1 if best else None
    initial_makespan = best["makespan"] if best else None
    first_witness = separate(inst, initial)
    witnesses.append(first_witness)
    seen_keys.add(first_witness.key())
    proof = "globally_optimal_at_lower_bound" if best and best["makespan"] == lb else None
    stop_reason = "proved" if proof else "iteration_budget"
    stall, evaluation_count = 0, 1
    initial_evaluation_seconds = perf_counter() - started

    for iteration in range(max_iterations):
        if proof:
            break
        remaining = time_limit - (perf_counter() - started)
        if remaining <= 0:
            stop_reason = "time_limit"
            break
        tic = perf_counter()
        cuts, dropped = compile_cuts(pool, witnesses, t_target)
        step = {"iteration": iteration, "witnesses": len(witnesses),
                "cuts": len(cuts), "dropped_never_active": dropped,
                "target_T": t_target, "compile_seconds": perf_counter() - tic}
        tic = perf_counter()
        if master == "milp":
            status, choice = solve_master_milp(
                pool, cuts, rank_bias=True,
                time_limit=max(0, time_limit - (perf_counter() - started)))
            choices = [choice] if choice is not None else []
            step["master_status"] = status
            if status == "infeasible":
                proof = (f"pool_lower_bound:{t_target + 1}"
                         if t_target is not None else "pool_infeasible")
                stop_reason = "proved"
            elif not choices:
                stop_reason = "master_error" if status == "error" else "master_unknown"
        else:
            try:
                q, meta = build_qubo(pool, cuts, max_vars=max_qubo_vars)
            except ValueError as exc:
                step["error"] = str(exc)
                trace.append(step)
                stop_reason = "resource_limit"
                break
            try:
                samples = solve_qubo_kaiwu(q, meta, seed=seed + iteration, **sa_options)
            except (RuntimeError, ValueError, ImportError) as exc:
                step.update({"error": str(exc), "master_seconds": perf_counter() - tic})
                trace.append(step)
                stop_reason = "annealing_error"
                break
            choices, sample_stats = _legal_sample_choices(pool, cuts, samples,
                                                          samples_per_iter)
            step.update(sample_stats)
            step.update({"qubo_vars": meta["total_vars"], "qubo_nnz": int(np.count_nonzero(q)),
                         "qubo_bytes": q.nbytes, "qubo_constant": meta["constant"],
                         "best_qubo_energy": samples[0][0] if samples else None,
                         "qubo_zero_found": any(energy == 0 for energy, _ in samples)})
        step["master_seconds"] = perf_counter() - tic
        if proof or (master == "milp" and not choices):
            trace.append(step)
            break

        tic = perf_counter()
        improved, added, feasible = False, 0, 0
        for choice in choices:
            result = decode(inst, choice_orders(pool, choice))
            evaluation_count += 1
            if result["feasible"]:
                feasible += 1
                if best is None or result["makespan"] < best["makespan"]:
                    best = _verified_best(inst, choice, result)
                    t_target = best["makespan"] - 1
                    improved = True
                    if best["makespan"] == lb:
                        proof, stop_reason = "globally_optimal_at_lower_bound", "proved"
            w = separate(inst, result)
            if w.key() not in seen_keys:
                seen_keys.add(w.key())
                witnesses.append(w)
                added += 1
        stall = 0 if (improved or added) else stall + 1
        step.update({"improved": improved, "witnesses_added": added,
                     "best_makespan": best["makespan"] if best else None,
                     "decoded": len(choices), "graph_feasible": feasible,
                     "evaluation_seconds": perf_counter() - tic,
                     "budget_overrun_seconds": max(0, perf_counter() - started - time_limit)})
        trace.append(step)
        if log:
            log(f"iter {iteration}: master={master} T={t_target} "
                f"cuts={len(cuts)} added={added} improved={improved} "
                f"best={best['makespan'] if best else None} "
                f"({step['master_seconds']:.2f}s)")
        if perf_counter() - started >= time_limit:
            stop_reason = "proved" if proof else "time_limit"
            break
        if not proof and stall >= max(5, max_iterations // 4):
            stop_reason = "stall"
            break
    elapsed = perf_counter() - started
    gap = (best["makespan"] - lb) / lb if best else None
    return {
        "schema_version": 2,
        "instance": inst.name, "jobs": inst.jobs, "machines": inst.machines,
        "master": master, "simple_lower_bound": lb,
        "best_makespan": best["makespan"] if best else None,
        "best_choice": best["choice"] if best else None,
        "best_starts": best["starts"] if best else None,
        "best_orders": choice_orders(pool, best["choice"]) if best else None,
        "initial_makespan": initial_makespan,
        "gap_vs_simple_lb": gap, "certificate": proof or stop_reason,
        "proof_certificate": proof, "stop_reason": stop_reason,
        "final_target_T": t_target, "witnesses": len(witnesses),
        "iterations": len(trace), "wall_seconds": elapsed,
        "evaluation_count": evaluation_count,
        "master_seconds": sum(s.get("master_seconds", 0) for s in trace),
        "evaluation_seconds": initial_evaluation_seconds + sum(
            s.get("evaluation_seconds", 0) for s in trace),
        "budget_overrun_seconds": max(0, elapsed - time_limit),
        "settings": {"seed": seed, "max_iterations": max_iterations,
                     "time_limit": time_limit, "samples_per_iter": samples_per_iter,
                     "sa_options": sa_options if master == "kaiwu" else None,
                     "max_qubo_vars": max_qubo_vars},
        "instance_data": {"durations": inst.durations.tolist(),
                          "routes_zero_based": inst.routes.tolist()},
        "pool_data": pool, "pool_sha256": content_hash(pool),
        "witness_data": [{"kind": w.kind, "relations": w.relations,
                          "length": w.length, "nodes": w.nodes} for w in witnesses],
        "trace": trace,
        "notes": [
            "Fixed-T feasibility master; T lowered only by verified feasible decodes.",
            "MILP infeasibility certifies a POOL lower bound, not a JSP global bound.",
            "SA failure to find zero violation is not an infeasibility proof.",
            "Simple LB is max(job sum, machine sum); gap is an upper bound on suboptimality.",
            "Kaiwu SDK SA is classical; no QPU, quantum evolution or quantum advantage.",
            "Illegal one-hot readings are rejected, never repaired.",
            "Kaiwu SDK 1.0.7 has no initial-state or per-call time-limit argument.",
        ],
    }


def _verified_best(inst, choice, decoded):
    if validate_schedule(inst, decoded["starts"]) != decoded["makespan"]:
        raise RuntimeError("independent schedule validation failed")
    return {"makespan": decoded["makespan"], "choice": list(choice),
            "starts": decoded["starts"]}


def _legal_sample_choices(pool, cuts, samples, limit):
    choices, seen = [], set()
    illegal, cut_valid = 0, 0
    for _, sample in samples:
        one_hot_bad, cut_bad = qubo_violation(pool, cuts, sample)
        if one_hot_bad:
            illegal += 1
            continue
        cut_valid += int(cut_bad == 0)
        choice = one_hot_choice(pool, sample)
        key = tuple(choice)
        if key not in seen and len(choices) < limit:
            choices.append(choice)
            seen.add(key)
    return choices, {"unique_binary_samples": len(samples),
                     "illegal_one_hot_samples": illegal,
                     "illegal_one_hot_fraction": illegal / len(samples) if samples else None,
                     "master_valid_samples": cut_valid,
                     "unique_decoded_choices": len(choices)}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               default=_json_default, allow_nan=False) + "\n", encoding="utf-8")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)
