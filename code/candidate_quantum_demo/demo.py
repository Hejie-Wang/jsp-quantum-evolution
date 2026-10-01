#!/usr/bin/env python3
"""Tiny, exhaustive mathematical reference. No CUDA, QPU, or speedup claim.

The simulated basis contains only legal candidate labels (one-hot subspace),
plus a binary makespan register. Never use this dense solver at production size.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import sys

import scipy

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply


# Operation id = job * 3 + position within job. All indices are zero-based.
DURATIONS = (3, 2, 2, 2, 1, 4, 4, 3, 1)
MACHINES = (0, 1, 2, 1, 2, 0, 2, 0, 1)
JOB_ARCS = ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8))
# Deliberately chosen validation fixture: contains cycles and an optimal schedule.
# This is not a candidate-generation algorithm or evidence of discovering it.
POOLS = (((0, 5, 7), (7, 5, 0)),
         ((1, 3, 8), (3, 1, 8)),
         ((2, 4, 6), (4, 6, 2)))
CHOICES = tuple(product(*(range(len(pool)) for pool in POOLS)))
U = sum(DURATIONS)
PENALTY = U + 1
TIME_BITS = U.bit_length()
TIME_SIZE = 1 << TIME_BITS


def decode(choice):
    """Classical sparse DAG decoder; returns one longest path or one cycle."""
    rows = [POOLS[m][a] for m, a in enumerate(choice)]
    edges = set(JOB_ARCS)
    edges.update((u, v) for row in rows for u, v in zip(row, row[1:]))
    adj = [[] for _ in DURATIONS]
    degree = [0] * len(DURATIONS)
    for u, v in sorted(edges):
        adj[u].append(v)
        degree[v] += 1
    queue = [u for u, d in enumerate(degree) if d == 0]
    starts, parent = [0] * len(DURATIONS), [-1] * len(DURATIONS)
    for u in queue:
        for v in adj[u]:
            if starts[u] + DURATIONS[u] > starts[v]:
                starts[v], parent[v] = starts[u] + DURATIONS[u], u
            degree[v] -= 1
            if degree[v] == 0:
                queue.append(v)
    if len(queue) == len(DURATIONS):
        last = max(range(len(DURATIONS)), key=lambda u: starts[u] + DURATIONS[u])
        path = [last]
        while parent[path[-1]] != -1:
            path.append(parent[path[-1]])
        return {"feasible": True, "makespan": starts[last] + DURATIONS[last],
                "starts": starts, "nodes": tuple(reversed(path))}
    # Recursive DFS is deliberately limited to this nine-operation demo.
    color, stack = [0] * len(DURATIONS), []
    def visit(u):
        color[u] = 1
        stack.append(u)
        for v in adj[u]:
            if color[v] == 1:
                return tuple(stack[stack.index(v):] + [v])
            if color[v] == 0:
                found = visit(v)
                if found is not None:
                    return found
        stack.pop()
        color[u] = 2
        return None
    for u in range(len(DURATIONS)):
        if color[u] == 0:
            cycle = visit(u)
            if cycle is not None:
                return {"feasible": False, "nodes": cycle}
    raise AssertionError("a cycle must exist")


def verify_starts(starts):
    """Independent direct precedence/non-overlap check, without machine orders."""
    assert len(starts) == len(DURATIONS) and all(s >= 0 for s in starts)
    for u, v in JOB_ARCS:
        assert starts[u] + DURATIONS[u] <= starts[v]
    for u in range(len(DURATIONS)):
        for v in range(u + 1, len(DURATIONS)):
            if MACHINES[u] == MACHINES[v]:
                assert (starts[u] + DURATIONS[u] <= starts[v]
                        or starts[v] + DURATIONS[v] <= starts[u])
    return max(s + p for s, p in zip(starts, DURATIONS))


@dataclass(frozen=True)
class Witness:
    kind: str
    nodes: tuple[int, ...]
    allowed: tuple[tuple[int, ...], ...]
    length: int = 0

    def active(self, choice):
        return all(a in allowed for a, allowed in zip(choice, self.allowed))


def separate(result):
    """Compile one graph witness into per-machine candidate membership sets."""
    requirements = [[] for _ in POOLS]
    for u, v in zip(result["nodes"], result["nodes"][1:]):
        if (u, v) in JOB_ARCS:
            continue
        assert MACHINES[u] == MACHINES[v]
        requirements[MACHINES[u]].append((u, v))
    allowed = []
    for pool, req in zip(POOLS, requirements):
        allowed.append(tuple(a for a, row in enumerate(pool)
                             if all(row.index(u) < row.index(v) for u, v in req)))
    return Witness("path" if result["feasible"] else "cycle", result["nodes"],
                   tuple(allowed), sum(DURATIONS[u] for u in result["nodes"])
                   if result["feasible"] else 0)


def energy_table(witnesses):
    """H = T + (U+1) * (active cycles + active paths longer than T).

    Exhaustive table is a tiny-demo diagnostic, NOT the production phase oracle.
    Layout: candidate label index * TIME_SIZE + T (T is the fastest axis).
    """
    times = np.arange(TIME_SIZE)
    energy = np.tile(times, (len(CHOICES), 1))
    for i, choice in enumerate(CHOICES):
        for w in witnesses:
            if w.active(choice):
                energy[i] += PENALTY * (1 if w.kind == "cycle" else times < w.length)
    return energy


def exact_cut_loop():
    """Exact tiny master + classical separator; exactness is explicit here."""
    witnesses, trace = [], []
    for _ in range(2 * len(CHOICES) + 1):
        table = energy_table(witnesses)
        i, target = np.unravel_index(int(table.argmin()), table.shape)
        result = decode(CHOICES[i])
        trace.append({"cuts": len(witnesses), "pool_master_lower_bound": int(table[i, target]),
                      "choice": list(CHOICES[i]), "target": int(target),
                      "feasible": result["feasible"], "makespan": result.get("makespan")})
        if result["feasible"] and result["makespan"] <= target:
            verify_starts(result["starts"])
            return witnesses, trace
        witness = separate(result)
        assert witness not in witnesses, "the exact master must obey existing cuts"
        witnesses.append(witness)
    raise AssertionError("finite tiny cut loop did not terminate")


def covering_witnesses():
    """Exhaustive validation only: one cycle/longest path for every combination.

    Not all graph paths are listed, but these valid witnesses suffice on this
    finite pool: every cyclic choice has a cycle; every feasible choice has its
    own longest path. Do not confuse this with scalable constraint generation.
    """
    return tuple(dict.fromkeys(separate(decode(c)) for c in CHOICES))


def joint_specs(witnesses):
    """Deterministic experimental transitions from witness masks, not optimal labels.

    Change 2..3 constrained machines jointly from their first allowed candidate
    to their first disallowed candidate. This breaks that witness, but may
    create another violation. No directional descent guarantee is asserted.
    """
    specs = set()
    for w in witnesses:
        support = tuple(m for m, allowed in enumerate(w.allowed)
                        if 0 < len(allowed) < len(POOLS[m]))
        support = support[:3]
        if len(support) < 2:
            continue
        left = tuple(w.allowed[m][0] for m in support)
        right = tuple(next(a for a in range(len(POOLS[m])) if a not in w.allowed[m])
                      for m in support)
        specs.add((support, left, right))
    return tuple(sorted(specs))


def drivers(specs):
    """Exact legal-subspace matrices, not a compiled one-hot hardware circuit."""
    n = len(CHOICES)
    lookup = {c: i for i, c in enumerate(CHOICES)}
    local, joint = np.zeros((n, n)), np.zeros((n, n))
    for i, choice in enumerate(CHOICES):
        for m, pool in enumerate(POOLS):
            for a in range(len(pool)):
                if a != choice[m]:
                    other = choice[:m] + (a,) + choice[m + 1:]
                    local[i, lookup[other]] -= 1 / (len(pool) - 1)
        for support, left, right in specs:
            if tuple(choice[m] for m in support) == left:
                other = list(choice)
                for m, a in zip(support, right):
                    other[m] = a
                j = lookup[tuple(other)]
                joint[i, j] -= 1
                joint[j, i] -= 1
    clock = np.zeros((TIME_SIZE, TIME_SIZE))
    for t in range(TIME_SIZE):
        for bit in range(TIME_BITS):
            clock[t, t ^ (1 << bit)] = -1
    h0 = (np.kron(local, np.eye(TIME_SIZE)) + np.kron(np.eye(n), clock))
    h0 /= len(POOLS) + TIME_BITS  # exact norm 1 for this regular driver
    joint /= max(1.0, float(np.max(np.sum(np.abs(joint), axis=1))))
    return h0, np.kron(joint, np.eye(TIME_SIZE))


def evolve(table, h0, joint, strength, tau, steps):
    raw = table.ravel().astype(float)
    scale = max(1.0, float(np.max(np.abs(raw))))
    cost = np.diag(raw / scale)
    # All variants have ||H(s)|| <= 1; catalyst has zero endpoint weight.
    def hamiltonian(s):
        weight = strength * 4 * s * (1 - s)
        return ((1 - s) * h0 + s * cost + weight * joint) / (1 + weight)
    psi = np.ones(len(raw), dtype=complex) / np.sqrt(len(raw))
    dt = tau / steps
    for k in range(steps):
        h = csr_matrix(hamiltonian((k + 0.5) / steps))
        psi = expm_multiply((-1j * dt) * h, psi)
    probabilities = np.abs(psi) ** 2
    spectra = []
    for s in np.linspace(0, 1, 11):
        eig = eigh(hamiltonian(float(s)), subset_by_index=[0, 1], eigvals_only=True)
        spectra.append({"s": float(s), "e1_minus_e0": float(eig[1] - eig[0])})
    optimal = raw == raw.min()
    return {"joint_strength": strength, "tau": tau, "midpoint_steps": steps,
            "ground_probability": float(probabilities[optimal].sum()),
            "norm_error": abs(float(probabilities.sum()) - 1),
            "expected_problem_energy": float(probabilities @ raw),
            "sampled_spectrum": spectra,
            "note": "Classical ideal simulation; grid gaps are not continuous gap certificates."}


def run_demo(tau=20.0, steps=80):
    if not np.isfinite(tau) or tau <= 0 or steps < 1:
        raise ValueError("tau and steps must be positive and finite")
    decoded = [decode(c) for c in CHOICES]
    optimum = min(r["makespan"] for r in decoded if r["feasible"])
    witnesses = covering_witnesses()
    table = energy_table(witnesses)
    for i, result in enumerate(decoded):
        if result["feasible"]:
            assert verify_starts(result["starts"]) == int(table[i].min())
            assert table[i].min() == result["makespan"]
        else:
            assert table[i].min() > U
    assert int(table.min()) == optimum
    _, trace = exact_cut_loop()
    assert trace[-1]["pool_master_lower_bound"] == optimum
    specs = joint_specs(witnesses)
    h0, joint = drivers(specs)
    return {
        "scope": "3x3 mathematical demo; exhaustive classical reference; no QPU/CUDA",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "operations": len(DURATIONS), "candidate_pool_sizes": [len(p) for p in POOLS],
        "candidate_combinations": len(CHOICES), "one_hot_data_qubits": sum(map(len, POOLS)),
        "time_qubits": TIME_BITS, "simulated_legal_subspace_dimension": table.size,
        "feasible_combinations": sum(r["feasible"] for r in decoded),
        "pool_optimum": optimum, "ground_energy": int(table.min()), "penalty": PENALTY,
        "witness_count": len(witnesses), "joint_transition_count": len(specs),
        "ground_encoding_verified": True, "cut_loop": trace,
        "quantum_simulations": [evolve(table, h0, joint, s, tau, steps) for s in (0.0, 0.5)],
        "limitations": ["Fixed hand-selected pool; no general candidate coverage proof.",
                        "Joint transitions are a proposal, not a speedup or gap guarantee.",
                        "Exhaustive tables and dense matrices are validation-only.",
                        "Actual gate compilation, noise and hardware runtime are not implemented."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_demo(args.tau, args.steps)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
