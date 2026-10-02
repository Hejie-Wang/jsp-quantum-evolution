#!/usr/bin/env python3
"""Dynamic machine candidates + critical-block tabu + optional quantum proposals.

Classical local search supplies new machine orders. Periodically a small pool
of entire machine candidates is recombined using uniform samples or the exact
classical simulation of the existing witness-phase/XY/joint quantum ansatz.
The full JSP graph (including frozen machines) evaluates every proposal.
This executable does not submit QPU jobs or claim quantum advantage.
"""
from pathlib import Path
import argparse
import json
import sys
from time import perf_counter
import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "candidate_quantum_medium"))
import medium_qjsp as mq
from batch_evaluator import BatchEvaluator, INVALID, evaluate_one
from circuits import CandidatePool
from compact_simulator import CompactSimulator
from search_loop import graph_witness_specs, propose_joint_actions


@njit(cache=True)
def gt_initial(processing, routes, jobs, machines, seed):
    """Randomized Giffler-Thompson conflict-set construction, no known solution."""
    np.random.seed(seed)
    nxt = np.zeros(jobs, np.int32)
    jr, mr = np.zeros(jobs, np.int32), np.zeros(machines, np.int32)
    remaining = processing.reshape((jobs, machines)).sum(axis=1)
    orders, counts = np.empty((machines, jobs), np.int32), np.zeros(machines, np.int32)
    for _ in range(jobs * machines):
        earliest, machine = INVALID, -1
        for j in range(jobs):
            if nxt[j] < machines:
                u = j * machines + nxt[j]
                end = max(jr[j], mr[routes[u]]) + processing[u]
                if end < earliest:
                    earliest, machine = end, routes[u]
        selected, score = -1, -1.0
        for j in range(jobs):
            if nxt[j] < machines:
                u = j * machines + nxt[j]
                if routes[u] == machine and max(jr[j], mr[machine]) < earliest:
                    value = remaining[j] * (0.75 + np.random.random())
                    if value > score:
                        selected, score = j, value
        u = selected * machines + nxt[selected]
        nxt[selected] += 1
        jr[selected] = max(jr[selected], mr[machine]) + processing[u]
        mr[machine] = jr[selected]
        remaining[selected] -= processing[u]
        orders[machine, counts[machine]] = u
        counts[machine] += 1
    return orders


def critical_moves(orders, processing, head, tail, makespan, insertion=True):
    """Boundary insertions of all critical machine blocks, including nonadjacent."""
    moves = set()
    for m, row in enumerate(orders):
        a = 0
        while a < len(row) - 1:
            def tight(k):
                u, v = row[k], row[k + 1]
                return head[u] + processing[u] + processing[v] + tail[v] == makespan
            if not tight(a):
                a += 1
                continue
            b = a + 1
            while b < len(row) - 1 and tight(b):
                b += 1
            if insertion:
                for k in range(a + 1, b + 1):
                    moves.add((m, a, k)); moves.add((m, k, a))
                for k in range(a, b):
                    moves.add((m, b, k)); moves.add((m, k, b))
            else:
                moves.add((m, a, a + 1)); moves.add((m, b - 1, b))
            a = b
    return sorted(moves)


def apply_moves(current, moves):
    batch = np.repeat(current[None], len(moves), axis=0)
    for order, (m, a, b) in zip(batch, moves):
        row = order[m].tolist()
        row.insert(b, row.pop(a))
        order[m] = row
    return batch


def refreshed_pool(current, moves, batch, values, rng, active_machines=6, candidates=3):
    """Retain current order; select diverse critical machines and whole-row alternatives."""
    alternatives = {}
    for i in np.argsort(values, kind="stable"):
        if values[i] == INVALID:
            continue
        m = moves[i][0]
        alternatives.setdefault(m, [])
        row = tuple(map(int, batch[i, m]))
        if row not in alternatives[m]:
            alternatives[m].append(row)
    machines = list(alternatives)
    rng.shuffle(machines)
    selected = set(machines[:active_machines])
    pool = []
    for m, row in enumerate(current):
        base = tuple(map(int, row))
        more = alternatives.get(m, [])[:candidates - 1] if m in selected else []
        pool.append(tuple([base] + [r for r in more if r != base]))
    return CandidatePool(tuple(pool))


def solve(inst, *, seconds=10.0, seed=7, mode="classical", backend="cpu",
          max_iterations=100000, proposal_every=40, shots=64, train_budget=12,
          initial_orders=None, initial_schedules=240, neighborhood="mixed"):
    if mode not in {"classical", "uniform", "quantum"}:
        raise ValueError("mode must be classical, uniform or quantum")
    if neighborhood not in {"adjacent", "insertion", "mixed"}:
        raise ValueError("unknown neighborhood")
    if seconds <= 0 or max_iterations < 1 or proposal_every < 1 or shots < 1 or train_budget < 1:
        raise ValueError("budgets must be positive")
    started = perf_counter()
    rng, proposal_rng = np.random.default_rng(seed), np.random.default_rng(seed + 100003)
    evaluator = BatchEvaluator(inst, backend)
    processing = evaluator.processing
    routes = np.array(inst.machine_of, dtype=np.int32)
    if initial_orders is None:
        _, info = mq.build_pool(inst, 8, initial_schedules, seed)
        initial_orders = info["orders"]
    current = np.asarray(initial_orders, dtype=np.int32).copy()
    decoded = mq.decode(inst, current)
    if not decoded["feasible"]:
        raise ValueError("initial schedule is infeasible")
    best_value = current_value = int(decoded["makespan"])
    best = current.copy()
    initial_value = best_value
    # Warm JIT explicitly; setup is reported separately from bounded search.
    evaluator.evaluate(current[None])
    gt_initial(processing, routes, inst.jobs, inst.machines, seed)
    setup_seconds = perf_counter() - started
    search_start, trace = perf_counter(), []
    deadline = search_start + seconds
    tabu, witnesses, witness_keys = {}, [], set()
    restarts = improvements = 0
    last_improvement = 0
    proposal_calls = proposal_evaluations = proposal_improvements = training_evaluations = 0
    proposal_feasible = 0
    proposal_seconds = 0.0
    maximum_subspace, maximum_data_qubits = 1, 0

    def remember(result):
        w = mq.separate(inst, result)
        if w.key() not in witness_keys:
            witnesses.append(w); witness_keys.add(w.key())
            # bounded active set; full graph verification is never truncated
            if len(witnesses) > 96:
                witness_keys.remove(witnesses.pop(0).key())

    remember(decoded)
    iteration = 0
    while iteration < max_iterations and perf_counter() < deadline:
        iteration += 1
        current_value, head, tail = evaluate_one(current, processing, inst.machines)
        insertion = neighborhood == "insertion" or (neighborhood == "mixed" and iteration % 10 == 0)
        moves = critical_moves(current, processing, head, tail, current_value, insertion=insertion)
        rng.shuffle(moves)
        batch = apply_moves(current, moves)
        values = evaluator.evaluate(batch, validate=False)
        chosen, score = -1, INVALID
        # Tabu stores forbidden restored pair orientations; insertion tests all crossed pairs.
        for i, (m, a, b) in enumerate(moves):
            value = int(values[i])
            if value >= INVALID:
                continue
            u = int(current[m, a])
            crossed = current[m, a + 1:b + 1] if a < b else current[m, b:a]
            created = [(int(v), u) if a < b else (u, int(v)) for v in crossed]
            forbidden = any(tabu.get(pair, 0) > iteration for pair in created)
            if forbidden and value >= best_value:
                continue
            if value < score:
                score, chosen = value, i

        if mode != "classical" and iteration % proposal_every == 0 and perf_counter() < deadline:
            tic = perf_counter()
            remember(mq.decode(inst, current, validate=False))
            pool = refreshed_pool(current, moves, batch, values, proposal_rng)
            specs = graph_witness_specs(pool, witnesses)
            specs = tuple(s for s in specs if all(s.allowed)
                          and (s.kind == "cycle" or s.length > best_value - 1))
            actions = propose_joint_actions(pool, specs, [0] * inst.machines, max_actions=6)
            simulator = CompactSimulator(pool, specs, actions, best_value - 1)
            maximum_subspace = max(maximum_subspace, simulator.dimension)
            maximum_data_qubits = max(maximum_data_qubits, pool.data_qubits)
            parameters = []
            if mode == "quantum":
                trained = simulator.train(budget=train_budget,
                    seed=seed + iteration, deadline=deadline)
                parameters = trained["params"]
                training_evaluations += trained["evaluations"]
            choices = simulator.sample(parameters, shots, proposal_rng,
                                        "xy_joint" if mode == "quantum" else "uniform")
            # Sampling duplicates and the incumbent need no graph reevaluation.
            choices = choices[np.any(choices != 0, axis=1)]
            combined = np.array([[pool.candidates[m][a] for m, a in enumerate(c)]
                                 for c in choices], dtype=np.int32).reshape((-1, inst.machines, inst.jobs))
            scores = evaluator.evaluate(combined, validate=False)
            proposal_calls += 1
            proposal_evaluations += len(scores)
            proposal_feasible += int(np.sum(scores < INVALID))
            # Add evidence from a bounded selection, including infeasible proposals.
            for i in proposal_rng.permutation(len(scores))[:8]:
                remember(mq.decode(inst, combined[i], validate=False))
            if len(scores):
                q = int(np.argmin(scores))
                if scores[q] < min(current_value, score):
                    current = combined[q].copy()
                    current_value = int(scores[q])
                    chosen = -2
                    if current_value < best_value:
                        proposal_improvements += 1
            proposal_seconds += perf_counter() - tic

        if chosen >= 0:
            m, a, b = moves[chosen]
            u = int(current[m, a])
            crossed = current[m, a + 1:b + 1] if a < b else current[m, b:a]
            tenure = int(rng.integers(7, 19))
            for v in crossed:
                old_pair = (u, int(v)) if a < b else (int(v), u)
                tabu[old_pair] = iteration + tenure
            current, current_value = batch[chosen].copy(), score
        elif chosen != -2:
            current = gt_initial(processing, routes, inst.jobs, inst.machines,
                                 int(rng.integers(1, 2**30)))
            current_value = evaluate_one(current, processing, inst.machines)[0]
            tabu.clear(); restarts += 1
            last_improvement = iteration

        if current_value < best_value:
            checked = mq.decode(inst, current)
            if not checked["feasible"] or mq.validate_schedule(inst, checked["starts"]) != current_value:
                raise RuntimeError("independent schedule verification failed")
            best_value, best = current_value, current.copy()
            last_improvement = iteration
            improvements += 1
            remember(checked)
            trace.append({"iteration": iteration, "search_seconds": perf_counter() - search_start,
                          "makespan": int(best_value), "source": "proposal" if chosen == -2 else "tabu"})
        if iteration - last_improvement >= 600:
            current = gt_initial(processing, routes, inst.jobs, inst.machines,
                                 int(rng.integers(1, 2**30)))
            tabu.clear(); restarts += 1
            last_improvement = iteration
        if iteration % 100 == 0:
            tabu = {key: expiry for key, expiry in tabu.items() if expiry > iteration}
    final = mq.decode(inst, best)
    verified = mq.validate_schedule(inst, final["starts"]) == best_value
    return {"schema_version": 1, "instance": inst.name, "jobs": inst.jobs, "machines": inst.machines,
            "instance_sha256": mq.content_hash({"durations": inst.durations.tolist(), "routes": inst.routes.tolist()}),
            "seed": seed, "mode": mode, "evaluator": evaluator.backend,
            "neighborhood": neighborhood,
            "fallback_reason": evaluator.fallback_reason, "initial_makespan": initial_value,
            "best_makespan": int(best_value), "best_orders": best.tolist(), "starts": final["starts"],
            "independent_schedule_verification": bool(verified), "iterations": iteration,
            "restarts": restarts, "improvements": improvements, "trace": trace,
            "search_budget_seconds": seconds, "max_iterations": max_iterations,
            "setup_seconds": setup_seconds, "search_seconds": perf_counter() - search_start,
            "wall_seconds": perf_counter() - started, "graph_evaluations": evaluator.evaluations,
            "batch_evaluation_seconds": evaluator.seconds, "proposal_calls": proposal_calls,
            "proposal_evaluations": proposal_evaluations, "proposal_feasible": proposal_feasible,
            "proposal_improvements": proposal_improvements, "proposal_seconds": proposal_seconds,
            "training_evaluations": training_evaluations, "maximum_subspace_states": maximum_subspace,
            "maximum_candidate_data_qubits": maximum_data_qubits, "hardware_jobs_submitted": 0,
            "quantum_backend": "exact_classical_candidate_subspace" if mode == "quantum" else None,
            "notes": ["Dynamic candidate pools; global optimality is not certified.",
                      "Quantum mode is classical simulation of witness/XY/joint evolution.",
                      "Wall time includes setup/JIT; search budget starts after setup."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["classical", "uniform", "quantum"], default="classical")
    parser.add_argument("--backend", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = solve(mq.Instance.read(args.instance), seconds=args.seconds, seed=args.seed,
                   mode=args.mode, backend=args.backend, max_iterations=args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("instance", "mode", "best_makespan", "wall_seconds",
                                           "iterations", "proposal_improvements")}))


if __name__ == "__main__":
    main()
