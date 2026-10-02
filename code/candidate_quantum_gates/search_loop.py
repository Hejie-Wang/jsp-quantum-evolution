#!/usr/bin/env python3
"""Fixed-T quantum-assisted candidate search loop on small pools (P2).

Implements docs/candidate_quantum_improvements_20261002.md P2: the multilayer
fixed-T circuit ``legal initial state -> [witness phase -> single-machine XY
-> multi-machine joint transition] x p -> measurement`` plus the outer loop

    keep the verified initial schedule, T = best_makespan - 1
    repeat within budget:
        propose 2-4 machine joint actions from the current witnesses
        build and tune the multilayer circuit, sample candidate combinations
        deduplicate combinations, run the existing full graph evaluation
        feasible and better: store the schedule and update T
        infeasible or over T: extract, deduplicate and add effective witnesses
        record probabilities, evaluation counts, makespan and all timings

Small pools only: statevector training goes through ``ensure_simulation_budget``
and large pools are compiled offline by ``run_offline.compile_medium``.  Zero
energy on the known witness set only means the known constraints hold; every
accepted schedule still passes the independent full graph evaluation, and a
failed sample never proves infeasibility.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from circuits import (CandidatePool, JointTransition, WitnessSpec,
                      build_joint_mixer, build_phase_circuit, build_xy_mixer,
                      circuit_resources, decode_one_hot_counts, derive_witness,
                      ensure_simulation_budget, prepare_legal_basis,
                      prepare_uniform_legal)

_MEDIUM_DIR = Path(__file__).resolve().parents[1] / "candidate_quantum_medium"
if str(_MEDIUM_DIR) not in sys.path:
    sys.path.insert(0, str(_MEDIUM_DIR))
import medium_qjsp as mq  # noqa: E402  (sibling package, graph evaluation)

# The 3x3 validation fixture of code/candidate_quantum_demo/demo.py: nine
# operations whose pool contains cycles and a feasible optimum.  Operation id
# = job * 3 + position in job.
DEMO_DURATIONS = [[3, 2, 2], [2, 1, 4], [4, 3, 1]]
DEMO_ROUTES = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
DEMO_POOL = (((0, 5, 7), (7, 5, 0)),
             ((1, 3, 8), (3, 1, 8)),
             ((2, 4, 6), (4, 6, 2)))


def demo_instance():
    return mq.Instance(DEMO_DURATIONS, DEMO_ROUTES, name="demo_3x3")


def demo_candidate_pool():
    return CandidatePool(DEMO_POOL)


def random_instance(jobs, machines, seed):
    """Seeded random JSP instance; routes are per-job machine permutations."""
    rng = np.random.default_rng(seed)
    durations = rng.integers(1, 21, size=(jobs, machines))
    routes = np.array([rng.permutation(machines) for _ in range(jobs)])
    return mq.Instance(durations, routes, name=f"random_{jobs}x{machines}_seed{seed}")


def graph_witness_specs(pool: CandidatePool, witnesses):
    """Project medium graph witnesses onto pool-bound WitnessSpec objects."""
    return tuple(derive_witness(pool, {"kind": w.kind, "relations": w.relations,
                                       "length": w.length, "nodes": w.nodes})
                 for w in witnesses)


def covering_demo_witnesses(pool: CandidatePool, inst=None):
    """Real witnesses from full graph evaluation over the whole demo pool.

    Replaces the former hand-written fake witnesses: choice (0, 0, 0) of this
    pool is feasible, so it must never be labelled a cycle again.
    """
    inst = demo_instance() if inst is None else inst
    raw, seen = [], set()
    for choice in product(*[range(size) for size in pool.sizes]):
        result = mq.decode(inst, [list(pool.candidates[m][a])
                                  for m, a in enumerate(choice)])
        witness = mq.separate(inst, result)
        if witness.key() not in seen:
            seen.add(witness.key())
            raw.append(witness)
    return tuple(raw), graph_witness_specs(pool, raw)


def witness_active(spec: WitnessSpec, choice, t_target=None):
    """Classical predicate: every machine sits inside the allowed labels."""
    inside = all(label in allowed
                 for label, allowed in zip(choice, spec.allowed))
    if spec.kind == "path" and t_target is not None:
        return inside and spec.length > t_target
    return inside


def witness_energy(specs, choice, t_target, penalty):
    """H_T value of one choice on the frozen witness set (positive weights)."""
    return penalty * sum(witness_active(spec, choice, t_target) for spec in specs)


def propose_joint_actions(pool: CandidatePool, specs, incumbent, *,
                          max_actions=8, max_support=4):
    """Witness-derived 2-4 machine joint endpoints, no optimal-label access.

    For each witness the support machines (0 < |allowed| < pool size) are
    ranked most-constrained first; the left endpoint keeps the incumbent label
    when it is allowed (else the first allowed label) so the action starts
    where the conflict is active, and each right endpoint moves to the
    disallowed label closest in rank to the left one.  Every support machine
    leaves the witness's allowed set together, which single-machine XY moves
    cannot do in one step.
    """
    transitions, seen = [], set()
    ranked = []
    for index, spec in enumerate(specs):
        support = [(len(spec.allowed[m]), m) for m in range(pool.machines)
                   if 0 < len(spec.allowed[m]) < pool.sizes[m]]
        if len(support) < 2:
            continue
        support.sort()
        machines = tuple(m for _, m in support[:max_support])
        ranked.append((machines, index))
    ranked.sort()
    for machines, index in ranked[:max_actions]:
        spec = specs[index]
        left, right = [], []
        for m in machines:
            left.append(incumbent[m] if incumbent[m] in spec.allowed[m]
                        else spec.allowed[m][0])
            right.append(min((a for a in range(pool.sizes[m])
                              if a not in spec.allowed[m]),
                             key=lambda a: (abs(a - left[-1]), a)))
        key = (machines, tuple(left), tuple(right))
        if key in seen:
            continue
        seen.add(key)
        transitions.append(JointTransition(machines, tuple(left), tuple(right),
                                           witness_id=f"w{index}"))
    return tuple(transitions)


def _work_qubits(pool: CandidatePool, fixed_t=True):
    return pool.machines + 3


def build_fixed_t_ansatz(pool: CandidatePool, specs, transitions, layer_params,
                         fixed_t, penalty, *, initial_state="uniform",
                         mode="xy_joint"):
    """Legal initial state -> [phase -> XY -> joint] x layers circuit.

    ``mode`` selects the mixer content: ``uniform`` prepares and measures only,
    ``xy`` adds the witness phase and single-machine XY exchanges, and
    ``xy_joint`` additionally applies the witness-derived joint transitions.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    data = QuantumRegister(pool.data_qubits, "candidate")
    work = QuantumRegister(_work_qubits(pool), "work")
    circuit = QuantumCircuit(data, work)
    if mode == "uniform":
        preparation = prepare_uniform_legal(pool)
    elif initial_state == "uniform":
        preparation = prepare_uniform_legal(pool)
    else:
        preparation = prepare_legal_basis(pool)
    circuit.compose(preparation, qubits=list(range(pool.data_qubits)),
                    inplace=True)
    for gamma, beta_xy, beta_joint in layer_params:
        if mode == "uniform":
            break
        phase = build_phase_circuit(pool, specs, phase_angle=gamma,
                                    fixed_t=fixed_t, penalty=penalty)
        circuit.compose(phase, qubits=list(range(circuit.num_qubits)),
                        inplace=True)
        xy = build_xy_mixer(pool, beta_xy)
        circuit.compose(xy, qubits=list(range(pool.data_qubits)), inplace=True)
        if mode == "xy_joint" and transitions:
            joint = build_joint_mixer(pool, transitions, theta=beta_joint)
            circuit.compose(joint, qubits=list(range(pool.data_qubits)),
                            inplace=True)
    circuit.metadata = {
        "mode": mode, "fixed_t": fixed_t, "penalty": penalty,
        "layers": len(layer_params) if mode != "uniform" else 0,
        "initial_state": "uniform_legal" if mode == "uniform" or
                         initial_state == "uniform" else "legal_basis",
        "witnesses": len(specs), "transitions": len(transitions),
        "hardware_jobs_submitted": 0,
        "pool_hash": pool.content_hash,
    }
    return circuit


def data_probabilities(circuit, pool: CandidatePool, *, max_qubits=20):
    """Exact legal-subspace probabilities; work qubits must return to zero."""
    ensure_simulation_budget(circuit, max_qubits=max_qubits)
    from qiskit.quantum_info import Statevector
    vector = np.asarray(Statevector(circuit).data)
    data_mask = (1 << pool.data_qubits) - 1
    legal_mass = float(np.sum(np.abs(vector[(np.arange(len(vector)) >> pool.data_qubits) == 0]) ** 2))
    if abs(legal_mass - 1.0) > 1e-8:
        raise RuntimeError("work register not clean after ansatz")
    probabilities = np.abs(vector[:1 << pool.data_qubits]) ** 2
    per_choice = {}
    illegal_mass = 0.0
    for choice in product(*[range(size) for size in pool.sizes]):
        index = sum(1 << pool.offset(m, a) for m, a in enumerate(choice))
        per_choice[choice] = float(probabilities[index])
    illegal_mass = 1.0 - sum(per_choice.values())
    return per_choice, illegal_mass


def _expected_energy(per_choice, specs, t_target, penalty):
    return sum(prob * witness_energy(specs, choice, t_target, penalty)
               for choice, prob in per_choice.items() if prob > 0)


def train_layer_params(pool, specs, transitions, fixed_t, penalty, *,
                       layer_counts=(1, 2), gammas=(0.05, 0.12, 0.25),
                       betas=(0.1, 0.25, 0.45), max_evaluations=48):
    """Greedy coordinate search on the frozen witness energy landscape.

    Every evaluation is one circuit construction plus one statevector
    execution; both are counted and timed as training cost.  The objective is
    the expected H_T over the legal distribution, which uses only the frozen
    witness set, never optimal labels.
    """
    started = perf_counter()
    evaluations, log = 0, []
    best = None
    for layers in layer_counts:
        params = [(gammas[len(gammas) // 2], betas[len(betas) // 2],
                   betas[len(betas) // 2])] * layers
        if evaluations >= max_evaluations:
            break

        def score(parameters):
            nonlocal evaluations
            circuit = build_fixed_t_ansatz(pool, specs, transitions, parameters,
                                           fixed_t, penalty)
            per_choice, _ = data_probabilities(circuit, pool)
            evaluations += 1
            return _expected_energy(per_choice, specs, fixed_t, penalty)

        current = score(params)
        for position in range(layers):
            for axis in range(3):
                grid = gammas if axis == 0 else betas
                for value in grid:
                    if evaluations >= max_evaluations:
                        break
                    candidate = [list(p) for p in params]
                    candidate[position][axis] = value
                    candidate = tuple(tuple(p) for p in candidate)
                    energy = score(candidate)
                    if energy < current - 1e-12:
                        params, current = candidate, energy
        log.append({"layers": layers, "params": [list(p) for p in params],
                    "expected_energy": current})
        if best is None or current < best[1]:
            best = (params, current)
    return {"params": [list(p) for p in best[0]], "expected_energy": best[1],
            "evaluations": evaluations, "seconds": perf_counter() - started,
            "log": log}


def sample_choices(circuit, pool: CandidatePool, shots, seed):
    """Shot-based sampling of the data register on the local Aer simulator."""
    from qiskit import ClassicalRegister, QuantumCircuit
    from qiskit_aer import AerSimulator
    measured = QuantumCircuit(*circuit.qregs,
                              ClassicalRegister(pool.data_qubits, "readout"))
    measured.compose(circuit, qubits=list(range(circuit.num_qubits)),
                     inplace=True)
    measured.measure(list(range(pool.data_qubits)),
                     list(range(pool.data_qubits)))
    from qiskit import transpile
    measured = transpile(measured, basis_gates=["u", "cx"], optimization_level=0,
                         seed_transpiler=seed)
    counts = AerSimulator().run(measured, shots=shots,
                                seed_simulator=seed).result().get_counts()
    decoded = decode_one_hot_counts(counts, pool)
    legal = sorted(decoded["legal"].items(), key=lambda kv: -kv[1])
    return legal, decoded["illegal_fraction"]


def run_search_loop(inst, pool: CandidatePool, initial_choice, *,
                    mode="xy_joint", shots=256, seed=7, max_rounds=6,
                    time_budget=120.0, per_round_evaluations=8,
                    layer_counts=(1, 2), train_budget=48,
                    penalty=None, log=None):
    """Outer fixed-T closed loop; every accepted schedule is verified twice.

    ``mode``: ``uniform`` (uniform legal sampling), ``xy`` (phase + XY) or
    ``xy_joint`` (phase + XY + witness-derived joint transitions).
    """
    if mode not in {"uniform", "xy", "xy_joint"}:
        raise ValueError(f"unknown mode {mode!r}")
    started = perf_counter()
    if penalty is None:
        penalty = float(inst.total_duration + 1)
    choice = list(initial_choice)
    decoded = mq.decode(inst, mq.choice_orders(pool.candidates, choice))
    if not decoded["feasible"]:
        raise ValueError("initial choice must decode to a feasible schedule")
    best = mq._verified_best(inst, choice, decoded)
    initial_makespan = best["makespan"]
    t_target = best["makespan"] - 1
    witnesses, seen_keys = [], set()
    first = mq.separate(inst, decoded)
    witnesses.append(first)
    seen_keys.add(first.key())
    trace = []
    evaluation_count, training_seconds, training_evaluations = 1, 0.0, 0
    shots_total = 0
    stall = 0
    for round_index in range(max_rounds):
        if perf_counter() - started > time_budget or t_target < 0:
            break
        tic = perf_counter()
        specs = graph_witness_specs(pool, witnesses)
        effective = [s for s in specs
                     if any(0 < len(a) < size for a, size in
                            zip(s.allowed, pool.sizes))]
        transitions = (propose_joint_actions(pool, specs, best["choice"])
                       if mode == "xy_joint" else ())
        training = {"evaluations": 0, "seconds": 0.0}
        layer_params = []
        if mode != "uniform":
            training = train_layer_params(
                pool, effective, transitions, t_target, penalty,
                layer_counts=layer_counts, max_evaluations=train_budget)
            training_evaluations += training["evaluations"]
            training_seconds += training["seconds"]
            layer_params = [tuple(p) for p in training["params"]]
        ansatz = build_fixed_t_ansatz(pool, effective, transitions,
                                      layer_params, t_target, penalty,
                                      mode=mode)
        resources = circuit_resources(ansatz)
        sampled, illegal_fraction = sample_choices(ansatz, pool, shots,
                                                   seed + round_index)
        shots_total += shots
        round_elapsed_sampling = perf_counter() - tic
        evaluated, improved, added, feasible = 0, False, 0, 0
        probabilities = {tuple(ch): count / shots
                         for (ch, _t), count in sampled}
        for (choice_tuple, _t), count in sampled[:per_round_evaluations]:
            result = mq.decode(inst, mq.choice_orders(pool.candidates, list(choice_tuple)))
            evaluation_count += 1
            evaluated += 1
            if result["feasible"]:
                feasible += 1
                if result["makespan"] < best["makespan"]:
                    best = mq._verified_best(inst, list(choice_tuple), result)
                    t_target = best["makespan"] - 1
                    improved = True
            witness = mq.separate(inst, result)
            if witness.key() not in seen_keys:
                seen_keys.add(witness.key())
                witnesses.append(witness)
                added += 1
        trace.append({
            "round": round_index, "mode": mode, "target_T": t_target,
            "witnesses_frozen": len(witnesses),
            "effective_witnesses": len(effective),
            "joint_actions": len(transitions),
            "layers": len(layer_params), "trained_params": layer_params,
            "training": training, "shots": shots,
            "illegal_fraction": illegal_fraction,
            "unique_sampled": len(sampled),
            "top_probabilities": [
                {"choice": list(ch), "probability": probabilities[tuple(ch)]}
                for (ch, _t), _count in sampled[:5]],
            "evaluated": evaluated, "graph_feasible": feasible,
            "improved": improved, "witnesses_added": added,
            "best_makespan": best["makespan"],
            "sampling_seconds": round_elapsed_sampling,
            "round_seconds": perf_counter() - tic,
            "circuit": {"num_qubits": resources["num_qubits"],
                        "depth": resources["depth"],
                        "size": resources["size"],
                        "count_ops": resources["count_ops"]},
        })
        if log:
            log(f"round {round_index} [{mode}] T={t_target} "
                f"witnesses={len(witnesses)} actions={len(transitions)} "
                f"improved={improved} best={best['makespan']}")
        stall = 0 if (improved or added) else stall + 1
        if stall >= 2:
            break
    elapsed = perf_counter() - started
    verified = mq.validate_schedule(inst, best["starts"]) == best["makespan"]
    return {
        "schema_version": 1, "mode": mode, "instance": inst.name,
        "jobs": inst.jobs, "machines": inst.machines,
        "pool_sizes": list(pool.sizes),
        "pool_hash": pool.content_hash,
        "data_qubits": pool.data_qubits,
        "initial_makespan": initial_makespan,
        "best_makespan": best["makespan"], "best_choice": best["choice"],
        "improved": best["makespan"] < initial_makespan,
        "final_target_T": t_target,
        "independent_schedule_verification": verified,
        "witnesses": len(witnesses), "rounds": len(trace), "trace": trace,
        "evaluation_count": evaluation_count,
        "training_seconds": training_seconds,
        "training_evaluations": training_evaluations,
        "shots_total": shots_total,
        "wall_seconds": elapsed,
        "hardware_jobs_submitted": 0,
        "notes": [
            "Zero energy on the frozen witness set only satisfies the known "
            "constraints; the full graph evaluation remains mandatory.",
            "Sampling failures never prove infeasibility and quantum energies "
            "are not promised to decrease monotonically.",
            "The outer best makespan can only improve; T follows verified "
            "feasible decodes only.",
            "Statevector training is budget-guarded; large pools are offline "
            "compilation only.",
        ],
    }


def run_demo_closed_loop(shots=256, seed=7, mode="xy_joint", max_rounds=4):
    inst = demo_instance()
    pool = demo_candidate_pool()
    return run_search_loop(inst, pool, [0] * pool.machines, mode=mode,
                           shots=shots, seed=seed, max_rounds=max_rounds,
                           time_budget=300.0)


def classical_joint_search(inst, pool: CandidatePool, initial_choice, *,
                           seed=7, max_rounds=6, steps_per_round=200,
                           per_round_evaluations=8, time_budget=120.0,
                           log=None):
    """Classical search with the same information and the same joint actions.

    Hill climbing on the frozen witness energy: moves are single-machine label
    changes or exactly the transitions ``propose_joint_actions`` derives, so
    the classical baseline sees the same cuts and the same joint moves.  A
    zero-violation configuration triggers the full graph evaluation, exactly
    like a sampled combination in the quantum loop.
    """
    rng = np.random.default_rng(seed)
    started = perf_counter()
    decoded = mq.decode(inst, mq.choice_orders(pool.candidates, list(initial_choice)))
    if not decoded["feasible"]:
        raise ValueError("initial choice must decode to a feasible schedule")
    best = mq._verified_best(inst, list(initial_choice), decoded)
    initial_makespan = best["makespan"]
    t_target = best["makespan"] - 1
    witnesses, seen_keys = [], set()
    first = mq.separate(inst, decoded)
    witnesses.append(first)
    seen_keys.add(first.key())
    trace = []
    evaluation_count = 1
    stall = 0
    for round_index in range(max_rounds):
        if perf_counter() - started > time_budget or t_target < 0:
            break
        tic = perf_counter()
        specs = graph_witness_specs(pool, witnesses)
        effective = [s for s in specs
                     if any(0 < len(a) < size for a, size in
                            zip(s.allowed, pool.sizes))]
        penalty = float(inst.total_duration + 1)
        current = list(best["choice"])
        current_energy = witness_energy(specs, current, t_target, penalty)
        steps = 0
        zero_seen = 0
        while steps < steps_per_round and zero_seen < per_round_evaluations:
            steps += 1
            actions = propose_joint_actions(pool, specs, current)
            if actions and rng.random() < 0.5:
                action = actions[int(rng.integers(len(actions)))]
                candidate = list(current)
                for m, a in zip(action.support, action.right):
                    candidate[m] = a
            else:
                m = int(rng.integers(pool.machines))
                candidate = list(current)
                candidate[m] = int(rng.integers(pool.sizes[m]))
            energy = witness_energy(specs, candidate, t_target, penalty)
            if energy <= current_energy:
                current, current_energy = candidate, energy
            if current_energy == 0.0:
                zero_seen += 1
                key = tuple(current)
                result = mq.decode(inst, mq.choice_orders(pool.candidates, list(key)))
                evaluation_count += 1
                if result["feasible"] and result["makespan"] < best["makespan"]:
                    best = mq._verified_best(inst, list(key), result)
                    t_target = best["makespan"] - 1
                witness = mq.separate(inst, result)
                if witness.key() not in seen_keys:
                    seen_keys.add(witness.key())
                    witnesses.append(witness)
                # restart the walk from the incumbent after an evaluation
                current = list(best["choice"])
                current_energy = witness_energy(specs, current, t_target, penalty)
        trace.append({
            "round": round_index, "steps": steps,
            "witnesses": len(witnesses), "joint_actions": len(actions),
            "evaluations": zero_seen, "best_makespan": best["makespan"],
            "round_seconds": perf_counter() - tic,
        })
        if log:
            log(f"classical round {round_index}: best={best['makespan']}")
        improved_this_round = trace[-1]["best_makespan"] < (
            trace[round_index - 1]["best_makespan"] if round_index else initial_makespan)
        if not improved_this_round and not any(
                t.get("witnesses_added") for t in trace[round_index:]):
            stall += 1
        else:
            stall = 0
        if stall >= 2:
            break
    elapsed = perf_counter() - started
    verified = mq.validate_schedule(inst, best["starts"]) == best["makespan"]
    return {
        "schema_version": 1, "mode": "classical_joint",
        "instance": inst.name, "jobs": inst.jobs, "machines": inst.machines,
        "pool_sizes": list(pool.sizes), "pool_hash": pool.content_hash,
        "initial_makespan": initial_makespan,
        "best_makespan": best["makespan"], "best_choice": best["choice"],
        "improved": best["makespan"] < initial_makespan,
        "final_target_T": t_target,
        "independent_schedule_verification": verified,
        "witnesses": len(witnesses), "rounds": len(trace), "trace": trace,
        "evaluation_count": evaluation_count, "training_seconds": 0.0,
        "training_evaluations": 0, "shots_total": 0,
        "wall_seconds": elapsed, "hardware_jobs_submitted": 0,
        "notes": ["Classical hill climbing on the same frozen witness cuts "
                  "and the same joint actions as the quantum loop."],
    }


def _cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true",
                        help="run the 3x3 demo closed loop")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--machines", type=int)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", default="xy_joint",
                        choices=["uniform", "xy", "xy_joint", "classical"])
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--schedules", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.demo:
        report = run_demo_closed_loop(shots=args.shots, seed=args.seed,
                                      mode="xy_joint" if args.mode == "classical"
                                      else args.mode,
                                      max_rounds=args.rounds)
    else:
        if not (args.jobs and args.machines):
            parser.error("random instances need --jobs and --machines")
        inst = random_instance(args.jobs, args.machines, args.seed)
        pool_data, _info = mq.build_pool(
            inst, candidates_per_machine=args.candidates,
            n_schedules=args.schedules, seed=args.seed)
        pool = CandidatePool(tuple(tuple(tuple(order) for order in machine)
                                   for machine in pool_data))
        if args.mode == "classical":
            report = classical_joint_search(inst, pool, [0] * pool.machines,
                                            seed=args.seed,
                                            max_rounds=args.rounds)
        else:
            report = run_search_loop(inst, pool, [0] * pool.machines,
                                     mode=args.mode, shots=args.shots,
                                     seed=args.seed, max_rounds=args.rounds,
                                     time_budget=600.0)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
