#!/usr/bin/env python3
"""T06: independent contribution of the quantum distribution and joint-strength ablation.

Implements the T06 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md on frozen windows: the
machine-candidate pool, the projection witness set and the fixed target T are
identical for every arm, only the proposal mechanism changes.

Arms (all with the same pool / witnesses / T / initial information):

  ``uniform``     independent uniform sampling of legal one-hot candidates
                  (the cheap classical control from T00, exact marginals);
  ``sa_witness``  classical simulated annealing on the *same* witness energy
                  H_T = penalty * #{triggered witnesses}, evaluated directly
                  (no precomputed energy table); a zero-violation configuration
                  triggers the full graph evaluation, exactly like a sampled
                  combination;
  ``xy``          phase + single-machine XY mixing, trained by greedy
                  coordinate search on the expected H_T;
  ``xy_joint``    phase + XY + witness-derived joint transitions (multi-witness
                  conflict repair, never label-index adjacency);
  ``exact``       offline complete enumeration of the pool, listed separately
                  and never fed back into any arm.

Reported per arm and window: hit probability (legal + acyclic + improving),
feasibility and uniqueness of the samples, parameter evaluations, training
shots, training time, graph evaluations per hit and total time.  The two
training modes are reported separately as the work package requires: the exact
expectation over the compact simulator (ideal mechanism) and the finite-shot
estimate at the pre-registered 64 shots per parameter evaluation
(hardware-facing).  A variational arm never learns from the verification set.

Scope discipline: the objective every arm optimises is the FROZEN witness
energy, never the true makespan table; the true makespan is used only for the
final verification of each accepted combination, and the pool optimum is an
offline diagnostic.  No backend or hardware is used.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import Counter
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
import search_loop as sl  # noqa: E402
from window_diagnostics import check_choice  # noqa: E402
from witness_surrogate import collect_witnesses  # noqa: E402

SCHEMA_VERSION = 1
MAIN_SHOTS = 64
SECONDARY_SHOTS = 256
TRAIN_EVALUATIONS = 12
PENALTY = 1.0
LAYER_COUNTS = (1, 2)
GAMMAS = (0.05, 0.12, 0.25)
BETAS = (0.1, 0.25, 0.45)


# ---------------------------------------------------------------------------
# Shared window setup
# ---------------------------------------------------------------------------

def candidate_pool(pool):
    return circuits.CandidatePool(
        tuple(tuple(tuple(int(v) for v in order) for order in machine)
              for machine in pool))


def window_setup(inst, pool, incumbent, u0, n_probe=16, seed=0, target_t=None):
    """Frozen (pool, witness specs, transitions, T) shared by every arm."""
    witnesses, stats = collect_witnesses(inst, pool, tuple(incumbent),
                                         n_probe=n_probe, seed=seed)
    medium = [mq.Witness(w["kind"], tuple(tuple(r) for r in w["relations"]),
                         int(w["length"]), ()) for w in witnesses]
    pool_obj = candidate_pool(pool)
    specs = sl.graph_witness_specs(pool_obj, medium)
    transitions = sl.propose_joint_actions(pool_obj, specs, tuple(incumbent))
    return {"pool_obj": pool_obj, "specs": specs, "transitions": transitions,
            "witness_stats": stats,
            "target_t": int(u0) - 1 if target_t is None else int(target_t)}


# ---------------------------------------------------------------------------
# Arm 1: independent uniform sampling (exact marginals, zero training)
# ---------------------------------------------------------------------------

def uniform_arm(pool, inst, u0, target_t):
    sizes = [len(machine) for machine in pool]
    total = int(np.prod(sizes))
    improving = 0
    feasible = 0
    for choice in itertools.product(*(range(k) for k in sizes)):
        result = check_choice(inst, pool, choice)
        if result["feasible"]:
            feasible += 1
            if result["makespan"] < u0:
                improving += 1
    p_hit = improving / total
    return {"arm": "uniform", "p_hit": p_hit, "feasible_rate": feasible / total,
            "unique_choices": total, "parameter_evaluations": 0,
            "training_shots": 0, "training_seconds": 0.0,
            "graph_evaluations_per_hit": (1.0 / p_hit) if p_hit > 0 else None,
            "evaluations_to_first_hit": (1.0 / p_hit) if p_hit > 0 else None,
            "sample_source": "exact_uniform_marginals"}


# ---------------------------------------------------------------------------
# Arm 2: simulated annealing on the same witness energy
# ---------------------------------------------------------------------------

def sa_arm(inst, pool, setup, u0, *, runs=20, steps=200, seed=7,
           t0_fraction=0.5):
    """Classical SA on H_T; a zero-violation configuration hits the graph check."""
    specs = setup["specs"]
    target_t = setup["target_t"]
    sizes = [len(machine) for machine in pool]
    improving_hits = 0
    witness_evaluations = 0
    graph_evaluations = 0
    first_hit_steps = []
    started = time.perf_counter()
    for run in range(runs):
        rng = np.random.default_rng(seed + run)
        choice = [int(rng.integers(k)) for k in sizes]
        current = sl.witness_energy(specs, choice, target_t, PENALTY)
        witness_evaluations += 1
        best = None
        temperature = t0_fraction * PENALTY
        cooling = (0.01 / max(temperature, 1e-9)) ** (1.0 / max(1, steps))
        for step in range(steps):
            candidate = list(choice)
            if setup["transitions"] and rng.random() < 0.3:
                action = setup["transitions"][int(rng.integers(len(setup["transitions"])))]
                for m, a in zip(action.support, action.right):
                    candidate[m] = a
            else:
                m = int(rng.integers(len(sizes)))
                candidate[m] = int(rng.integers(sizes[m]))
            energy = sl.witness_energy(specs, candidate, target_t, PENALTY)
            witness_evaluations += 1
            if energy <= current or rng.random() < math.exp(-(energy - current) / max(temperature, 1e-9)):
                choice, current = candidate, energy
            if current == 0.0:
                graph_evaluations += 1
                result = check_choice(inst, pool, tuple(choice))
                if result["feasible"] and result["makespan"] < u0:
                    if best is None:
                        best = result["makespan"]
                        first_hit_steps.append(step)
            temperature *= cooling
        improving_hits += int(best is not None)
    seconds = time.perf_counter() - started
    p_hit = improving_hits / runs
    return {"arm": "sa_witness", "p_hit": p_hit,
            "feasible_rate": None, "unique_choices": None,
            "parameter_evaluations": 0, "training_shots": 0,
            "training_seconds": 0.0,
            "witness_evaluations_per_run": witness_evaluations / runs,
            "graph_evaluations_per_hit": (graph_evaluations / improving_hits
                                          if improving_hits else None),
            "mean_steps_to_first_hit": (float(np.mean(first_hit_steps))
                                        if first_hit_steps else None),
            "runs": runs, "steps_per_run": steps, "seconds": seconds,
            "sample_source": "simulated_annealing_on_witness_energy"}


# ---------------------------------------------------------------------------
# Arms 3-4: variational phase + XY (+ joint), exact and shot-based training
# ---------------------------------------------------------------------------

def variational_arm(inst, pool, setup, u0, *, mode, training="exact",
                    train_evaluations=TRAIN_EVALUATIONS, shots=MAIN_SHOTS,
                    seed=7):
    """Train one variational arm and measure its sampling quality.

    ``training='exact'`` scores parameter candidates with the exact expectation
    over the compact simulator (ideal mechanism, no shots); ``training='shots'``
    scores them from a finite-shot estimate at the pre-registered budget, which
    is the number a hardware-facing claim must use.
    """
    specs, transitions = setup["specs"], setup["transitions"]
    target_t = setup["target_t"]
    pool_obj = setup["pool_obj"]
    from compact_simulator import CompactSimulator
    simulator = CompactSimulator(pool_obj, specs, transitions, target_t, PENALTY)
    evaluations = 0
    deadline_note = None
    started = time.perf_counter()
    train_shots = 0

    def score(params):
        nonlocal evaluations, train_shots
        evaluations += 1
        if training == "exact":
            return float(simulator.probabilities(params, mode) @ simulator.energy)
        circuit = sl.build_fixed_t_ansatz(pool_obj, specs, transitions, params,
                                          target_t, PENALTY, mode=mode)
        legal, _illegal = sl.sample_choices(circuit, pool_obj, shots, seed + evaluations)
        train_shots += shots
        total = sum(frequency for _key, frequency in legal) or 1
        energy = 0.0
        for key, frequency in legal:
            choice = key[0] if isinstance(key, tuple) else key
            energy += (frequency / total) * sl.witness_energy(specs, choice,
                                                              target_t, PENALTY)
        return energy

    params = [(GAMMAS[1], BETAS[1], BETAS[1])] * 2
    current = score(params)
    for position in range(2):
        for axis in range(3):
            grid = GAMMAS if axis == 0 else BETAS
            for value in grid:
                if evaluations >= train_evaluations:
                    break
                candidate = [list(p) for p in params]
                candidate[position][axis] = value
                candidate = tuple(tuple(p) for p in candidate)
                energy = score(candidate)
                if energy < current - 1e-12:
                    params, current = candidate, energy
    training_seconds = time.perf_counter() - started

    # evaluation of the trained arm: exact legal distribution and shot estimate
    exact_probabilities = simulator.probabilities(params, mode)
    improving_mass, feasible_mass, unique = 0.0, 0.0, 0
    for index, choice in enumerate(itertools.product(
            *(range(k) for k in [len(m) for m in pool]))):
        probability = float(exact_probabilities[index])
        if probability <= 0:
            continue
        unique += 1
        result = check_choice(inst, pool, choice)
        if not result["feasible"]:
            continue
        feasible_mass += probability
        if result["makespan"] < u0:
            improving_mass += probability
    circuit = sl.build_fixed_t_ansatz(pool_obj, specs, transitions, params,
                                      target_t, PENALTY, mode=mode)
    shot_estimates = {}
    for label, shot_count in (("main", shots), ("secondary", SECONDARY_SHOTS)):
        sampled, illegal_fraction = sl.sample_choices(circuit, pool_obj, shot_count,
                                                      seed + 991 + shot_count)
        sampled_hits = 0
        sampled_total = 0
        for key, frequency in sampled:
            choice = key[0] if isinstance(key, tuple) else key
            sampled_total += frequency
            result = check_choice(inst, pool, choice)
            if result["feasible"] and result["makespan"] < u0:
                sampled_hits += frequency
        shot_estimates[label] = {
            "shots": shot_count,
            "sampled_p_hit": (sampled_hits / sampled_total) if sampled_total else None,
            "illegal_fraction": illegal_fraction,
        }
    main_estimate = shot_estimates["main"]["sampled_p_hit"]
    return {"arm": mode, "training": training, "p_hit": improving_mass,
            "sampled_p_hit": main_estimate,
            "shot_estimates": shot_estimates,
            "feasible_rate": feasible_mass, "unique_choices": unique,
            "parameter_evaluations": evaluations, "training_shots": train_shots,
            "training_seconds": training_seconds,
            "trained_params": [list(p) for p in params],
            "expected_witness_energy": float(exact_probabilities @ simulator.energy),
            "illegal_fraction_shots": shot_estimates["main"]["illegal_fraction"],
            "graph_evaluations_per_hit": (1.0 / improving_mass) if improving_mass > 0 else None,
            "shots": shots,
            "sample_source": f"variational_{mode}_{training}_training"}


# ---------------------------------------------------------------------------
# Offline reference (never fed back)
# ---------------------------------------------------------------------------

def exact_reference(inst, pool, u0, target_t, specs=None):
    """Offline enumeration: pool optimum, improving/target sets and, when the
    witness set is given, how many improving combinations the surrogate can see
    at all (zero witness energy).  This alignment count is the attribution key
    for the variational arms."""
    sizes = [len(machine) for machine in pool]
    total = int(np.prod(sizes))
    best, improving, reaching, improving_zero_energy = None, 0, 0, 0
    zero_energy = 0
    for choice in itertools.product(*(range(k) for k in sizes)):
        if specs is not None and sl.witness_energy(specs, choice, target_t,
                                                   PENALTY) == 0.0:
            zero_energy += 1
        result = check_choice(inst, pool, choice)
        if not result["feasible"]:
            continue
        if best is None or result["makespan"] < best:
            best = result["makespan"]
        if result["makespan"] < u0:
            improving += 1
            if specs is not None and sl.witness_energy(specs, choice, target_t,
                                                       PENALTY) == 0.0:
                improving_zero_energy += 1
        reaching += int(result["makespan"] <= target_t)
    return {"arm": "exact", "pool_optimum": best, "improving_combinations": improving,
            "target_reaching_combinations": reaching,
            "improving_with_zero_witness_energy": (improving_zero_energy
                                                   if specs is not None else None),
            "zero_witness_energy_combinations": (zero_energy if specs is not None
                                                 else None),
            "p_hit_ceiling_on_zero_energy_set": (
                (improving_zero_energy / zero_energy)
                if (specs is not None and zero_energy) else None),
            "graph_evaluations": total, "p_hit": improving / total}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_window(inst, pool, incumbent, u0, *, seed=0, shots=MAIN_SHOTS,
               runs=20, sa_steps=200, train_evaluations=TRAIN_EVALUATIONS):
    setup = window_setup(inst, pool, incumbent, u0, seed=seed)
    rows = [uniform_arm(pool, inst, u0, setup["target_t"])]
    rows.append(sa_arm(inst, pool, setup, u0, runs=runs, steps=sa_steps,
                       seed=seed + 1))
    for mode in ("xy", "xy_joint"):
        for training in ("exact", "shots"):
            rows.append(variational_arm(inst, pool, setup, u0, mode=mode,
                                        training=training,
                                        train_evaluations=train_evaluations,
                                        shots=shots, seed=seed + 7))
    reference = exact_reference(inst, pool, u0, setup["target_t"],
                                specs=setup["specs"])
    return {"target_t": setup["target_t"], "arms": rows, "reference": reference,
            "witness_stats": setup["witness_stats"],
            "transitions": len(setup["transitions"])}


def aggregate(rows):
    """Macro average over windows plus a paired bootstrap vs the uniform arm."""
    paired = {}
    for row in rows:
        for arm in row["arms"]:
            key = (arm["arm"], arm.get("training", "n/a"))
            paired.setdefault(key, []).append((row["window"]["scale"],
                                               row["window"]["seed"], arm))
    summary = []
    for (arm, training), entries in sorted(paired.items()):
        hits = [e[2]["p_hit"] for e in entries]
        summary.append({"arm": arm, "training": training, "windows": len(entries),
                        "mean_p_hit": float(np.mean(hits)),
                        "median_p_hit": float(np.median(hits)),
                        "windows_with_hit": int(sum(1 for h in hits if h > 0))})
    base = {e[:2]: e[2]["p_hit"] for e in paired[("uniform", "n/a")]}
    rng = np.random.default_rng(7)
    for row in summary:
        if row["arm"] == "uniform":
            row["paired_ratio_vs_uniform"] = None
            continue
        entries = paired[(row["arm"], row["training"])]
        ratios = [entry[2]["p_hit"] / base[entry[:2]]
                  for entry in entries if base.get(entry[:2], 0) > 0]
        if not ratios:
            row["paired_ratio_vs_uniform"] = None
            continue
        boot = [float(np.mean(rng.choice(ratios, len(ratios), replace=True)))
                for _ in range(10000)]
        row["paired_ratio_vs_uniform"] = {
            "mean": float(np.mean(ratios)), "n_windows_ratio_defined": len(ratios),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "exploratory": True,
            "note": "window resampling over a handful of pilot windows; not a "
                    "confirmation-set result (T06 requires D2 windows)",
        }
    return summary


def main(argv=None):
    import data_identity
    from window_diagnostics import build_window
    ap = argparse.ArgumentParser(description="T06 sampling ablation")
    ap.add_argument("--scales", default="3x3,4x3")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--pool-k", type=int, default=3)
    ap.add_argument("--shots", type=int, default=MAIN_SHOTS)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--sa-steps", type=int, default=200)
    ap.add_argument("--train-evaluations", type=int, default=TRAIN_EVALUATIONS)
    ap.add_argument("--max-per-scale", type=int, default=3,
                    help="windows with at least one improving combination per scale")
    ap.add_argument("--negatives", type=int, default=1,
                    help="no-improvement windows kept per scale as negative controls")
    ap.add_argument("--out", default=str(GATES_DIR / "results_sampling_ablation_20261003"
                                         / "sampling_ablation.json"))
    args = ap.parse_args(argv)
    lo, hi = args.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    wanted = set()
    for token in args.scales.split(","):
        jobs, machines = token.strip().lower().split("x")
        wanted.add((int(jobs), int(machines)))

    # Window selection follows the T03 recommendation: reserve windows that do
    # contain an improving combination for the mechanism ablation, and keep
    # no-improvement windows only as negative controls (invalid-call cost).
    chosen = []
    for jobs, machines in data_identity.D0_SIZES:
        if (jobs, machines) not in wanted:
            continue
        positive, negative = [], []
        for seed in seeds:
            inst = data_identity.build_d0(jobs, machines, seed)
            pool, incumbent, u0 = build_window(inst, args.pool_k, seed + 1000)
            reference = exact_reference(inst, pool, u0, u0 - 1)
            entry = (seed, inst, pool, incumbent, u0, reference)
            (positive if reference["improving_combinations"] > 0 else negative).append(entry)
        chosen.extend(positive[:args.max_per_scale])
        chosen.extend(negative[:args.negatives])
        print(f"  {jobs}x{machines}: {len(positive)} improving windows, "
              f"{len(negative)} negative windows; using "
              f"{min(len(positive), args.max_per_scale)} + {min(len(negative), args.negatives)}",
              flush=True)

    results = []
    for seed, inst, pool, incumbent, u0, _reference in chosen:
        row = run_window(inst, pool, incumbent, u0, seed=seed, shots=args.shots,
                         runs=args.runs, sa_steps=args.sa_steps,
                         train_evaluations=args.train_evaluations)
        row["window"] = {"scale": f"{inst.jobs}x{inst.machines}", "seed": seed,
                         "u0": int(u0),
                         "instance_sha256": data_identity.instance_sha256(inst),
                         "pool_sha256": mq.content_hash(pool)}
        results.append(row)
        hits = {a["arm"] + "/" + str(a.get("training", "n/a")): round(a["p_hit"], 4)
                for a in row["arms"]}
        print(f"  {inst.jobs}x{inst.machines} seed {seed}: {hits}", flush=True)
    output = {
        "schema_version": SCHEMA_VERSION,
        "work_package": "T06 (Issue #19, PR #18 docs TODO)",
        "protocol": {"main_shots": MAIN_SHOTS, "secondary_shots": SECONDARY_SHOTS,
                     "train_evaluations": args.train_evaluations,
                     "penalty": PENALTY, "layer_counts": list(LAYER_COUNTS),
                     "gamma_grid": list(GAMMAS), "beta_grid": list(BETAS),
                     "sa": {"runs": args.runs, "steps": args.sa_steps}},
        "windows": results, "summary": aggregate(results),
        "notes": [
            "Every arm shares the same pool, witness set, fixed T and initial information.",
            "No arm reads the exact enumeration result; it is an offline reference only.",
            "Training is reported twice: exact compact-simulator expectation and finite-shot estimate.",
            "The objective is the frozen witness energy, never the true makespan table.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=1, default=_json_default),
                   encoding="utf-8")
    print(f"wrote {out}")
    for row in output["summary"]:
        print("  summary", row["arm"], row["training"], "mean p_hit=",
              round(row["mean_p_hit"], 4), "windows_with_hit=", row["windows_with_hit"])
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
