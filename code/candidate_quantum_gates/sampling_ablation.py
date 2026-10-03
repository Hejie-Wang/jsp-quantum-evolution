#!/usr/bin/env python3
"""T06: controlled sampling ablation on frozen D2 windows.

Implements the T06 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md as a *frozen-window*
component comparison: every arm sees the SAME pool, witness set, target T and
incumbent information; only the proposal distribution differs.  No arm reads
the exact enumeration answers - the oracle is computed offline solely to
score the sampled raw frequencies and is reported separately as the
small-pool benchmark.

Arms (shots=64 main batch, raw frequencies kept, duplicates never dropped):

  uniform          independent per-machine label draws (UniformSampler)
  sa_energy        classical annealing chain on the SAME surrogate energy the
                   quantum arms train on (64 energy evaluations, visited
                   states kept as raw samples)
  xy               witness phase + XY mixer, uniform legal initial state,
                   exact-expectation training, 12 parameter evaluations
  xy_joint         phase + XY + witness-derived joint transitions
  xy_joint_warm    xy_joint with the incumbent basis state as initial state
  mixers_only      XY + joint transitions with an EMPTY witness set: the
                   phase ablation (nothing to trigger, pure mixer sampling)

The single-toggle ablations are therefore: phase (xy_joint vs mixers_only),
joint gates (xy_joint vs xy), warm start (xy_joint vs xy_joint_warm).
Training-objective ablation: exact full-state expectation (main arms) vs
finite-shots energy estimation at the same 12-evaluation budget
(`xy_joint_shots`), with all shots counted.

Metrics per window/arm: raw p_imp (probability mass on feasible improvements),
feasible rate, unique rate, parameter evaluations, training seconds, graph
evaluations (unique scored combos, cached across arms per window) and the
exact-oracle rho for reference.  p_imp=0 windows are kept as negative
controls and report the invalid-call cost instead of being dropped.

Gate discipline: the pre-registered comparison is quantum vs the uniform
control at the same shots; the common-benefit threshold (ratio >= 1.10 with
95% bootstrap CI lower bound > 1, paired across windows) decides whether a
distribution-level claim is allowed.  End-to-end claims additionally require
training shots/evaluations to be counted; both are reported, neither is
assumed away.  Classical simulation only: no QPU, no hardware claims.
"""
from __future__ import annotations

import argparse
import gzip
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

import medium_qjsp as mq  # noqa: E402
from circuits import CandidatePool  # noqa: E402
from compact_simulator import CompactSimulator  # noqa: E402
from search_loop import graph_witness_specs, propose_joint_actions  # noqa: E402
from uniform_sampler import UniformSampler  # noqa: E402
from window_diagnostics import check_choice  # noqa: E402
from window_pool_builder import STRATEGIES, _stable_rng  # noqa: E402

SCHEMA_VERSION = 1
MAIN_SHOTS = 64
SECONDARY_SHOTS = 256
TRAIN_BUDGET = 12
LAYERS = 2
ARMS = ("uniform", "sa_energy", "xy", "xy_joint", "xy_joint_warm",
        "mixers_only", "xy_joint_shots")


# ---------------------------------------------------------------------------
# Frozen-window loading (D2-dev artifacts of window_pool_builder)
# ---------------------------------------------------------------------------

def load_d2_windows(results_dir=None):
    """Load frozen D2-dev windows: (windows, snapshot lookup, witness lookup)."""
    directory = Path(results_dir) if results_dir else \
        GATES_DIR / "results_window_pool_20261003"
    payload = json.loads((directory / "window_pool_d2_dev.json")
                         .read_text(encoding="utf-8"))
    snapshots = {s["snapshot_key"]: s for s in payload["d2"]["snapshots"]}
    witnesses = {}
    for table in payload["d2"].get("witness_table", []):
        witnesses[table["trajectory_key"]] = table["witnesses"]
    if not witnesses:
        sidecar = directory / "window_pool_d2_dev_witnesses.json.gz"
        with gzip.open(sidecar, "rt", encoding="utf-8") as fh:
            witnesses = {t["trajectory_key"]: t["witnesses"]
                         for t in json.load(fh)}
    return payload["d2"]["windows"], snapshots, witnesses


def window_context(record, snapshots, witness_tables):
    """Rebuild (pool, witnesses, target, incumbent) for one frozen window."""
    pool = CandidatePool(
        tuple(tuple(tuple(int(v) for v in order) for order in candidates)
              for candidates in record["pool"]))
    snap = snapshots[record["snapshot_key"]]
    table = witness_tables[snap["trajectory_key"]]
    raw = []
    for index in snap["witness_indices"]:
        w = table[index]
        rels = tuple(tuple(w["relations_flat"][i:i + 3])
                     for i in range(0, len(w["relations_flat"]), 3))
        raw.append(mq.Witness(w["kind"], rels, int(w["length"]), ()))
    incumbent = tuple(int(a) for a in record["incumbent"])
    return pool, raw, int(record["target_t"]), incumbent


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def arm_uniform(sizes, shots, rng):
    sampler = UniformSampler(sizes)
    return [tuple(int(x) for x in draw)
            for draw in sampler.sample(shots, rng, deduplicate=False)]


def arm_sa(surrogate, incumbent, shots, rng):
    """Annealing chain on the surrogate energy; visited states = raw samples.

    Same evaluation accounting as the quantum training budget would impose on
    an oracle-free classical consumer: one energy lookup per proposal, 64
    proposals, Metropolis acceptance over the label space.
    """
    sizes = list(surrogate["sizes"])
    state = list(incumbent)
    energy = surrogate["energy_of"](tuple(state))
    temperature = max(1.0, float(surrogate["scale"]))
    samples = [tuple(state)]
    for step in range(shots - 1):
        trial = list(state)
        m = int(rng.integers(len(sizes)))
        trial[m] = int(rng.integers(sizes[m]))
        e = surrogate["energy_of"](tuple(trial))
        delta = e - energy
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            state, energy = trial, e
        temperature = max(0.05, temperature * 0.97)
        samples.append(tuple(state))
    return samples


def arm_quantum(simulator, shots, rng, mode, warm=False,
                train_budget=TRAIN_BUDGET, shots_training=False):
    """Train (exact expectation or finite-shots) and draw one raw batch.

    ``warm=True`` keeps the training pipeline identical and only changes the
    sampling-time initial state to the incumbent basis state (the simulator's
    train/probabilities path is uniform-initialised); the ablation therefore
    isolates the initial-state effect at sampling time with shared params.
    """
    if shots_training:
        params, evaluations, seconds, shots_used = _train_finite_shots(
            simulator, train_budget, rng, mode)
    else:
        trained = simulator.train(mode=mode, budget=train_budget,
                                  seed=int(rng.integers(1 << 30)))
        params, evaluations, seconds = trained["params"], \
            trained["evaluations"], trained["seconds"]
        shots_used = 0
    if warm:
        state = simulator.state(np.array(params), mode, initial_state="basis")
        probabilities = np.abs(state) ** 2
        probabilities = probabilities / probabilities.sum()
    else:
        probabilities = simulator.probabilities(np.array(params), mode)
    draws = rng.choice(simulator.dimension, shots, p=probabilities)
    samples = [tuple(simulator.labels[i]) for i in draws]
    return samples, {"parameter_evaluations": evaluations,
                     "training_seconds": seconds,
                     "training_shots": shots_used,
                     "unique_training_states": int(np.count_nonzero(probabilities))}


def _train_finite_shots(simulator, budget, rng, mode):
    """Energy estimate from multinomial shots instead of exact expectation.

    Mirrors ``CompactSimulator.train`` (same initialisation, same proposal
    structure) with the exact expectation replaced by a raw sample mean; every
    shot is counted in ``training_shots``.
    """
    start = time.perf_counter()
    shots_per_eval = 64
    rng_train = np.random.default_rng(int(rng.integers(1 << 30)))
    scale = max(1.0, float(simulator.energy.max()))
    params = np.tile([0.7 / scale, 0.3, 0.2], (LAYERS, 1))
    best, best_value, used, shots_used = params, float("inf"), 0, 0
    for i in range(budget):
        trial = params.copy()
        if i:
            k = int(rng_train.integers(LAYERS))
            axis = int(rng_train.integers(3 if mode == "xy_joint" else 2))
            trial[k, axis] = float(rng_train.uniform(
                0, 2 * np.pi / scale if axis == 0 else np.pi))
        probabilities = simulator.probabilities(trial, mode)
        draws = rng_train.choice(simulator.dimension, shots_per_eval,
                                 p=probabilities)
        value = float(np.mean(simulator.energy[draws]))
        used += 1
        shots_used += shots_per_eval
        if value < best_value:
            best, best_value = trial, value
    return best, used, time.perf_counter() - start, shots_used


def surrogate_view(pool, specs, target, penalty):
    """Shared surrogate: per-state energy over the whole legal label space."""
    simulator = CompactSimulator(pool, specs, (), target, penalty)
    energy = simulator.energy
    labels = [tuple(int(x) for x in row) for row in simulator.labels]
    lookup = {labels[i]: float(energy[i]) for i in range(len(labels))}
    return {
        "sizes": pool.sizes,
        "energy_of": lambda choice: lookup.get(tuple(choice),
                                               float(penalty * (target + 1))),
        "scale": float(np.max(energy)) if len(energy) else 1.0,
        "dimension": len(labels), "labels": labels,
    }


# ---------------------------------------------------------------------------
# One window, all arms
# ---------------------------------------------------------------------------

def run_window(inst, record, snapshots, witness_tables, *, shots=MAIN_SHOTS,
               secondary_shots=SECONDARY_SHOTS, train_budget=TRAIN_BUDGET,
               penalty=None, seed=7):
    pool, raw_witnesses, target, incumbent = window_context(
        record, snapshots, witness_tables)
    if penalty is None:
        penalty = float(inst.total_duration + 1)
    specs = [s for s in graph_witness_specs(pool, raw_witnesses)
             if any(0 < len(a) < size for a, size in
                    zip(s.allowed, pool.sizes))]
    transitions = propose_joint_actions(pool, specs, incumbent, max_actions=6)
    surrogate = surrogate_view(pool, specs, target, penalty)

    # Offline oracle cache: every unique combo ever drawn is scored once and
    # shared across arms; the full enumeration row is reported separately as
    # the small-pool benchmark and never fed back into any arm.
    cache = {}
    total_space = 1
    for size in pool.sizes:
        total_space *= size

    def score(samples):
        feasible = improving = 0
        for choice in samples:
            if choice not in cache:
                cache[choice] = check_choice(inst, pool.candidates, choice)
            result = cache[choice]
            if result["feasible"]:
                feasible += 1
                if result["makespan"] is not None and result["makespan"] < record["u0"]:
                    improving += 1
        return feasible, improving

    rows = []
    arms = {}
    arms["uniform"] = lambda rng: (arm_uniform(pool.sizes, shots, rng), {})
    arms["sa_energy"] = lambda rng: (arm_sa(surrogate, incumbent, shots, rng),
                                      {"parameter_evaluations": shots,
                                       "training_seconds": 0.0,
                                       "training_shots": 0})
    arms["xy"] = lambda rng: arm_quantum(
        CompactSimulator(pool, specs, (), target, penalty), shots, rng, "xy")
    arms["xy_joint"] = lambda rng: arm_quantum(
        CompactSimulator(pool, specs, transitions, target, penalty), shots,
        rng, "xy_joint")
    arms["xy_joint_warm"] = lambda rng: arm_quantum(
        CompactSimulator(pool, specs, transitions, target, penalty), shots,
        rng, "xy_joint", warm=True)
    arms["mixers_only"] = lambda rng: arm_quantum(
        CompactSimulator(pool, (), transitions, target, penalty), shots, rng,
        "xy_joint")
    arms["xy_joint_shots"] = lambda rng: arm_quantum(
        CompactSimulator(pool, specs, transitions, target, penalty), shots,
        rng, "xy_joint", train_budget=train_budget, shots_training=True)
    for name in ARMS:
        rng = _stable_rng(seed, _arm_index(name), record["seed"],
                          record["trajectory_iteration"])
        tic = time.perf_counter()
        samples, training = arms[name](rng)
        feasible, improving = score(samples)
        rows.append({
            "arm": name, "shots": shots, "raw_p_imp": improving / shots,
            "feasible_rate": feasible / shots, "unique_rate":
                len(set(samples)) / shots,
            "raw_improving_mass": improving, **training,
            "seconds": time.perf_counter() - tic,
        })
    # secondary batch (256) for the two head arms on the same window
    secondary = []
    for name in ("uniform", "xy_joint"):
        rng = _stable_rng(seed + 1, _arm_index(name), record["seed"],
                          record["trajectory_iteration"])
        samples, training = arms[name](rng) if name != "uniform" \
            else (arm_uniform(pool.sizes, secondary_shots, rng), {})
        feasible, improving = score(samples)
        secondary.append({
            "arm": name, "shots": secondary_shots,
            "raw_p_imp": improving / secondary_shots,
            "feasible_rate": feasible / secondary_shots,
            "unique_rate": len(set(samples)) / secondary_shots,
            "raw_improving_mass": improving, **training,
        })
    # exact oracle benchmark (offline, diagnostic only)
    oracle_improving = 0
    oracle_feasible = 0
    for choice in itertools.product(*(range(s) for s in pool.sizes)):
        if choice not in cache:
            cache[choice] = check_choice(inst, pool.candidates, choice)
        result = cache[choice]
        if result["feasible"]:
            oracle_feasible += 1
            if result["makespan"] is not None and result["makespan"] < record["u0"]:
                oracle_improving += 1
    return {
        "window_id": record["snapshot_key"] + "|" + record["strategy"],
        "snapshot_key": record["snapshot_key"],
        "strategy": record["strategy"],
        "instance": record["instance"], "seed": record["seed"],
        "trajectory_iteration": record["trajectory_iteration"],
        "u0": record["u0"], "target_t": target,
        "pool_sizes": list(pool.sizes),
        "total_space": total_space,
        "effective_witnesses": len(specs), "joint_actions": len(transitions),
        "oracle": {"rho": oracle_improving / total_space,
                   "improving": oracle_improving,
                   "feasible": oracle_feasible},
        "arms": rows, "secondary": secondary,
        "graph_evaluations_unique": len(cache),
    }


_ARM_ORDER = {name: i for i, name in enumerate(ARMS)}


def _arm_index(name):
    return _ARM_ORDER[name]


# ---------------------------------------------------------------------------
# Window selection and drivers
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="T06 sampling ablation pilot")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--max-improving", type=int, default=12)
    ap.add_argument("--max-control", type=int, default=6)
    ap.add_argument("--shots", type=int, default=MAIN_SHOTS)
    ap.add_argument("--out", default=str(GATES_DIR /
                                         "results_sampling_ablation_20261003"
                                         / "sampling_ablation.json"))
    args = ap.parse_args(argv)
    windows, snapshots, witness_tables = load_d2_windows(args.results_dir)
    # selection: critical_path snapshots WITH oracle improvement (positive)
    # and iteration-5000 no-improvement snapshots (negative controls)
    evaluations = {}
    eval_path = Path(args.results_dir or GATES_DIR / "results_window_pool_20261003") \
        / "window_pool_d2_dev.json"
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    for e in payload["d2"]["evaluations"]:
        evaluations[(e["instance"], e["seed"], e["trajectory_iteration"],
                     e["strategy"])] = e
    positives, controls = [], []
    seen = set()
    for record in windows:
        if record["strategy"] != "critical_path":
            continue
        key = (record["instance"], record["seed"],
               record["trajectory_iteration"])
        if key in seen:
            continue
        seen.add(key)
        e = evaluations[key + ("critical_path",)]
        if not e["no_improvement"]:
            positives.append(record)
    controls = [r for r in windows if r["strategy"] == "critical_path"
                and r["trajectory_iteration"] == 5000]
    selected = positives[:args.max_improving] + controls[:args.max_control]
    instances = {}
    rows = []
    for record in selected:
        inst = instances.get(record["instance"])
        if inst is None:
            path = ROOT / "task_data" / (
                record["instance"] if record["instance"].endswith(".txt")
                else record["instance"] + ".txt")
            inst = mq.Instance.read(path)
            instances[record["instance"]] = inst
        result = run_window(inst, record, snapshots, witness_tables,
                            shots=args.shots)
        result["split"] = "confirmation" if record["seed"] in (3, 4) \
            else "development"
        rows.append(result)
        print(f"  {record['instance']} s{record['seed']} "
              f"it{record['trajectory_iteration']} "
              f"rho={result['oracle']['rho']:.3f} "
              f"p_imp xy_joint="
              f"{[a['raw_p_imp'] for a in result['arms'] if a['arm'] == 'xy_joint'][0]:.3f}",
              file=sys.stderr)
    summary = summarize(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "work_package": "T06 (Issue #19, PR #18 docs TODO)",
        "shots": args.shots, "train_budget": TRAIN_BUDGET,
        "windows": rows, "summary": summary,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


def summarize(rows):
    """Paired arm vs uniform comparison with a window-level bootstrap."""
    by_arm = {}
    for row in rows:
        if row["oracle"]["improving"] == 0:
            continue          # negative controls reported separately
        for arm in row["arms"]:
            by_arm.setdefault(arm["arm"], []).append(
                (arm["raw_p_imp"], _uniform_p_imp(row)))
    rng = np.random.default_rng(7)
    summary = {}
    for arm, pairs in sorted(by_arm.items()):
        values = np.array([p for p, _u in pairs])
        base = np.array([u for _p, u in pairs])
        ratios = values / base if False else None
        # paired ratio against uniform, zero-guarded per pre-registration:
        # windows where the control is 0 contribute value/base = 0 mass, so
        # the mean ratio is replaced by the paired difference mean and the
        # ratio mean over windows with a nonzero control (both reported).
        nonzero = base > 0
        ratio_mean = float(np.mean(values[nonzero] / base[nonzero])) \
            if nonzero.any() else None
        boot_diff, boot_ratio = [], []
        for _ in range(10000):
            pick = rng.integers(0, len(pairs), len(pairs))
            boot_diff.append(float(np.mean(values[pick] - base[pick])))
            if nonzero[pick].any():
                boot_ratio.append(float(np.mean(
                    values[pick][nonzero[pick]] / base[pick][nonzero[pick]])))
        diff_ci = (float(np.percentile(boot_diff, 2.5)),
                   float(np.percentile(boot_diff, 97.5)))
        ratio_ci = (float(np.percentile(boot_ratio, 2.5)),
                    float(np.percentile(boot_ratio, 97.5))) \
            if boot_ratio else (None, None)
        summary[arm] = {
            "windows": len(pairs),
            "mean_p_imp": float(np.mean(values)),
            "mean_uniform_p_imp": float(np.mean(base)),
            "mean_ratio_vs_uniform_nonzero": ratio_mean,
            "ratio_ci95_nonzero": ratio_ci,
            "mean_diff_vs_uniform": float(np.mean(values - base)),
            "diff_ci95": diff_ci,
            "gate_ratio_110_ci_gt_1": bool(
                ratio_ci[0] is not None and ratio_mean and ratio_mean >= 1.10
                and ratio_ci[0] > 1.0),
            "gate_diff_gt_0": bool(diff_ci[0] > 0.0),
        }
    controls = [row for row in rows if row["oracle"]["improving"] == 0]
    if controls:
        wasted = []
        for row in controls:
            for arm in row["arms"]:
                wasted.append(arm["raw_p_imp"])
        summary["_negative_controls"] = {
            "windows": len(controls),
            "mean_p_imp_all_arms": float(np.mean(wasted)),
            "note": "p_imp=0 windows: any nonzero mass here is wasted call "
                    "cost; arms should stay at 0.",
        }
    return summary


def _uniform_p_imp(row):
    for arm in row["arms"]:
        if arm["arm"] == "uniform":
            return arm["raw_p_imp"]
    raise KeyError("uniform arm missing")


if __name__ == "__main__":
    sys.exit(main())
