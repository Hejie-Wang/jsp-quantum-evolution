#!/usr/bin/env python3
"""T07: quantum call policy, parameter reuse and budget allocation.

Implements the T07 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md as a *measured-cost* policy
study on the frozen windows of T06 (Issue #19).  Every number a policy sees is
either measured on the machine (classical sample cost, quantum call cost) or
taken from the T06 arm measurements (hit probability per arm); policies never
look at future outcomes or at the verification answers.

Actions a policy can take (all costed in wall seconds, all logged):

  ``classical``       one independent uniform sample plus its full graph check;
  ``quantum``         one call of the trained variational module: the measured
                      training time of the chosen arm (T06) plus one 64-shot
                      sampling round plus the graph checks of its samples.

Policies compared (the work package asks for exactly these three plus a
no-quantum strong classical baseline):

  ``fixed_frequency`` every 40 classical trials triggers one quantum call;
  ``stagnation``      a quantum call after ``k`` consecutive trials without a hit;
  ``adaptive``        epsilon-greedy on the *observed* hits-per-second posterior
                      of both actions, with automatic disabling of a module whose
                      upper credible bound stays below the classical action;
  ``disabled``        classical only (strong classical reference).

Reported per window and policy: truncated arrival time to the target, success
rate, quantum time share, total cost, and the T07 non-inferiority test
(quantum-module time reduced by >= 50% while the success-rate difference has a
95% CI lower bound >= -0.05), plus whether the end-to-end arrival time passes
the common benefit gate (paired ratio <= 0.90 with CI upper bound < 1).  The
verdict is reported as measured, including "no benefit".
"""
from __future__ import annotations

import argparse
import json
import math
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
import search_loop as sl  # noqa: E402
import sampling_ablation as sa  # noqa: E402
from window_diagnostics import build_window, check_choice  # noqa: E402

SCHEMA_VERSION = 1
FIXED_PERIOD = 40
STAGNATION_K = 5
EPSILON = 0.1
SHOTS_PER_CALL = 64
REPETITIONS = 200
BUDGET_SECONDS = 60.0
NONINFERIORITY_TIME_REDUCTION = 0.50
NONINFERIORITY_SUCCESS_MARGIN = -0.05
BENEFIT_RATIO = 0.90


# ---------------------------------------------------------------------------
# Measured costs of the two actions
# ---------------------------------------------------------------------------

def measured_costs(inst, pool, *, repeats=300, seed=7):
    """Wall time of one classical action (uniform sample + full graph check)."""
    sizes = [len(machine) for machine in pool]
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for _ in range(repeats):
        choice = tuple(int(rng.integers(k)) for k in sizes)
        check_choice(inst, pool, choice)
    return (time.perf_counter() - started) / repeats


def measure_sampling_seconds(inst, pool, target_t, *, shots=SHOTS_PER_CALL,
                             repeats=3, seed=11):
    """Wall time of one 64-shot sampling round of the trained-form circuit."""
    from witness_surrogate import collect_witnesses
    witnesses, _stats = collect_witnesses(inst, pool, tuple(0 for _ in pool),
                                          n_probe=16, seed=0)
    medium = [mq.Witness(w["kind"], tuple(tuple(r) for r in w["relations"]),
                         int(w["length"]), ()) for w in witnesses]
    pool_obj = circuits.CandidatePool(
        tuple(tuple(tuple(int(v) for v in order) for order in machine)
              for machine in pool))
    specs = sl.graph_witness_specs(pool_obj, medium)
    transitions = sl.propose_joint_actions(pool_obj, specs, tuple(0 for _ in pool))
    params = ((0.05, 0.25, 0.1), (0.05, 0.25, 0.25))
    circuit = sl.build_fixed_t_ansatz(pool_obj, specs, transitions, params,
                                      target_t, 1.0, mode="xy_joint")
    times = []
    for repeat in range(repeats):
        started = time.perf_counter()
        sl.sample_choices(circuit, pool_obj, shots, seed + repeat)
        times.append(time.perf_counter() - started)
    return {"sampling_seconds": float(np.mean(times)),
            "graph_checks": shots, "shots": shots}


# ---------------------------------------------------------------------------
# Policy simulation (no future information)
# ---------------------------------------------------------------------------

class Policy:
    """Base policy: decide the next action from the observed history only."""

    name = "base"

    def __init__(self, classical_seconds, quantum_seconds, p_classical, p_quantum):
        self.c_classical = classical_seconds
        self.c_quantum = quantum_seconds
        self.p_classical = p_classical
        self.p_quantum = p_quantum
        self.history = []
        self.quantum_calls = 0
        self.classical_trials = 0
        self.quantum_seconds = 0.0
        self.classical_seconds = 0.0
        self.disabled = False
        self.disable_round = None

    def next_action(self, round_index, stagnation) -> str:  # pragma: no cover
        raise NotImplementedError

    def record(self, action, hit):
        self.history.append((action, hit))
        if action == "quantum":
            self.quantum_calls += 1
            self.quantum_seconds += self.c_quantum
        else:
            self.classical_trials += 1
            self.classical_seconds += self.c_classical

    def summary(self):
        total = self.quantum_seconds + self.classical_seconds
        return {"policy": self.name, "quantum_calls": self.quantum_calls,
                "classical_trials": self.classical_trials,
                "quantum_seconds": self.quantum_seconds,
                "classical_seconds": self.classical_seconds,
                "total_seconds": total,
                "quantum_time_share": (self.quantum_seconds / total) if total else 0.0,
                "disabled": self.disabled, "disable_round": self.disable_round}


class FixedFrequencyPolicy(Policy):
    name = "fixed_frequency"

    def next_action(self, round_index, stagnation):
        return "quantum" if round_index % FIXED_PERIOD == FIXED_PERIOD - 1 else "classical"


class StagnationPolicy(Policy):
    name = "stagnation"

    def next_action(self, round_index, stagnation):
        return "quantum" if stagnation >= STAGNATION_K else "classical"


class AdaptivePolicy(Policy):
    """epsilon-greedy on observed hits per second, with automatic disabling.

    Posterior: Beta(1 + hits, 1 + misses) per action; efficiency = posterior mean
    of the hit probability divided by the action's cost.  Exploration is spent
    with probability EPSILON and stops for an action whose upper 95% credible
    bound on efficiency is below the other action's posterior mean after at
    least 20 observations - that is the "disable a long-useless module" rule.
    """

    name = "adaptive"

    def __init__(self, *args, epsilon=EPSILON, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.stats = {"quantum": [1, 1], "classical": [1, 1]}  # hits+1, total+1
        self.rng = np.random.default_rng(7)

    def _efficiency(self, action):
        hits, total = self.stats[action]
        mean = hits / total
        cost = self.c_quantum if action == "quantum" else self.c_classical
        return mean / cost, (hits + 1.0) / total / cost

    def next_action(self, round_index, stagnation):
        if self.disabled:
            return "classical"
        if self.stats["quantum"][1] >= 20:
            _, upper = self._efficiency("quantum")
            classical_mean, _ = self._efficiency("classical")
            if upper < classical_mean:
                self.disabled = True
                self.disable_round = round_index
                return "classical"
        if self.rng.random() < self.epsilon:
            return "quantum" if self.stats["quantum"][1] <= self.stats["classical"][1] else "classical"
        quantum_mean, _ = self._efficiency("quantum")
        classical_mean, _ = self._efficiency("classical")
        return "quantum" if quantum_mean > classical_mean else "classical"

    def record(self, action, hit):
        self.stats[action][0] += int(hit)
        self.stats[action][1] += 1
        super().record(action, hit)


class DisabledPolicy(Policy):
    name = "disabled"

    def next_action(self, round_index, stagnation):
        return "classical"


POLICIES = (FixedFrequencyPolicy, StagnationPolicy, AdaptivePolicy, DisabledPolicy)


# ---------------------------------------------------------------------------
# One policy run
# ---------------------------------------------------------------------------

def simulate(policy, p_classical, p_quantum, *, budget=BUDGET_SECONDS, seed=0):
    """Run one policy to the first hit or the budget; returns arrival + cost."""
    rng = np.random.default_rng(seed)
    stagnation = 0
    elapsed = 0.0
    hits = 0
    round_index = 0
    while elapsed < budget:
        action = policy.next_action(round_index, stagnation)
        cost = policy.c_quantum if action == "quantum" else policy.c_classical
        if action == "quantum":
            hit = rng.random() < p_quantum
        else:
            hit = rng.random() < p_classical
        policy.record(action, hit)
        elapsed += cost
        stagnation = 0 if hit else stagnation + 1
        hits += int(hit)
        round_index += 1
        if hit:
            break
    return {"arrival_seconds": elapsed if hits else None,
            "elapsed_seconds": elapsed, "hit": bool(hits),
            "rounds": round_index, **policy.summary()}


def evaluate_policy(policy_cls, p_classical, p_quantum, classical_seconds,
                    quantum_seconds, *, repetitions=REPETITIONS,
                    budget=BUDGET_SECONDS, seed=0):
    """Run `repetitions` trials inside ONE policy session.

    The policy keeps its history across trials, because a deployed algorithm
    does not forget what it learned after each improvement round; this is what
    lets the adaptive rule actually disable a long-useless module.
    """
    policy = policy_cls(classical_seconds, quantum_seconds, p_classical, p_quantum)
    rows = []
    disable_index = None
    for repeat in range(repetitions):
        before = (policy.quantum_seconds, policy.classical_seconds,
                  policy.quantum_calls, policy.classical_trials)
        result = simulate(policy, p_classical, p_quantum, budget=budget,
                          seed=seed + repeat)
        result["trial_quantum_seconds"] = policy.quantum_seconds - before[0]
        result["trial_classical_seconds"] = policy.classical_seconds - before[1]
        result["trial_quantum_calls"] = policy.quantum_calls - before[2]
        result["trial_classical_trials"] = policy.classical_trials - before[3]
        trial_total = (result["trial_quantum_seconds"]
                       + result["trial_classical_seconds"])
        result["trial_quantum_share"] = ((result["trial_quantum_seconds"] / trial_total)
                                         if trial_total else 0.0)
        if policy.disabled and disable_index is None:
            disable_index = repeat
        rows.append(result)
    arrivals = [r["arrival_seconds"] for r in rows if r["hit"]]
    after_disable = rows[disable_index + 1:] if disable_index is not None else []
    return {
        "policy": policy_cls.name,
        "repetitions": repetitions,
        "success_rate": len(arrivals) / repetitions,
        "mean_arrival_seconds": float(np.mean(arrivals)) if arrivals else None,
        "truncated_arrival_seconds": float(np.mean(
            [r["arrival_seconds"] if r["hit"] else budget for r in rows])),
        "mean_quantum_share": float(np.mean([r["trial_quantum_share"] for r in rows])),
        "cumulative_quantum_share": policy.summary()["quantum_time_share"],
        "mean_quantum_seconds": float(np.mean([r["quantum_seconds"] for r in rows])),
        "mean_total_seconds": float(np.mean([r["total_seconds"] for r in rows])),
        "disabled": policy.disabled,
        "disabled_at_trial": disable_index,
        "quantum_share_after_disable": (
            float(np.mean([r["trial_quantum_share"] for r in after_disable]))
            if after_disable else None),
        "mean_quantum_calls": float(np.mean([r["quantum_calls"] for r in rows])),
        "mean_classical_trials": float(np.mean([r["classical_trials"] for r in rows])),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_t06_arm_probabilities(path, arm="xy_joint", training="exact"):
    """Hit probabilities measured by T06, per window (scale, seed)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    probabilities = {}
    for row in data["windows"]:
        key = (row["window"]["scale"], row["window"]["seed"])
        entry = {"p_classical": None, "p_quantum": None, "training_seconds": 0.0}
        for arm_row in row["arms"]:
            if arm_row["arm"] == "uniform":
                entry["p_classical"] = arm_row["p_hit"]
            if arm_row["arm"] == arm and arm_row.get("training") == training:
                entry["p_quantum"] = arm_row["p_hit"]
                entry["training_seconds"] = arm_row["training_seconds"]
        probabilities[key] = entry
    return probabilities


def paired_bootstrap(ratios, *, samples=10000, seed=7):
    rng = np.random.default_rng(seed)
    if not ratios:
        return None
    draws = [float(np.mean(rng.choice(ratios, len(ratios), replace=True)))
             for _ in range(samples)]
    return {"mean": float(np.mean(ratios)),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def run(t06_path, *, repetitions=REPETITIONS, budget=BUDGET_SECONDS,
        shots=SHOTS_PER_CALL, quantum_arm="xy_joint", training="exact",
        scales=("3x3", "4x3"), max_windows=None):
    import data_identity
    probabilities = load_t06_arm_probabilities(t06_path, arm=quantum_arm,
                                               training=training)
    windows = []
    for (scale, seed), entry in sorted(probabilities.items()):
        if scale not in scales:
            continue
        jobs, machines = (int(v) for v in scale.split("x"))
        inst = data_identity.build_d0(jobs, machines, seed)
        pool, incumbent, u0 = build_window(inst, 3, seed + 1000)
        reference = sa.exact_reference(inst, pool, u0, u0 - 1)
        if reference["improving_combinations"] == 0:
            continue  # negative controls have no target to arrive at
        sampling = measure_sampling_seconds(inst, pool, u0 - 1, shots=shots)
        classical = measured_costs(inst, pool)
        quantum_seconds = entry["training_seconds"] + sampling["sampling_seconds"]
        rows = [evaluate_policy(cls, entry["p_classical"], entry["p_quantum"],
                                classical, quantum_seconds, repetitions=repetitions,
                                budget=budget, seed=seed)
                for cls in POLICIES]
        windows.append({
            "scale": scale, "seed": seed, "u0": int(u0),
            "p_classical": entry["p_classical"], "p_quantum": entry["p_quantum"],
            "classical_seconds_per_trial": classical,
            "quantum_seconds_per_call": quantum_seconds,
            "quantum_call_cost_breakdown": {
                "training_seconds": entry["training_seconds"],
                "sampling_seconds": sampling["sampling_seconds"],
                "graph_checks_per_call": sampling["graph_checks"],
            },
            "policies": rows,
        })
        print(f"  {scale} seed {seed}: p_c={entry['p_classical']:.4f} "
              f"p_q={entry['p_quantum']:.4f} c_c={classical:.2e}s "
              f"c_q={quantum_seconds:.2e}s", flush=True)
        if max_windows and len(windows) >= max_windows:
            break

    by_policy = {}
    for window in windows:
        for row in window["policies"]:
            by_policy.setdefault(row["policy"], []).append((window, row))
    summary = []
    for policy, entries in sorted(by_policy.items()):
        ratios, quantum_share, reductions = [], [], []
        success_diffs = []
        for window, row in entries:
            baseline = next(r for r in window["policies"] if r["policy"] == "disabled")
            fixed = next(r for r in window["policies"] if r["policy"] == "fixed_frequency")
            if baseline["truncated_arrival_seconds"]:
                ratios.append(row["truncated_arrival_seconds"]
                              / baseline["truncated_arrival_seconds"])
            quantum_share.append(row["mean_quantum_share"])
            success_diffs.append(row["success_rate"] - baseline["success_rate"])
            if fixed["mean_quantum_seconds"] > 0:
                reductions.append(1.0 - row["mean_quantum_seconds"]
                                  / fixed["mean_quantum_seconds"])
        bootstrap = paired_bootstrap(ratios)
        success_ci = paired_bootstrap(success_diffs)["ci95"] if success_diffs else None
        time_reduction = float(np.mean(reductions)) if reductions else None
        summary.append({
            "policy": policy,
            "mean_quantum_share": float(np.mean(quantum_share)),
            "paired_ratio_vs_classical": bootstrap,
            "success_rate_difference": {
                "mean": float(np.mean(success_diffs)),
                "ci95": success_ci,
            },
            "quantum_seconds_reduction_vs_fixed_frequency": time_reduction,
            "passes_benefit_gate": bool(bootstrap and bootstrap["ci95"][1] < 1.0
                                        and bootstrap["mean"] <= BENEFIT_RATIO),
            "non_inferiority": {
                "time_reduction_required": NONINFERIORITY_TIME_REDUCTION,
                "success_margin": NONINFERIORITY_SUCCESS_MARGIN,
                "measured_time_reduction": time_reduction,
                "passed": bool(time_reduction is not None
                               and time_reduction >= NONINFERIORITY_TIME_REDUCTION
                               and success_ci is not None
                               and success_ci[0] >= NONINFERIORITY_SUCCESS_MARGIN),
            },
        })
    return {"windows": windows, "summary": summary,
            "protocol": {"fixed_period": FIXED_PERIOD, "stagnation_k": STAGNATION_K,
                         "epsilon": EPSILON, "shots_per_call": shots,
                         "repetitions": repetitions, "budget_seconds": budget,
                         "quantum_arm": quantum_arm, "training": training,
                         "benefit_ratio": BENEFIT_RATIO}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="T07 quantum call-policy study")
    ap.add_argument("--t06", default=str(GATES_DIR / "results_sampling_ablation_20261003"
                                         / "sampling_ablation.json"))
    ap.add_argument("--repetitions", type=int, default=REPETITIONS)
    ap.add_argument("--budget", type=float, default=BUDGET_SECONDS)
    ap.add_argument("--shots", type=int, default=SHOTS_PER_CALL)
    ap.add_argument("--quantum-arm", default="xy_joint")
    ap.add_argument("--training", default="exact")
    ap.add_argument("--scales", default="3x3,4x3")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--out", default=str(GATES_DIR / "results_call_policy_20261003"
                                         / "call_policy.json"))
    args = ap.parse_args(argv)
    result = {"schema_version": SCHEMA_VERSION,
              "work_package": "T07 (Issue #19, PR #18 docs TODO)",
              "measured_from": str(args.t06)}
    result.update(run(args.t06, repetitions=args.repetitions, budget=args.budget,
                      shots=args.shots, quantum_arm=args.quantum_arm,
                      training=args.training,
                      scales=tuple(args.scales.split(",")),
                      max_windows=args.max_windows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=_json_default),
                   encoding="utf-8")
    print(f"wrote {out}")
    for row in result["summary"]:
        print(f"  {row['policy']}: quantum share {row['mean_quantum_share']:.3f} "
              f"ratio {None if not row['paired_ratio_vs_classical'] else round(row['paired_ratio_vs_classical']['mean'], 3)} "
              f"benefit={row['passes_benefit_gate']} noninferior={row['non_inferiority']['passed']}")
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
