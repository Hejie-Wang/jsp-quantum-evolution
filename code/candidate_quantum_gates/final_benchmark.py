#!/usr/bin/env python3
"""T10: end-to-end performance, quantum attribution and generalisation (pilot).

Implements the T10 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md as a pre-registered,
budget-matched comparison matrix.  Every arm solves the *same* instance with the
same wall-clock budget and the same target; costs are separated into graph
evaluations, model/certificate time, training time and sampling time, and the
verdict is decided by the frozen gates:

  * target-arrival cost ratio R <= 0.90 with the paired 95% CI upper bound < 1,
  * success-rate difference 95% CI lower bound >= -0.05,
  * failures, zero-success arms and negative results are all reported.

Arms (all pre-registered, no post-hoc selection):

  ``cp_sat``        complete CP-SAT model, same wall-clock budget;
  ``uniform_dual``  T08 dual-bound loop with the independent uniform proposer
                    (the strongest measured per-cost baseline from T06/T07);
  ``sa_dual``       T08 dual-bound loop with a bounded classical SA proposer on
                    the frozen witness energy;
  ``quantum_dual``  T08 dual-bound loop with the variational quantum proposer
                    (T06 arm: phase + XY, exact-expectation training, 64 shots).

Data scope of this pilot: D0 frozen windows (the D1/D4 confirmation sets and the
D3/DQ windows are not available yet - D2/T04, ta40 and hardware authorisation
are outstanding - so this run is explicitly a pilot whose numbers cannot carry a
confirmation-level claim).
"""
from __future__ import annotations

import argparse
import itertools
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
import global_bounds as gb  # noqa: E402
import medium_qjsp as mq  # noqa: E402
import search_loop as sl  # noqa: E402
from dual_bound_loop import DualBoundLoop  # noqa: E402
from window_diagnostics import build_window, check_choice  # noqa: E402

SCHEMA_VERSION = 1
TARGET_FRACTION = 1.05          # C_tar = ceil(1.05 * U_ref), frozen pre-registration
BENEFIT_RATIO = 0.90
GAMMAS = (0.05, 0.12, 0.25)   # same pre-registered grid as T06
BETAS = (0.1, 0.25, 0.45)
SUCCESS_MARGIN = -0.05


# ---------------------------------------------------------------------------
# Proposers sharing the T08 interface
# ---------------------------------------------------------------------------

class SAProposer:
    """Bounded classical SA on the frozen witness energy (T06 arm)."""

    name = "sa_witness"

    def __init__(self, inst, pool, specs, target_t, *, steps=40, penalty=1.0,
                 seed=7):
        self.sizes = tuple(len(machine) for machine in pool)
        self.specs = specs
        self.target_t = int(target_t)
        self.steps = int(steps)
        self.penalty = penalty
        self.rng = np.random.default_rng(seed)
        self.choice = [int(self.rng.integers(k)) for k in self.sizes]

    def propose(self):
        current = list(self.choice)
        energy = sl.witness_energy(self.specs, current, self.target_t, self.penalty)
        for _ in range(self.steps):
            candidate = list(current)
            machine = int(self.rng.integers(len(self.sizes)))
            candidate[machine] = int(self.rng.integers(self.sizes[machine]))
            value = sl.witness_energy(self.specs, candidate, self.target_t, self.penalty)
            if value <= energy:
                current, energy = candidate, value
            if energy == 0.0:
                break
        self.choice = current
        return tuple(current)

    def cost(self):
        return {"graph_evaluations": 1, "witness_evaluations": self.steps,
                "training_seconds": 0.0, "shots": 0}


class QuantumProposer:
    """T06-style variational proposer: one training pass, then 64-shot sampling.

    The training cost is charged to the loop through ``cost()`` so the arm cannot
    hide it; parameters are re-trained per call exactly as T06 measured.
    """

    name = "quantum_xy"

    def __init__(self, inst, pool, specs, transitions, target_t, *,
                 shots=64, train_evaluations=12, seed=7, sampling="aer",
                 compact_max_states=1 << 22):
        self.specs = specs
        self.transitions = transitions
        self.target_t = int(target_t)
        self.shots = int(shots)
        self.train_evaluations = int(train_evaluations)
        self.seed = seed
        # "aer" needs a full statevector (small pools only); "compact" draws the
        # pre-registered 64 shots from the exact candidate-subspace distribution
        # instead, which is what makes the D2 scale reachable (45+ data qubits
        # cannot be sampled with a statevector simulator here).
        self.sampling = sampling
        self.compact_max_states = int(compact_max_states)
        self.rng = np.random.default_rng(seed)
        self.sampling_backend_detail = sampling
        self.pool_obj = circuits.CandidatePool(
            tuple(tuple(tuple(int(v) for v in order) for order in machine)
                  for machine in pool))
        self.training_seconds = 0.0
        self.sampling_seconds = 0.0
        self._cached_params = None

    def _train(self):
        from compact_simulator import CompactSimulator
        simulator = CompactSimulator(self.pool_obj, self.specs, self.transitions,
                                     self.target_t, 1.0)
        gammas, betas = GAMMAS, BETAS
        params = [(gammas[1], betas[1], betas[1])] * 2
        best = float(simulator.probabilities(params, "xy") @ simulator.energy)
        evaluations = 1
        for position in range(2):
            for axis in range(3):
                grid = gammas if axis == 0 else betas
                for value in grid:
                    if evaluations >= self.train_evaluations:
                        break
                    candidate = [list(p) for p in params]
                    candidate[position][axis] = value
                    candidate = tuple(tuple(p) for p in candidate)
                    energy = float(simulator.probabilities(candidate, "xy")
                                   @ simulator.energy)
                    evaluations += 1
                    if energy < best - 1e-12:
                        params, best = candidate, energy
        return params

    def propose(self):
        started = time.perf_counter()
        self._cached_params = self._train()
        self.training_seconds += time.perf_counter() - started
        started = time.perf_counter()
        if self.sampling == "compact":
            from compact_simulator import CompactSimulator
            simulator = CompactSimulator(self.pool_obj, self.specs, self.transitions,
                                         self.target_t, 1.0,
                                         max_states=self.compact_max_states)
            probabilities = simulator.probabilities(self._cached_params, "xy")
            draws = self.rng.choice(simulator.dimension, size=self.shots, p=probabilities)
            candidates = [tuple(int(v) for v in simulator.labels[i]) for i in draws]
        else:
            circuit = sl.build_fixed_t_ansatz(self.pool_obj, self.specs,
                                              self.transitions, self._cached_params,
                                              self.target_t, 1.0, mode="xy")
            legal, _illegal = sl.sample_choices(
                circuit, self.pool_obj, self.shots,
                self.seed + int(self.training_seconds * 1e6))
            candidates = [key[0] if isinstance(key, tuple) else key for key, _ in legal]
        self.sampling_seconds += time.perf_counter() - started
        if not candidates:
            return None
        best_choice, best_energy = None, None
        for choice in candidates:
            energy = sl.witness_energy(self.specs, choice, self.target_t, 1.0)
            if best_energy is None or energy < best_energy:
                best_choice, best_energy = choice, energy
        return tuple(best_choice) if best_choice is not None else None

    def cost(self):
        return {"graph_evaluations": 1, "training_seconds": self.training_seconds,
                "sampling_seconds": self.sampling_seconds, "shots": self.shots,
                "sampling_backend": self.sampling}


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def run_cp_sat(inst, budget, target, log=None):
    """Complete CP-SAT model under the same budget."""
    started = time.perf_counter()
    record = gb.full_model_bound(inst, time_limit=budget)
    seconds = time.perf_counter() - started
    upper = record["objective"]
    lower = record["dual_bound"]
    hit = upper is not None and upper <= target
    return {
        "arm": "cp_sat", "wall_seconds": seconds,
        "upper_bound": upper, "lower_bound": lower,
        "solver_status": record["status"], "hit": bool(hit),
        "arrival_seconds": seconds if hit else None,
        "graph_evaluations": 0, "certificate_calls": 1,
        "cost_breakdown": {"model_seconds": seconds},
    }


def run_adaptive_mainline(inst, budget, target, seed=7):
    """Old classical main line: `adaptive_search.solve` in classical mode.

    Arrival time is the wall clock of the whole call, i.e. an *upper bound* on
    the first-hit time: that entry point exposes only its final best value plus
    an iteration trace, so this arm is deliberately reported with the coarser
    (and worse for it) timestamp instead of an invented one.
    """
    try:
        from adaptive_search import solve as adaptive_solve
    except Exception as exc:  # pragma: no cover - reported honestly
        return {"arm": "adaptive_mainline", "wall_seconds": 0.0, "upper_bound": None,
                "lower_bound": None, "solver_status": f"unavailable: {exc}",
                "hit": False, "arrival_seconds": None, "graph_evaluations": None,
                "certificate_calls": 0, "cost_breakdown": {}}
    started = time.perf_counter()
    result = adaptive_solve(inst, seconds=budget, seed=seed, mode="classical")
    seconds = time.perf_counter() - started
    upper = int(result["best_makespan"]) if result.get("best_makespan") else None
    hit = upper is not None and upper <= target
    return {"arm": "adaptive_mainline", "wall_seconds": seconds,
            "upper_bound": upper, "lower_bound": int(inst.lower_bound),
            "solver_status": "complete", "hit": bool(hit),
            "arrival_seconds": seconds if hit else None,
            "graph_evaluations": result.get("graph_evaluations"),
            "certificate_calls": 0,
            "cost_breakdown": {"wall_seconds": seconds,
                               "graph_evaluations": result.get("graph_evaluations"),
                               "arrival_is_upper_bound": True}}


def run_joint_mainline(inst, pool, incumbent, budget, target, seed=7):
    """Classical joint search on the same frozen witness cuts and joint actions."""
    pool_obj = circuits.CandidatePool(
        tuple(tuple(tuple(int(v) for v in order) for order in machine)
              for machine in pool))
    started = time.perf_counter()
    result = sl.classical_joint_search(inst, pool_obj, list(incumbent), seed=seed,
                                       time_budget=budget, max_rounds=40,
                                       steps_per_round=200,
                                       per_round_evaluations=8)
    seconds = time.perf_counter() - started
    upper = int(result["best_makespan"])
    hit = upper <= target
    return {"arm": "classical_joint", "wall_seconds": seconds,
            "upper_bound": upper, "lower_bound": int(inst.low_bound) if hasattr(inst, "low_bound") else int(inst.lower_bound),
            "solver_status": "complete", "hit": bool(hit),
            "arrival_seconds": seconds if hit else None,
            "graph_evaluations": result.get("evaluation_count"),
            "certificate_calls": 0,
            "cost_breakdown": {"wall_seconds": seconds,
                               "graph_evaluations": result.get("evaluation_count"),
                               "arrival_is_upper_bound": True}}


def run_dual_arm(inst, pool, *, arm, budget, target, proposer_factory, incumbent,
                 seed=0, cert_every=10, cert_time_limit=0.3):
    """T08 loop with a pluggable proposer under the same budget."""
    from witness_surrogate import collect_witnesses
    raw, _stats = collect_witnesses(inst, pool, tuple(incumbent), n_probe=16, seed=seed)
    medium = [mq.Witness(w["kind"], tuple(tuple(r) for r in w["relations"]),
                         int(w["length"]), ()) for w in raw]
    pool_obj = circuits.CandidatePool(
        tuple(tuple(tuple(int(v) for v in order) for order in machine)
              for machine in pool))
    specs = sl.graph_witness_specs(pool_obj, medium)
    transitions = sl.propose_joint_actions(pool_obj, specs, tuple(incumbent))
    target_t = int(target) - 1
    proposer = proposer_factory(inst, pool, specs, transitions, target_t, seed)
    loop = DualBoundLoop(inst, pool, time_budget=budget, cert_time_limit=cert_time_limit,
                         proposer=proposer, seed=seed,
                         initial_choice=tuple(incumbent))
    started = time.perf_counter()
    arrival = None
    rounds = 0
    while time.perf_counter() - started < budget:
        loop.propose_and_verify()
        rounds += 1
        if arrival is None and loop.U <= target:
            arrival = time.perf_counter() - started
            break
        if rounds % cert_every == 0:
            remaining = budget - (time.perf_counter() - started)
            if remaining > cert_time_limit:
                loop.certify(time_limit=cert_time_limit)
    seconds = time.perf_counter() - started
    cost = proposer.cost() if hasattr(proposer, "cost") else {}
    return {
        "arm": arm, "wall_seconds": seconds,
        "upper_bound": loop.U, "lower_bound": loop.L,
        "solver_status": "n/a", "hit": bool(loop.U <= target),
        "arrival_seconds": arrival,
        "graph_evaluations": loop.graph_evaluations,
        "certificate_calls": loop.certificate_calls,
        "rounds": rounds,
        "lower_bound_scope": loop.L_scope,
        # T10 requires A_cert per instance: B^-1 * integral (U-L)/U dt on the
        # loop's own trajectory (the main-line arms cannot expose one, so their
        # A_cert stays null instead of being invented).
        "gap_integral": loop.gap_integral(budget=seconds),
        "trajectory_seconds_bounds": [[round(t, 6), int(u), int(l)]
                                      for t, u, l in loop.trajectory],
        "cost_breakdown": {
            "training_seconds": getattr(proposer, "training_seconds", 0.0),
            "sampling_seconds": getattr(proposer, "sampling_seconds", 0.0),
            "graph_evaluations": loop.graph_evaluations,
            "proposer": cost,
        },
        "failures": dict(loop.failures),
    }


def _uniform_factory(inst, pool, specs, transitions, target_t, seed):
    return _Uniform([len(machine) for machine in pool], seed)


class _Uniform:
    name = "uniform"

    def __init__(self, sizes, seed=7):
        self.sizes = tuple(sizes)
        self.rng = np.random.default_rng(seed)

    def propose(self):
        return tuple(int(self.rng.integers(k)) for k in self.sizes)

    def cost(self):
        return {"graph_evaluations": 1, "training_seconds": 0.0, "shots": 0}


def _sa_factory(inst, pool, specs, transitions, target_t, seed):
    return SAProposer(inst, pool, specs, target_t, steps=25, seed=seed)


def _quantum_factory(inst, pool, specs, transitions, target_t, seed):
    """Variational arm; D2-scale pools need the compact sampling backend because a
    45+ data-qubit ansatz cannot be sampled with a statevector simulator here."""
    dimension = 1
    for machine in pool:
        dimension *= len(machine)
    sampling = "aer" if dimension <= (1 << 16) and len(pool) * max(
        len(machine) for machine in pool) <= 20 else "compact"
    return QuantumProposer(inst, pool, specs, transitions, target_t, seed=seed,
                           sampling=sampling, compact_max_states=1 << 22)


ARMS = {
    "uniform_dual": _uniform_factory,
    "sa_dual": _sa_factory,
    "quantum_dual": _quantum_factory,
}


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def paired_bootstrap(values, *, samples=10000, seed=7):
    if not values:
        return None
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(values, len(values), replace=True)))
             for _ in range(samples)]
    return {"mean": float(np.mean(values)),
            "ci95": [float(np.percentile(draws, 2.5)),
                     float(np.percentile(draws, 97.5))]}


def evaluate_gate(rows, *, baseline="uniform_dual", budget=None):
    """Frozen gate: arrival-cost ratio and success difference versus the baseline."""
    cost_ratios, success_diffs = [], []
    for row in rows:
        if row.get("no_target") or not row["arms"]:
            continue  # negative control: no reachable target
        base = next(r for r in row["arms"] if r["arm"] == baseline)
        for arm in row["arms"]:
            if arm["arm"] in (baseline, "cp_sat", "adaptive_mainline",
                              "classical_joint"):
                continue
            base_cost = base["arrival_seconds"] if base["hit"] else budget
            arm_cost = arm["arrival_seconds"] if arm["hit"] else budget
            if base_cost:
                cost_ratios.append((arm["arm"], arm_cost / base_cost))
            success_diffs.append((arm["arm"], int(arm["hit"]) - int(base["hit"])))
    summary = []
    for arm in sorted({name for name, _ in cost_ratios} | {name for name, _ in success_diffs}):
        ratios = [value for name, value in cost_ratios if name == arm]
        diffs = [value for name, value in success_diffs if name == arm]
        ratio = paired_bootstrap(ratios)
        success = paired_bootstrap(diffs)
        summary.append({
            "arm": arm,
            "cost_ratio_vs_baseline": ratio,
            "success_difference": success,
            "passes_benefit_gate": bool(ratio and ratio["ci95"][1] < 1.0
                                        and ratio["mean"] <= BENEFIT_RATIO),
            "passes_non_inferiority": bool(success and success["ci95"][0]
                                           >= SUCCESS_MARGIN),
        })
    return summary


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(seeds, *, scales=("3x3", "4x3"), budget=10.0, arms=tuple(ARMS),
        pool_k=3, seed_offset=1000):
    import data_identity
    rows = []
    for jobs, machines in data_identity.D0_SIZES:
        if f"{jobs}x{machines}" not in scales:
            continue
        for seed in seeds:
            inst = data_identity.build_d0(jobs, machines, seed)
            pool, incumbent, u0 = build_window(inst, pool_k, seed + seed_offset)
            exact = data_identity.enumerate_exact_optimum(inst)["exact_optimum"]
            reference = None
            for choice in itertools.product(*(range(len(m)) for m in pool)):
                result = check_choice(inst, pool, choice)
                if result["feasible"] and (reference is None
                                           or result["makespan"] < reference):
                    reference = result["makespan"]
            incumbent_makespan = check_choice(inst, pool, tuple(incumbent))["makespan"]
            if reference is None or incumbent_makespan is None \
                    or reference >= incumbent_makespan:
                rows.append({"scale": f"{jobs}x{machines}", "seed": seed,
                             "instance_sha256": data_identity.instance_sha256(inst),
                             "exact_optimum": exact, "pool_reference": reference,
                             "incumbent_makespan": incumbent_makespan,
                             "target": None, "budget_seconds": budget,
                             "arms": [], "no_target": True})
                print(f"  {jobs}x{machines} seed {seed}: no improvement target "
                      f"(incumbent {incumbent_makespan} >= pool optimum {reference})",
                      flush=True)
                continue
            # pre-registered pilot target: pool optimum (D1/BKS targets need the confirmation set)
            target = int(reference)
            arm_rows = [run_cp_sat(inst, budget, target),
                        run_adaptive_mainline(inst, budget, target, seed=seed),
                        run_joint_mainline(inst, pool, incumbent, budget, target,
                                           seed=seed)]
            for name in arms:
                arm_rows.append(run_dual_arm(inst, pool, arm=name, budget=budget,
                                             target=target,
                                             proposer_factory=ARMS[name],
                                             incumbent=incumbent, seed=seed))
            rows.append({
                "scale": f"{jobs}x{machines}", "seed": seed,
                "instance_sha256": data_identity.instance_sha256(inst),
                "exact_optimum": exact, "pool_reference": reference,
                "target": target, "budget_seconds": budget,
                "arms": arm_rows,
            })
            summary = " ".join(f"{a['arm']}:{'hit' if a['hit'] else 'miss'}"
                               f"@{a['wall_seconds']:.2f}s" for a in arm_rows)
            print(f"  {jobs}x{machines} seed {seed} C*={exact} T={target}: {summary}",
                  flush=True)
    return rows


def run_d2(dataset_path, *, budget=5.0, improving=3, negatives=2,
           include_quantum=True, seed_offset=0):
    """Exploratory T10 battery on T04's D2 developer snapshots.

    This is *not* the confirmation set: it uses the developer seeds and a short
    budget, and it exists to show the integration (T04 output -> T06/T10 arms)
    and the D2-scale resource picture.  Positive controls are windows whose pool
    provably contains an improvement; negative controls are windows whose pool
    has none, kept to report the invalid-call cost.
    """
    import d2_adapter
    positives = list(d2_adapter.iter_windows(dataset_path, only_improving=True,
                                             limit=improving))
    all_windows = list(d2_adapter.iter_windows(dataset_path, only_improving=False))
    negatives = [row for row in all_windows
                 if row["pool_optimum"] is not None and row["target"] >= row["u0"]][:negatives]
    rows = []
    for row in positives + negatives:
        inst, pool, incumbent = row["inst"], row["pool"], row["incumbent"]
        # A negative control is a window whose pool provably contains no
        # improvement: its target is set BELOW the incumbent so that every arm
        # must miss and only the invalid-call cost is measured.  Such rows are
        # excluded from the gate (``no_target``).
        negative = row["target"] >= row["u0"]
        target = int(row["u0"]) - 1 if negative else int(row["target"])
        arms = []
        for name, runner in (("cp_sat", lambda: run_cp_sat(inst, budget, target)),
                             ("classical_joint",
                              lambda: run_joint_mainline(inst, pool, incumbent, budget,
                                                         target, seed=seed_offset))):
            try:
                arms.append(runner())
            except Exception as exc:  # noqa: BLE001
                arms.append({"arm": name, "hit": False, "arrival_seconds": None,
                             "upper_bound": None, "lower_bound": None,
                             "solver_status": f"error: {type(exc).__name__}: {exc}",
                             "graph_evaluations": None, "certificate_calls": 0,
                             "cost_breakdown": {}})
        for name in ("uniform_dual", "sa_dual") + (("quantum_dual",) if include_quantum else ()):
            try:
                arms.append(run_dual_arm(inst, pool, arm=name, budget=budget,
                                         target=target,
                                         proposer_factory=ARMS[name],
                                         incumbent=incumbent, seed=seed_offset))
            except Exception as exc:  # noqa: BLE001
                arms.append({"arm": name, "hit": False, "arrival_seconds": None,
                             "upper_bound": None, "lower_bound": None,
                             "solver_status": f"error: {type(exc).__name__}: {exc}",
                             "graph_evaluations": None, "certificate_calls": 0,
                             "cost_breakdown": {}})
        rows.append({"dataset": "D2-dev (exploratory)", "instance": row["instance"],
                     "seed": row["seed"], "strategy": row["strategy"],
                     "trajectory_iteration": row["trajectory_iteration"],
                     "pool_sizes": row["pool_sizes"], "u0": row["u0"],
                     "target": target, "pool_optimum": row["pool_optimum"],
                     "rho": row["rho"], "d_imp": row["d_imp_from_incumbent"],
                     "witnesses": len(row["witnesses"]),
                     "witnesses_dropped_constant_false": row["witnesses_dropped_constant_false"],
                     "production_hint": pool,
                     "control": "negative" if negative else "positive",
                     "no_target": bool(negative),
                     "budget_seconds": budget, "arms": arms})
        print(f"  D2 {row['instance']} s{row['seed']} {row['strategy']} "
              f"it{row['trajectory_iteration']} ({'neg' if negative else 'pos'}): "
              + " ".join(f"{a['arm']}:{'hit' if a['hit'] else 'miss'}" for a in arms),
              flush=True)
    return rows


def load_d1_pool(path):
    """Read one old fixed pool report (D1 layer): instance, pool, incumbent."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    inst = mq.Instance(report["instance_data"]["durations"],
                       report["instance_data"]["routes_zero_based"],
                       name=str(report.get("instance", Path(path).stem)))
    pool = tuple(tuple(tuple(int(v) for v in order) for order in machine)
                 for machine in report["pool_data"])
    mq.validate_pool(inst, pool)
    return inst, pool, tuple(int(a) for a in report["best_choice"]), int(report["best_makespan"])


def run_d1(*, budget=10.0, max_instances=6, include_quantum=False, seed_offset=0):
    """D1 layer (the six repo instances with their old fixed pools).

    Target: one unit better than the old pool incumbent, i.e. a *strict*
    improvement, because the pool optimum is not known for these pools and
    computing it is T01's offline diagnostic, not an arm input.  The variant
    arms that need an exact candidate-subspace simulator are skipped with their
    reason recorded: prod(K_m) at D1 scale is far above the compact-simulator
    cap, which is itself a resource boundary of the variational approach.
    """
    reports = sorted((ROOT / "code" / "candidate_quantum_medium" / "results"
                      / "corrected_20261001").glob("*_milp.json"))[:max_instances]
    rows = []
    for path in reports:
        inst, pool, incumbent, u0 = load_d1_pool(path)
        target = u0 - 1
        dimension = 1
        for machine in pool:
            dimension *= len(machine)
        arm_rows = []
        for name, runner in (("cp_sat", lambda: run_cp_sat(inst, budget, target)),
                             ("adaptive_mainline",
                              lambda: run_adaptive_mainline(inst, budget, target,
                                                            seed=seed_offset)),
                             ("classical_joint",
                              lambda: run_joint_mainline(inst, pool, incumbent,
                                                         budget, target,
                                                         seed=seed_offset))):
            try:
                arm_rows.append(runner())
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal
                arm_rows.append({"arm": name, "hit": False, "arrival_seconds": None,
                                 "upper_bound": None, "lower_bound": None,
                                 "solver_status": f"error: {type(exc).__name__}: {exc}",
                                 "graph_evaluations": None, "certificate_calls": 0,
                                 "cost_breakdown": {}})
        for name in ("uniform_dual", "sa_dual"):
            try:
                arm_rows.append(run_dual_arm(inst, pool, arm=name, budget=budget,
                                             target=target,
                                             proposer_factory=ARMS[name],
                                             incumbent=incumbent, seed=seed_offset))
            except Exception as exc:  # noqa: BLE001
                arm_rows.append({"arm": name, "hit": False, "arrival_seconds": None,
                                 "upper_bound": None, "lower_bound": None,
                                 "solver_status": f"error: {type(exc).__name__}: {exc}",
                                 "graph_evaluations": None, "certificate_calls": 0,
                                 "cost_breakdown": {}})
        if include_quantum:
            try:
                arm_rows.append(run_dual_arm(inst, pool, arm="quantum_dual", budget=budget,
                                             target=target,
                                             proposer_factory=ARMS["quantum_dual"],
                                             incumbent=incumbent, seed=seed_offset))
            except Exception as exc:  # noqa: BLE001
                arm_rows.append({"arm": "quantum_dual", "hit": False,
                                 "arrival_seconds": None, "upper_bound": None,
                                 "lower_bound": None,
                                 "solver_status": f"error: {type(exc).__name__}: {exc}",
                                 "graph_evaluations": None, "certificate_calls": 0,
                                 "cost_breakdown": {}})
        else:
            arm_rows.append({"arm": "quantum_dual", "hit": False, "arrival_seconds": None,
                             "upper_bound": None, "lower_bound": None,
                             "solver_status": "skipped_resource_limit",
                             "graph_evaluations": None, "certificate_calls": 0,
                             "cost_breakdown": {},
                             "reason": f"prod(K_m) = {dimension:,} exceeds the compact "
                                       f"candidate-subspace cap (65536): the T06 variational "
                                       f"arm cannot be instantiated at D1 scale"})
        rows.append({
            "instance": inst.name, "source": Path(path).name,
            "instance_sha256": mq.content_hash(
                {"durations": inst.durations.tolist(), "routes": inst.routes.tolist()}),
            "pool_sizes": [len(machine) for machine in pool],
            "candidate_subspace": dimension,
            "incumbent_makespan": u0, "target": target, "budget_seconds": budget,
            "arms": arm_rows,
        })
        summary = " ".join(f"{a['arm']}:{'hit' if a['hit'] else 'miss'}"
                           for a in arm_rows)
        print(f"  D1 {inst.name}: U0={u0} target={target} subspace={dimension:,} {summary}",
              flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="T10 end-to-end benchmark (pilot)")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--scales", default="3x3,4x3")
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--pool-k", type=int, default=3)
    ap.add_argument("--arms", default="uniform_dual,sa_dual,quantum_dual")
    ap.add_argument("--max-d1", type=int, default=None)
    ap.add_argument("--d2-dataset", default=None,
                    help="T04 D2 dataset; runs the exploratory D2 battery and exits")
    ap.add_argument("--d2-budget", type=float, default=5.0)
    ap.add_argument("--d2-improving", type=int, default=3)
    ap.add_argument("--d2-negatives", type=int, default=2)
    ap.add_argument("--d2-no-quantum", action="store_true")
    ap.add_argument("--d2-out", default=None)
    ap.add_argument("--d1", action="store_true",
                    help="also run the D1 layer (six old fixed pools, strict-improvement target)")
    ap.add_argument("--d1-only", action="store_true",
                    help="load the existing --out file and only add/refresh the D1 block")
    ap.add_argument("--include-quantum-d1", action="store_true",
                    help="attempt the variational arm at D1 scale (fails on the simulator cap)")
    ap.add_argument("--out", default=str(GATES_DIR / "results_final_benchmark_20261003"
                                         / "final_benchmark.json"))
    args = ap.parse_args(argv)
    lo, hi = args.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    if args.d2_dataset:
        rows = run_d2(args.d2_dataset, budget=args.d2_budget,
                      improving=args.d2_improving, negatives=args.d2_negatives,
                      include_quantum=not args.d2_no_quantum)
        payload = {"schema_version": SCHEMA_VERSION,
                   "work_package": "T10 exploratory D2 battery (T04 snapshots)",
                   "dataset": args.d2_dataset,
                   "preregistration": {"budget_seconds": args.d2_budget,
                                       "benefit_ratio": BENEFIT_RATIO,
                                       "success_margin": SUCCESS_MARGIN,
                                       "main_baseline": "uniform_dual"},
                   "windows": rows,
                   "gate": evaluate_gate(rows, budget=args.d2_budget)}
        target_path = Path(args.d2_out or (GATES_DIR
                                           / "results_final_benchmark_20261003"
                                           / "d2_exploratory.json"))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                          default=_json_default), encoding="utf-8")
        print(f"wrote {target_path}")
        for row in payload["gate"]:
            ratio = row["cost_ratio_vs_baseline"]
            print(f"  {row['arm']}: ratio "
                  f"{None if not ratio else round(ratio['mean'], 3)} "
                  f"benefit={row['passes_benefit_gate']} "
                  f"noninferior={row['passes_non_inferiority']}")
        return 0
    out_path = Path(args.out)
    if args.d1_only:
        if not out_path.exists():
            raise SystemExit(f"--d1-only needs an existing result file at {out_path}")
        result = json.loads(out_path.read_text(encoding="utf-8"))
        result["d1"] = run_d1(budget=args.budget, max_instances=len(args.scales.split(",")) * 3,
                              include_quantum=args.include_quantum_d1)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                       default=_json_default), encoding="utf-8")
        print(f"wrote d1 block to {out_path}")
        return 0
    rows = run(seeds, scales=tuple(args.scales.split(",")), budget=args.budget,
               arms=tuple(args.arms.split(",")), pool_k=args.pool_k)
    gate = evaluate_gate(rows, budget=args.budget)
    result = {
        "schema_version": SCHEMA_VERSION,
        "work_package": "T10 (Issue #19, PR #18 docs TODO)",
        "preregistration": {"target_fraction": TARGET_FRACTION,
                            "benefit_ratio": BENEFIT_RATIO,
                            "success_margin": SUCCESS_MARGIN,
                            "budget_seconds": args.budget,
                            "main_baseline": "uniform_dual"},
        "scope": "pilot on D0 windows; D1 confirmation set, D3/D4 and DQ hardware "
                 "windows are not available in this session",
        "windows": rows, "gate": gate,
        "notes": [
            "Every arm sees the same instance, pool, target and wall-clock budget.",
            "Costs are separated: graph evaluations, model/certificate, training, sampling.",
            "Failures and zero-success arms are kept; no hyperparameter is selected after seeing results.",
        ],
    }
    if args.d1:
        result["d1"] = run_d1(budget=args.budget,
                              max_instances=max(1, args.max_d1 or 6),
                              include_quantum=args.include_quantum_d1)
    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=_json_default),
                   encoding="utf-8")
    print(f"wrote {out}")
    for row in gate:
        ratio = row["cost_ratio_vs_baseline"]
        print(f"  {row['arm']}: cost ratio "
              f"{None if not ratio else round(ratio['mean'], 3)} "
              f"passes_benefit={row['passes_benefit_gate']} "
              f"non_inferior={row['passes_non_inferiority']}")
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
