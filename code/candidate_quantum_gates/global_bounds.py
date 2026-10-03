#!/usr/bin/env python3
"""T02: certified global lower bounds and threshold infeasibility (Issue #19).

Implements the T02 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md on top of the frozen T00
data layer (data_identity.py).  Four certified bound sources are provided,
all global-scope for the ORIGINAL JSP (no pool, no window):

  trivial_bound          L0 = max(max job load, max machine load).
  subset_relaxation      For a machine subset S, the relaxation R_S keeps all
                         job chains and processing times but enforces
                         non-overlap only on machines in S.  Its exact
                         optimum (CP-SAT, status OPTIMAL) is a valid lower
                         bound on C*.  Bounds merge by max, never by sum.
  threshold_infeasible   A feasibility model with makespan <= T.  ONLY the
                         explicit INFEASIBLE status may raise the global
                         lower bound to T+1.  UNKNOWN, timeouts and any
                         non-proof status never raise L.
  solver_dual            The full model's CP-SAT dual bound (best_objective_
                         bound), recorded with its status for reference.

Every bound is appended to bound_records.jsonl with its scope, parameters,
solver status and wall time, so that "certified" and "heuristic" numbers can
never be conflated later.  D0 acceptance (threshold verdicts consistent with
brute-force enumeration, L <= C* <= U, monotone relaxation) is checked in
test_global_bounds.py; the runner reproduces D1 known bounds as a reference
without claiming new public lower bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATES_DIR = Path(__file__).resolve().parent
MEDIUM_DIR = ROOT / "code" / "candidate_quantum_medium"
for _p in (str(GATES_DIR), str(MEDIUM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import medium_qjsp as mq  # noqa: E402
from ortools.sat.python import cp_model  # noqa: E402
import ortools  # noqa: E402

SCHEMA_VERSION = 1
ORTOOLS_VERSION = getattr(ortools, "__version__", "unknown")


# ---------------------------------------------------------------------------
# CP-SAT models
# ---------------------------------------------------------------------------

def _build_model(inst: mq.Instance, machine_subset=None, threshold=None,
                 minimize=True):
    horizon = int(inst.total_duration)  # serial schedule always fits
    model = cp_model.CpModel()
    starts = [model.NewIntVar(0, horizon - int(inst.processing[v]),
                              f"s_{v}")
              for v in range(inst.operations)]
    makespan = model.NewIntVar(0, horizon, "C")
    for u, v in inst.job_arcs:
        model.Add(starts[v] >= starts[u] + inst.processing[u])
    for v in range(inst.operations):
        model.Add(makespan >= starts[v] + inst.processing[v])
    for machine in (inst.groups if machine_subset is None
                    else [inst.groups[m] for m in machine_subset]):
        for a, b in combinations(machine, 2):
            before = model.NewBoolVar(f"{a}_before_{b}")
            model.Add(starts[b] >= starts[a] + inst.processing[a]) \
                .OnlyEnforceIf(before)
            model.Add(starts[a] >= starts[b] + inst.processing[b]) \
                .OnlyEnforceIf(before.Not())
    if threshold is not None:
        model.Add(makespan <= int(threshold))
    if minimize:
        model.Minimize(makespan)
    return model, makespan


def _solve(model, time_limit):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    t0 = time.perf_counter()
    status = solver.Solve(model)
    wall = time.perf_counter() - t0
    name = solver.StatusName(status)
    out = {
        "status": name,
        "wall_seconds": wall,
        "solver_version": getattr(solver, "solver_version", None),
    }
    if name in ("OPTIMAL", "FEASIBLE"):
        out["objective"] = int(round(solver.ObjectiveValue()))
        out["dual_bound"] = int(round(solver.BestObjectiveBound()))
    elif name == "INFEASIBLE":
        out["objective"] = None
        out["dual_bound"] = None
    else:  # UNKNOWN / MODEL_INVALID etc. — certifies nothing
        out["objective"] = None
        out["dual_bound"] = (int(round(solver.BestObjectiveBound()))
                             if name == "UNKNOWN" else None)
    return out


# ---------------------------------------------------------------------------
# Certified bound primitives
# ---------------------------------------------------------------------------

def trivial_bound(inst: mq.Instance) -> int:
    """L0 = max(max job load, max machine load).  Certified by definition."""
    return int(inst.lower_bound)


def full_model_bound(inst: mq.Instance, time_limit=60.0):
    """Solve the complete job-shop model; return status/objective/dual bound.

    The dual bound is certified for every terminating status; the objective
    is a feasible makespan (upper bound) only for OPTIMAL/FEASIBLE.
    """
    model, _ = _build_model(inst, minimize=True)
    r = _solve(model, time_limit)
    r["kind"] = "full_model_dual"
    return r


def subset_relaxation_bound(inst: mq.Instance, machine_subset, time_limit=60.0):
    """opt(R_S) for the given machine subset; certified only when OPTIMAL,
    otherwise the CP-SAT dual bound of the relaxation is used (also valid:
    any dual bound of a relaxation lower-bounds C*)."""
    subset = tuple(sorted(int(m) for m in machine_subset))
    model, _ = _build_model(inst, machine_subset=subset, minimize=True)
    r = _solve(model, time_limit)
    r["kind"] = "subset_relaxation"
    r["machine_subset"] = list(subset)
    certified = r["status"] == "OPTIMAL"
    r["certified_bound"] = (r["objective"] if certified
                            else r["dual_bound"])
    r["is_proven_optimum_of_relaxation"] = certified
    return r


def threshold_infeasibility(inst: mq.Instance, threshold: int,
                            time_limit=60.0):
    """Feasibility of C_max <= T.  Returns verdict; 'infeasible' CERTIFIES
    C* >= T+1, every other verdict certifies nothing about the lower bound."""
    model, _ = _build_model(inst, threshold=threshold, minimize=False)
    r = _solve(model, time_limit)
    r["kind"] = "threshold_infeasibility"
    r["threshold"] = int(threshold)
    r["verdict"] = {"INFEASIBLE": "infeasible",
                    "OPTIMAL": "feasible",
                    "FEASIBLE": "feasible"}.get(r["status"], "unproven")
    r["raises_lower_bound_to"] = (int(threshold) + 1
                                  if r["verdict"] == "infeasible" else None)
    return r


# ---------------------------------------------------------------------------
# Record keeping (bound_records.jsonl)
# ---------------------------------------------------------------------------

def record_fields(inst: mq.Instance, entry: dict) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "instance": inst.name,
        "instance_sha256": hashlib.sha256(
            np.ascontiguousarray(inst.durations, dtype=np.int64).tobytes()
            + np.ascontiguousarray(inst.routes, dtype=np.int64).tobytes()
        ).hexdigest(),
        "scope": "global",
        "ortools_version": ORTOOLS_VERSION,
        **{k: v for k, v in entry.items()},
    }
    return rec


def append_record(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_d0(time_limit, records_path):
    import data_identity
    rows = []
    for jobs, machines in data_identity.D0_SIZES:
        for seed in range(10):
            inst = data_identity.build_d0(jobs, machines, seed)
            enum = data_identity.enumerate_exact_optimum(inst)
            c_star = int(enum["exact_optimum"])
            l0 = trivial_bound(inst)
            # feasible upper bound from a serial SGS schedule, independently
            # verified with the production decoder
            starts, u0 = mq.serial_sgs(inst, "mwr",
                                       np.random.default_rng(seed))
            orders = [tuple(sorted(inst.groups[m], key=lambda v: starts[v]))
                      for m in range(inst.machines)]
            dec = mq.decode(inst, orders)
            assert dec["feasible"] and dec["makespan"] == u0
            checks = {"l0_le_cstar": l0 <= c_star, "cstar_le_u": c_star <= u0}
            # threshold verdicts vs enumeration (the T02 acceptance core)
            below = threshold_infeasibility(inst, c_star - 1, time_limit)
            at = threshold_infeasibility(inst, c_star, time_limit)
            checks["threshold_below_cstar_infeasible"] = \
                below["verdict"] == "infeasible"
            checks["threshold_at_cstar_feasible"] = at["verdict"] == "feasible"
            checks["l0_le_threshold_raise"] = (
                below["raises_lower_bound_to"] is None
                or below["raises_lower_bound_to"] <= c_star + 1)
            # relaxation monotonicity on a small and a larger subset
            half = subset_relaxation_bound(inst, range(inst.machines - 1),
                                           time_limit)
            single = subset_relaxation_bound(inst, (0,), time_limit)
            checks["subset_relax_le_cstar"] = \
                half["certified_bound"] is not None \
                and half["certified_bound"] <= c_star
            checks["relaxation_monotone"] = (
                single["certified_bound"] is not None
                and half["certified_bound"] is not None
                and single["certified_bound"] <= half["certified_bound"])
            rows.append({
                "scale": f"{jobs}x{machines}", "seed": seed,
                "c_star": c_star, "l0": l0, "u_sgs": u0,
                "l_relax_half": half["certified_bound"],
                "l_relax_single": single["certified_bound"],
                "threshold_below": below["verdict"],
                "threshold_at": at["verdict"],
                "checks": checks,
                "all_checks_pass": all(checks.values()),
            })
            for entry in ({"kind": "trivial", "bound": l0},
                          {"kind": "subset_relaxation_single_machine_0",
                           "bound": single["certified_bound"]},
                          {"kind": "threshold_infeasible",
                           "threshold": c_star - 1,
                           "bound": below["raises_lower_bound_to"],
                           "solver_status": below["status"]}):
                append_record(records_path, record_fields(inst, {**entry}))
    return rows


def run_d1(time_limit, records_path):
    """Reproduce known bounds on the six D1 Taillard instances.

    Reference optima (JSPLib, LB=UB): ta01 1231, ta11 1357, ta21 1642,
    ta61 2868, ta62 2869, ta63 2755.  We report our certified L next to
    them; a number is only claimed as a *new* lower bound if it exceeds the
    published one, which this pilot does not expect.
    """
    files = {
        "ta01": ("task_data/tai15_15_01_test.txt", 1231),
        "ta11": ("task_data/tai20_15_01_test.txt", 1357),
        "ta21": ("task_data/ta21.txt", 1642),
        "ta61": ("task_data/tai50_20_01.txt", 2868),
        "ta62": ("task_data/tai50_20_02.txt", 2869),
        "ta63": ("task_data/tai50_20_03.txt", 2755),
    }
    rows = []
    for name, (rel, known_opt) in files.items():
        path = ROOT / rel
        if not path.exists():
            rows.append({"instance": name, "error": f"missing {rel}"})
            continue
        inst = mq.Instance.read(path)
        l0 = trivial_bound(inst)
        rec = {"instance": name, "jobs": inst.jobs, "machines": inst.machines,
               "known_optimum_lb_eq_ub": known_opt, "l0": l0}
        # strongest cheap certified family: single-machine relaxations
        best_relax = None
        for m in range(inst.machines):
            r = subset_relaxation_bound(inst, (m,), time_limit)
            b = r["certified_bound"]
            append_record(records_path, record_fields(inst, {
                "kind": "subset_relaxation_single_machine",
                "machine": m, "bound": b, "solver_status": r["status"],
                "is_proven_optimum_of_relaxation":
                    r["is_proven_optimum_of_relaxation"]}))
            if b is not None and (best_relax is None or b > best_relax):
                best_relax = b
        rec["l_single_machine_relaxation_best"] = best_relax
        # one threshold probe just below the known optimum: INFEASIBLE would
        # reproduce the published LB; anything else stays unproven.
        probe = threshold_infeasibility(inst, known_opt - 1, time_limit)
        append_record(records_path, record_fields(inst, {
            "kind": "threshold_infeasible", "threshold": known_opt - 1,
            "bound": probe["raises_lower_bound_to"],
            "solver_status": probe["status"]}))
        rec["threshold_probe"] = {"threshold": known_opt - 1,
                                  "verdict": probe["verdict"],
                                  "status": probe["status"]}
        rec["certified_l"] = max(x for x in (l0, best_relax,
                                             probe["raises_lower_bound_to"])
                                 if x is not None)
        rec["no_new_lb_claimed"] = rec["certified_l"] <= known_opt
        rows.append(rec)
    return rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="T02 certified global bounds")
    ap.add_argument("--d0", action="store_true",
                    help="D0 battery with enumeration cross-checks")
    ap.add_argument("--d1", action="store_true",
                    help="D1 known-bound reproduction")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--out", default=str(GATES_DIR / "results_global_bounds_20261003"
                                         / "bound_records.jsonl"))
    ap.add_argument("--summary", default=str(GATES_DIR / "results_global_bounds_20261003"
                                             / "summary.json"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = Path(args.out)
    summary = {}
    if args.d0:
        rows = run_d0(args.time_limit, records)
        summary["d0"] = {"windows": len(rows),
                         "all_checks_pass":
                             all(r["all_checks_pass"] for r in rows)}
    if args.d1:
        rows = run_d1(args.time_limit, records)
        summary["d1"] = rows
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps({k: (v if k != "d1" else
                          [{kk: r.get(kk) for kk in
                            ("instance", "l0", "l_single_machine_relaxation_best",
                             "certified_l", "no_new_lb_claimed")}
                           for r in v])
                      for k, v in summary.items()},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
