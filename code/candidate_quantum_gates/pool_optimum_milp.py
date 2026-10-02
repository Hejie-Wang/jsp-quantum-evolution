#!/usr/bin/env python3
"""Pool-inner optimum MILP diagnostic (P3 of the 2026-10-02 handoff).

For a saved candidate-pool report this script decides the exact best
makespan INSIDE that pool with the one-hot formulation of
docs/candidate_quantum_improvements_20261002.md:

    one-hot selection variables x, start times s, makespan C
    job edges:            s_v >= s_u + p_u
    machine edges:        s_v >= s_u + p_u - U * (1 - x_{m,a})   for each
                          consecutive pair (u, v) of candidate a on machine m
    bounds:               0 <= s_u <= U - p_u,  s_u + p_u <= C <= U
    minimize C

U is a verified feasible makespan that belongs to this pool (the report's
best_choice is decoded and independently validated first).  This is a POOL
bound, never a JSP global bound.  A feasible incumbent alone is not an
optimality proof; the saved evidence therefore records the solver status,
the MIP dual bound, the pool content hash and the post-decode validation.

This diagnostic is exact and separate from any quantum timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

_MEDIUM_DIR = Path(__file__).resolve().parents[1] / "candidate_quantum_medium"
if str(_MEDIUM_DIR) not in sys.path:
    sys.path.insert(0, str(_MEDIUM_DIR))
import medium_qjsp as mq  # noqa: E402


def load_pool_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    inst = mq.Instance(report["instance_data"]["durations"],
                       report["instance_data"]["routes_zero_based"],
                       name=str(report.get("instance", path)))
    pool = tuple(tuple(tuple(int(v) for v in order) for order in machine)
                 for machine in report["pool_data"])
    mq.validate_pool(inst, pool)
    return report, inst, pool


def solve_pool_optimum_milp(inst, pool, u_bound, time_limit=600.0):
    """Exact pool-inner optimum; returns status, makespan, bound, choice."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix, vstack

    if u_bound is None or u_bound <= 0:
        raise ValueError("U must be a positive verified makespan of this pool")
    sizes = [len(p) for p in pool]
    n_x = sum(sizes)
    n_s = inst.operations
    n = n_x + n_s + 1  # x | s | C
    c_index = n - 1

    def xcol(m, a):
        return sum(sizes[:m]) + a

    def scol(op):
        return n_x + op

    rows, lb, ub = [], [], []
    for m, size in enumerate(sizes):
        row = lil_matrix((1, n))
        for a in range(size):
            row[0, xcol(m, a)] = 1.0
        rows.append(row)
        lb.append(1.0)
        ub.append(1.0)
    for u, v in inst.job_arcs:
        row = lil_matrix((1, n))
        row[0, scol(u)] = 1.0
        row[0, scol(v)] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(-inst.processing[u])
    machine_edges = 0
    for m, candidates in enumerate(pool):
        for a, order in enumerate(candidates):
            for u, v in zip(order, order[1:]):
                row = lil_matrix((1, n))
                row[0, scol(u)] = 1.0
                row[0, scol(v)] = -1.0
                row[0, xcol(m, a)] = float(u_bound)
                rows.append(row)
                lb.append(-np.inf)
                ub.append(float(u_bound - inst.processing[u]))
                machine_edges += 1
    for op in range(inst.operations):
        row = lil_matrix((1, n))
        row[0, scol(op)] = 1.0
        row[0, c_index] = -1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(-inst.processing[op])
    a_mat = vstack(rows).tocsr()
    objective = np.zeros(n)
    objective[c_index] = 1.0
    lower = np.concatenate([np.zeros(n_x), np.zeros(n_s), [0.0]])
    upper = np.concatenate([np.ones(n_x),
                            np.full(n_s, float(u_bound), dtype=float),
                            [float(u_bound)]])
    for op in range(inst.operations):
        upper[n_x + op] = u_bound - inst.processing[op]
    integrality = np.zeros(n)
    integrality[:n_x] = 1
    started = perf_counter()
    result = milp(objective,
                  constraints=LinearConstraint(a_mat, lb, ub),
                  integrality=integrality,
                  bounds=Bounds(lower, upper),
                  options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    elapsed = perf_counter() - started
    status = {0: "optimal", 1: "iteration_or_time_limit", 2: "infeasible",
              3: "unbounded", 4: "other"}.get(int(result.status), "unknown")
    out = {
        "status": status,
        "message": str(result.message),
        "objective": None, "mip_dual_bound": None, "choice": None,
        "machine_edges": machine_edges,
        "variables": int(n), "constraint_rows": int(a_mat.shape[0]),
        "seconds": elapsed,
    }
    if result.status == 2:
        return out
    if result.x is None:
        out["status"] = "no_solution"
        return out
    out["objective"] = float(result.fun)
    if result.mip_dual_bound is not None:
        out["mip_dual_bound"] = float(result.mip_dual_bound)
    values = np.rint(np.asarray(result.x[:n_x])).astype(int)
    if mq.qubo_violation(pool, [], values)[0]:
        out["status"] = "non_one_hot_solution"
        return out
    choice = mq.one_hot_choice(pool, values)
    out["choice"] = choice
    return out


def diagnose_report(path, time_limit=600.0):
    report, inst, pool = load_pool_report(path)
    pool_hash = mq.content_hash(pool)
    saved_hash = report.get("pool_sha256")
    choice = report.get("best_choice")
    decoded = mq.decode(inst, mq.choice_orders(pool, choice))
    verified = None
    if decoded["feasible"]:
        verified = mq.validate_schedule(inst, decoded["starts"])
    u_bound = verified
    milp_out = solve_pool_optimum_milp(inst, pool, u_bound, time_limit)
    # Post-decode validation of the MILP solution: a feasible incumbent alone
    # proves nothing, the certificate is status == optimal plus the bound.
    milp_choice_validation = None
    if milp_out.get("choice"):
        result = mq.decode(inst, mq.choice_orders(pool, milp_out["choice"]))
        if result["feasible"]:
            milp_choice_validation = {
                "makespan": mq.validate_schedule(inst, result["starts"]),
                "independent_check": mq.validate_schedule(
                    inst, result["starts"]) == result["makespan"],
            }
        else:
            milp_choice_validation = {"feasible": False}
    return {
        "source": str(Path(path).resolve()),
        "source_sha256": __import__("hashlib").sha256(
            Path(path).read_bytes()).hexdigest(),
        "instance": inst.name, "jobs": inst.jobs, "machines": inst.machines,
        "pool_sizes": [len(p) for p in pool],
        "pool_sha256": pool_hash,
        "saved_pool_sha256": saved_hash,
        "pool_hash_matches": (saved_hash == pool_hash) if saved_hash else None,
        "report_best_makespan": report.get("best_makespan"),
        "report_best_choice_verified": {
            "feasible": decoded["feasible"],
            "makespan": verified,
            "independent_validation": verified == report.get("best_makespan")
            if verified is not None else False,
        },
        "u_bound": u_bound,
        "milp": milp_out,
        "milp_choice_validation": milp_choice_validation,
        "pool_optimum": milp_out.get("objective"),
        "notes": [
            "This is a POOL-inner optimum, not a JSP global optimum.",
            "Optimality proof = solver status 'optimal' plus matching dual "
            "bound; a feasible incumbent alone is not a proof.",
            "Start times are continuous; with fixed durations the optimum C "
            "equals the integer one.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True,
                        help="saved medium report with pool_data/instance_data")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = []
    for path in args.report:
        print(f"diagnosing {path} ...", file=sys.stderr)
        evidence.append(diagnose_report(path, args.time_limit))
    payload = {
        "schema_version": 1,
        "formulation": "one-hot x + continuous s + C, machine edges relaxed "
                       "by U*(1-x_ma); minimize C",
        "results": evidence,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    for entry in evidence:
        print(f"{Path(entry['source']).name}: pool_optimum="
              f"{entry['pool_optimum']} status={entry['milp']['status']} "
              f"bound={entry['milp']['mip_dual_bound']} "
              f"({entry['milp']['seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
