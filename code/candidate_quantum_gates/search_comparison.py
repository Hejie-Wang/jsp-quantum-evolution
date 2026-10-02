#!/usr/bin/env python3
"""Multi-seed comparison of uniform / XY / XY+joint / classical search (P3).

Methods share the same candidate pool, the same initial verified schedule,
the same witness set evolution and the same per-round evaluation budget:

  uniform       uniform legal sampling, no phase, no mixer structure
  xy            witness phase + single-machine XY mixers
  xy_joint      witness phase + XY + witness-derived joint transitions
  classical     hill climbing on the same frozen witnesses and the same
                joint actions (no quantum circuit)

The exact pool-inner MILP optimum is timed SEPARATELY and only serves as the
success reference (target makespan).  Results are saved as JSON, including
failures and degenerate runs; a quantum energy is not claimed to decrease
monotonically and sampling failure proves nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from circuits import CandidatePool
from pool_optimum_milp import solve_pool_optimum_milp
from search_loop import (classical_joint_search, demo_candidate_pool,
                         demo_instance, random_instance, run_search_loop)

MODES = ("uniform", "xy", "xy_joint", "classical")


def build_case(case, seed):
    if case == "demo_3x3":
        inst = demo_instance()
        pool = demo_candidate_pool()
    else:
        jobs, machines = case.split("x")
        inst = random_instance(int(jobs), int(machines), seed)
        import medium_qjsp as mq
        pool_data, _info = mq.build_pool(inst, candidates_per_machine=2,
                                         n_schedules=120, seed=seed)
        pool = CandidatePool(tuple(tuple(tuple(order) for order in machine)
                                   for machine in pool_data))
    return inst, pool


def run_case(case, seed, rounds, shots, milp_time_limit):
    inst, pool = build_case(case, seed)
    initial = [0] * pool.machines
    milp_started = perf_counter()
    milp = solve_pool_optimum_milp(inst, pool.candidates,
                                   _u_bound(inst, pool, initial),
                                   time_limit=milp_time_limit)
    milp["seconds"] = perf_counter() - milp_started
    target = milp.get("objective")
    rows = []
    reports = {}
    for mode in MODES:
        if mode == "classical":
            report = classical_joint_search(inst, pool, initial, seed=seed,
                                            max_rounds=rounds)
        else:
            report = run_search_loop(inst, pool, initial, mode=mode,
                                     shots=shots, seed=seed, max_rounds=rounds,
                                     per_round_evaluations=8, train_budget=48,
                                     time_budget=900.0)
        reports[mode] = report
        evaluated = sum(entry.get("evaluated", 0) for entry in report["trace"])
        feasible = sum(entry.get("graph_feasible", 0) for entry in report["trace"])
        rows.append({
            "mode": mode,
            "initial_makespan": report["initial_makespan"],
            "best_makespan": report["best_makespan"],
            "improved": report["improved"],
            "reached_target": (report["best_makespan"] <= target
                               if target is not None else None),
            "feasible_rate": (feasible / evaluated if evaluated else None),
            "evaluation_count": report["evaluation_count"],
            "training_evaluations": report["training_evaluations"],
            "training_seconds": report["training_seconds"],
            "shots_total": report["shots_total"],
            "wall_seconds": report["wall_seconds"],
            "rounds": report["rounds"],
            "witnesses": report["witnesses"],
            "independent_verification": report["independent_schedule_verification"],
            "circuit_qubits": (report["trace"][0]["circuit"]["num_qubits"]
                               if report["trace"] and "circuit" in report["trace"][0]
                               else None),
            "circuit_depth_max": (max(entry["circuit"]["depth"]
                                      for entry in report["trace"]
                                      if "circuit" in entry)
                                  if any("circuit" in entry
                                         for entry in report["trace"]) else None),
        })
    return {
        "case": case, "seed": seed, "pool_sizes": list(pool.sizes),
        "pool_hash": pool.content_hash, "data_qubits": pool.data_qubits,
        "ansatz_qubits": pool.data_qubits + pool.machines + 3,
        "milp_reference": {"status": milp["status"], "objective": target,
                           "dual_bound": milp.get("mip_dual_bound"),
                           "seconds": milp["seconds"], "detail": milp},
        "rows": rows,
        "reports": reports,
    }


def _u_bound(inst, pool, initial):
    import medium_qjsp as mq
    decoded = mq.decode(inst, mq.choice_orders(pool.candidates, initial))
    if not decoded["feasible"]:
        raise ValueError("initial choice must decode feasible")
    return mq.validate_schedule(inst, decoded["starts"])


def aggregate(results):
    summary = {}
    for mode in MODES:
        rows = [row for result in results for row in result["rows"]
                if row["mode"] == mode]
        targets = [result["milp_reference"]["objective"] for result in results
                   for row in result["rows"] if row["mode"] == mode]
        success = sum(1 for row, target in zip(rows, targets)
                      if row["reached_target"])
        summary[mode] = {
            "cases": len(rows),
            "target_success_rate": success / len(rows) if rows else None,
            "improved_rate": (sum(1 for row in rows if row["improved"])
                              / len(rows)) if rows else None,
            "mean_wall_seconds": (sum(row["wall_seconds"] for row in rows)
                                  / len(rows)) if rows else None,
            "mean_training_seconds": (sum(row["training_seconds"] for row in rows)
                                      / len(rows)) if rows else None,
            "mean_evaluations": (sum(row["evaluation_count"] for row in rows)
                                 / len(rows)) if rows else None,
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+",
                        default=["demo_3x3", "5x5", "5x5"],
                        help="case names; pair with --seeds one to one")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[7, 11, 13])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--milp-time-limit", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.cases) != len(args.seeds):
        parser.error("--cases and --seeds must have the same length")
    results = []
    for case, seed in zip(args.cases, args.seeds):
        print(f"running {case} seed={seed} ...", flush=True)
        results.append(run_case(case, seed, args.rounds, args.shots,
                                args.milp_time_limit))
    payload = {
        "schema_version": 1,
        "budget": {"rounds": args.rounds, "shots": args.shots,
                   "per_round_evaluations": 8, "training_evaluations_cap": 48},
        "notes": [
            "Exact MILP reference is timed separately from the methods.",
            "Success = verified best makespan reaches the pool-inner optimum.",
            "Failures and degenerate runs are retained, never discarded.",
        ],
        "aggregate_by_mode": None,
        "results": results,
    }
    payload["aggregate_by_mode"] = aggregate(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                      default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate_by_mode"], indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
