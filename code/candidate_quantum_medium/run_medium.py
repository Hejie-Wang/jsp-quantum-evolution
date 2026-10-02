#!/usr/bin/env python3
"""Run the candidate-master cut loop on a medium JSP instance.

Example (conda Kaiwu env provides kaiwu + scipy + numpy):
    python run_medium.py --instance ../../task_data/tai20_15_01_test.txt \
        --master milp --candidates 8 --output results/tai20_15_01_milp.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter

from medium_qjsp import (Instance, build_pool, run_candidate_loop, write_json,
                         content_hash)
from verify_result import verify_report


def load_instance(path, first_jobs=None):
    path = Path(path).resolve()
    inst = Instance.read(path)
    source = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "original_jobs": inst.jobs, "original_machines": inst.machines,
              "derived": first_jobs is not None}
    if first_jobs is not None:
        if not 1 <= first_jobs <= inst.jobs:
            raise ValueError("first-jobs must be within source job count")
        inst = Instance(inst.durations[:first_jobs], inst.routes[:first_jobs],
                        f"{inst.name}_first{first_jobs}jobs_derived")
        source["derivation"] = f"first {first_jobs} jobs; all machines; not a published benchmark"
    return inst, source


def environment_info():
    versions = {}
    for package in ("numpy", "scipy", "kaiwu", "qiskit"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    runtime = {}
    if versions["kaiwu"] is not None:
        import kaiwu
        runtime["kaiwu"] = {
            "version": getattr(kaiwu, "__version__", None),
            "module_path": kaiwu.__file__,
            "solve_signature": str(inspect.signature(kaiwu.SimulatedAnnealingOptimizer.solve)),
        }
    return {"python_executable": sys.executable, "python_version": sys.version,
            "prefix": sys.prefix, "packages": versions, "runtime": runtime}


def run_experiment(inst, pool, pool_info, pool_seconds, source, *, master, seed,
                   iterations, time_limit, samples_per_iter, sa_options,
                   max_qubo_vars=2048, log=None):
    report = run_candidate_loop(
        inst, pool, master=master, max_iterations=iterations,
        time_limit=time_limit, seed=seed, samples_per_iter=samples_per_iter,
        sa_options=sa_options, max_qubo_vars=max_qubo_vars, log=log)
    report["pool"] = {"candidates_per_machine": [len(p) for p in pool],
                      "heuristic_best_makespan": pool_info["makespan"],
                      "schedules_generated": pool_info["schedules_generated"],
                      "build_seconds": pool_seconds}
    report["seed"] = seed
    report["source"] = source
    report["instance_sha256"] = content_hash(report["instance_data"])
    report["environment"] = environment_info()
    report["pool_plus_loop_seconds"] = pool_seconds + report["wall_seconds"]
    report["time_policy"] = "time-limit applies to cut loop; pool generation recorded separately; Kaiwu calls may overrun"
    report["independent_verification"] = verify_report(report)
    return report


def brief_summary(report):
    keys = ("instance", "jobs", "machines", "master", "seed", "initial_makespan",
            "best_makespan", "simple_lower_bound", "iterations", "wall_seconds",
            "pool_plus_loop_seconds", "proof_certificate", "stop_reason")
    return {k: report[k] for k in keys}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", required=True)
    p.add_argument("--first-jobs", type=int)
    p.add_argument("--master", choices=("milp", "kaiwu"), default="milp")
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--schedules", type=int, default=240)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--time-limit", type=float, default=600.0)
    p.add_argument("--samples-per-iter", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sa-iterations-per-t", type=int, default=100)
    p.add_argument("--sa-size-limit", type=int, default=50)
    p.add_argument("--max-qubo-vars", type=int, default=2048)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output")
    args = p.parse_args(argv)

    inst, source = load_instance(args.instance, args.first_jobs)
    tic = perf_counter()
    pool, pool_info = build_pool(inst, args.candidates, args.schedules, args.seed)
    pool_seconds = perf_counter() - tic
    print(f"pool built in {pool_seconds:.1f}s; heuristic best makespan "
          f"{pool_info['makespan']}; simple LB {inst.lower_bound}")

    sa_options = {"iterations_per_t": args.sa_iterations_per_t,
                  "size_limit": args.sa_size_limit}
    report = run_experiment(
        inst, pool, pool_info, pool_seconds, source, master=args.master,
        seed=args.seed, iterations=args.iterations, time_limit=args.time_limit,
        samples_per_iter=args.samples_per_iter, sa_options=sa_options,
        max_qubo_vars=args.max_qubo_vars, log=print if args.verbose else None)
    rendered = json.dumps(brief_summary(report), ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        write_json(args.output, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
