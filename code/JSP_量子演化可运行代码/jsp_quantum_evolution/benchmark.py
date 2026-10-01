#!/usr/bin/env python3
"""Reproducible CPU timings; tau is not a physical time or guarantee."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np

from qjsp import (Instance, demo_instance, estimate_resources, export_schedule,
                  solve, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--layers", type=int, default=2000)
    p.add_argument("--tau", type=float, default=20)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    root = Path(__file__).resolve().parent
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for jobs, machines in ((2, 2), (3, 3), (4, 3)):
        inst = demo_instance(jobs, machines, seed=7)
        write_json(output / f"{inst.name}_input.json", inst.as_dict())
        timings = []
        for repeat in range(args.repeats):
            report, _, model = solve(inst, tau=args.tau, layers=args.layers,
                                      shots=1000, seed=7)
            timings.append(report["timings_seconds"])
            print(f"{inst.name} repeat {repeat+1}: {timings[-1]['total']:.4f}s", flush=True)
        write_json(output / f"{inst.name}_result.json", report)
        export_schedule(inst, report["sampling"]["best_measured_schedule"],
                        output / f"{inst.name}_schedule.csv")
        rows.append({"name": inst.name, "jobs": jobs, "machines": machines,
                     "states": model.dimension,
                     "timing_median_seconds": {key: float(np.median([r[key] for r in timings]))
                                               for key in timings[0]},
                     "timing_all_runs_seconds": timings,
                     "final_diagnostics": report["final_state_diagnostics"],
                     "best_measured_makespan": report["sampling"]["best_measured_schedule"]["makespan"],
                     "optimal_shots": report["sampling"]["optimal_shots_diagnostic"],
                     "norm_squared_error": report["evolution"]["max_recorded_norm_squared_error"]})
    large = []
    for path in sorted((root / "data").glob("tai50_20_*.txt")):
        start = perf_counter()
        inst = Instance.read(path)
        estimate = estimate_resources(inst, args.layers)
        estimate["parse_and_estimate_seconds"] = perf_counter()-start
        estimate["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        large.append(estimate)
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    result = {"environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                               "platform": platform.platform(), "cpu": cpu,
                               "visible_logical_cpus": os.cpu_count(),
                               "blas_thread_env": {k: os.environ.get(k) for k in
                                                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
              "configuration": {"tau": args.tau, "layers": args.layers, "s_final": 0.95,
                                "shots": 1000, "seed": 7, "repeats": args.repeats},
              "small_instances": rows, "user_instances_estimates_only": large,
              "interpretation": "CPU simulation timings; no hardware execution and no claimed quantum speedup."}
    write_json(output / "benchmark.json", result)
    print(f"Saved {output / 'benchmark.json'}")


if __name__ == "__main__":
    main()
