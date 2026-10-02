#!/usr/bin/env python3
"""Compare MILP and classical Kaiwu SA on identical medium-scale pools."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from medium_qjsp import build_pool, write_json
from run_medium import brief_summary, load_instance, run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(Path(__file__).parents[2] / "task_data"))
    parser.add_argument("--output-dir", default="results/corrected_20261001")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--time-limit", type=float, default=40.)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--schedules", type=int, default=240)
    parser.add_argument("--sa-iterations-per-t", type=int, default=50)
    parser.add_argument("--sa-size-limit", type=int, default=30)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = [("15x15", "tai15_15_01_test.txt", None),
             ("20x15", "tai20_15_01_test.txt", None),
             ("15x20_derived", "ta21.txt", 15)]
    summaries = []
    for case, filename, first_jobs in cases:
        inst, source = load_instance(Path(args.data_dir) / filename, first_jobs)
        for seed in args.seeds:
            tic = perf_counter()
            pool, info = build_pool(inst, args.candidates, args.schedules, seed)
            pool_seconds = perf_counter() - tic
            hashes = set()
            for master in ("milp", "kaiwu"):
                print(f"RUN {case} seed={seed} master={master} baseline={info['makespan']}", flush=True)
                report = run_experiment(
                    inst, pool, info, pool_seconds, source, master=master, seed=seed,
                    iterations=args.iterations, time_limit=args.time_limit,
                    samples_per_iter=4, sa_options={"iterations_per_t": args.sa_iterations_per_t,
                                                  "size_limit": args.sa_size_limit})
                report["case"] = case
                hashes.add(report["pool_sha256"])
                path = output / f"{case}_seed{seed}_{master}.json"
                write_json(path, report)
                row = {"case": case, **brief_summary(report),
                       "pool_sha256": report["pool_sha256"],
                       "verified": report["independent_verification"]["valid"],
                       "result_path": str(path)}
                summaries.append(row)
                write_json(output / "summary.json", summaries)
                print(f"DONE best={row['best_makespan']} seconds={row['wall_seconds']:.2f} "
                      f"stop={row['stop_reason']} proof={row['proof_certificate']}", flush=True)
            if len(hashes) != 1:
                raise RuntimeError("backend candidate pools differ")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved {len(summaries)} verified runs in {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
