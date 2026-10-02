#!/usr/bin/env python3
"""Validate saved schedules without trusting graph-decoder feasibility flags."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from medium_qjsp import (Instance, choice_orders, content_hash,
                         validate_pool, validate_schedule)


def verify_report(report):
    data = report["instance_data"]
    if content_hash(data) != report["instance_sha256"]:
        raise ValueError("instance hash mismatch")
    inst = Instance(np.array(data["durations"]), np.array(data["routes_zero_based"]),
                    report["instance"])
    if (inst.jobs, inst.machines) != (report["jobs"], report["machines"]):
        raise ValueError("reported dimensions mismatch")
    if inst.lower_bound != report["simple_lower_bound"]:
        raise ValueError("reported lower bound mismatch")
    pool = report["pool_data"]
    validate_pool(inst, pool)
    if content_hash(pool) != report["pool_sha256"]:
        raise ValueError("pool hash mismatch")
    if report["best_makespan"] is None:
        if any(report[key] is not None for key in ("best_choice", "best_starts", "best_orders")):
            raise ValueError("inconsistent missing incumbent")
        return {"valid": True, "has_incumbent": False}
    orders = choice_orders(pool, report["best_choice"])
    if orders != report["best_orders"]:
        raise ValueError("selected orders mismatch")
    starts = report["best_starts"]
    makespan = validate_schedule(inst, starts)
    if makespan != report["best_makespan"]:
        raise ValueError("reported makespan mismatch")
    for order in orders:
        for u, v in zip(order, order[1:]):
            if starts[u] + inst.processing[u] > starts[v]:
                raise ValueError("starts do not respect selected machine orders")
    return {"valid": True, "has_incumbent": True, "makespan": makespan,
            "checks": "input/pool hashes, dimensions, lower bound, one-hot choice, job precedence, machine non-overlap, selected orders"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args(argv)
    for path in args.results:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        result = verify_report(report)
        print(json.dumps({"path": path, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
