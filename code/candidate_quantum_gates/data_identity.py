#!/usr/bin/env python3
"""Instance identity, D0 generation and exact small-instance optima (T00).

The rest of the T00-T10 plan compares methods across methods, seeds and
budgets.  Every such comparison is only meaningful if the compared runs used
the *same instance*, so identity is pinned by content, never by file name:
``instance_identity`` hashes the full duration/route matrix, and the record
carries that hash next to the wall-clock and cost fields.

Two further responsibilities live here, both deliberately free of any pool,
witness or sampler code so that they can serve as an independent referee:

``build_d0``
    Deterministic generation of the D0 correctness family (2x2, 3x3, 4x3) with
    the frozen split of the breakdown document: seeds 0-4 development, 5-9
    validation.  Durations are independent uniform integers on 1..9 and every
    job visits every machine exactly once.

``enumerate_exact_optimum``
    Brute-force optimum over *all* machine orderings of a small instance, which
    is a genuine optimality certificate for D0 sizes (4x3 costs 13,824
    combinations).  The answer is written to a separate diagnostic file: it is
    ground truth for checking a pool, and it must never be fed to a sampler or
    to any online search.
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

_MEDIUM_DIR = Path(__file__).resolve().parents[1] / "candidate_quantum_medium"
if str(_MEDIUM_DIR) not in sys.path:
    sys.path.insert(0, str(_MEDIUM_DIR))
import medium_qjsp as mq  # noqa: E402  (sibling package, graph evaluation)

__all__ = [
    "D0_SIZES",
    "DATA_VERSION",
    "DEVELOPMENT_SEEDS",
    "VALIDATION_SEEDS",
    "build_d0",
    "d0_records",
    "d0_split_of",
    "enumerate_exact_optimum",
    "instance_identity",
    "instance_sha256",
    "load_instance_file",
    "restricted_pool_optimum",
    "write_text_instance",
]

#: Frozen D0 family and split.  Changing either invalidates every downstream
#: number, so the values are recorded in the manifest and re-checked by tests.
D0_SIZES = ((2, 2), (3, 3), (4, 3))
DEVELOPMENT_SEEDS = (0, 1, 2, 3, 4)
VALIDATION_SEEDS = (5, 6, 7, 8, 9)
DATA_VERSION = "d0-v1"
DURATION_LOW, DURATION_HIGH = 1, 9


def instance_identity(inst) -> dict:
    """Content identity of an instance: durations, routes and their hash."""
    durations = np.asarray(inst.durations, dtype=np.int64)
    routes = np.asarray(inst.routes, dtype=np.int64)
    return {
        "name": str(inst.name),
        "jobs": int(durations.shape[0]),
        "machines": int(durations.shape[1]),
        "durations": durations.tolist(),
        "routes": routes.tolist(),
        "lower_bound": int(inst.lower_bound),
        "total_duration": int(inst.total_duration),
        "instance_sha256": instance_sha256(inst),
    }


def instance_sha256(inst) -> str:
    """SHA-256 over the canonical duration/route payload.

    Uses ``medium_qjsp.content_hash`` so that the identity recorded by the
    existing ``adaptive_search`` reports and by this module cannot drift apart.
    """
    return mq.content_hash({
        "durations": np.asarray(inst.durations, dtype=np.int64).tolist(),
        "routes": np.asarray(inst.routes, dtype=np.int64).tolist(),
    })


def build_d0(jobs: int, machines: int, seed: int, *, name: str | None = None) -> mq.Instance:
    """Deterministic D0 instance: integer durations 1..9, per-job route draw.

    Only ``numpy.random.default_rng(seed)`` is used and the draw order is fixed
    (durations first, then one route permutation per job), so the same
    ``(jobs, machines, seed)`` always yields the same instance and hash.
    """
    if jobs < 1 or machines < 1:
        raise ValueError("positive instance dimensions required")
    rng = np.random.default_rng(int(seed))
    durations = rng.integers(DURATION_LOW, DURATION_HIGH + 1, size=(jobs, machines))
    routes = np.array([rng.permutation(machines) for _ in range(jobs)], dtype=np.int64)
    return mq.Instance(durations, routes,
                       name=name or f"d0_{jobs}x{machines}_seed{int(seed)}")


def d0_split_of(seed: int) -> str:
    if int(seed) in DEVELOPMENT_SEEDS:
        return "development"
    if int(seed) in VALIDATION_SEEDS:
        return "validation"
    raise ValueError(f"seed {seed} is outside the frozen D0 split")


def d0_records():
    """Yield ``(jobs, machines, seed, split, instance)`` for the whole D0 family."""
    for jobs, machines in D0_SIZES:
        for seed in DEVELOPMENT_SEEDS + VALIDATION_SEEDS:
            yield jobs, machines, seed, d0_split_of(seed), build_d0(jobs, machines, seed)


def write_text_instance(inst, path) -> str:
    """Serialize to the repository's text format: durations then 1-based routes.

    Round-tripping through ``medium_qjsp.Instance.read`` must reproduce the same
    identity hash; ``verify_instance_file`` checks that, and the tests assert it
    for the whole D0 family.
    """
    path = Path(path)
    durations = np.asarray(inst.durations, dtype=np.int64)
    routes = np.asarray(inst.routes, dtype=np.int64)
    payload = np.vstack([durations, routes + 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(str(int(value)) for value in row) for row in payload]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return instance_sha256(inst)


def load_instance_file(path) -> mq.Instance:
    return mq.Instance.read(Path(path))


def enumerate_exact_optimum(inst, *, limit: int | None = None) -> dict:
    """Brute-force optimum over all machine orderings (D0-sized instances only).

    Returns the exact optimum, the number of combinations examined, the number
    of feasible ones, and the verified optimal machine orders.  The best
    schedule is re-checked with the independent ``validate_schedule`` routine so
    the certificate does not rest on the decoder alone.
    """
    groups = [list(group) for group in inst.groups]
    orders_per_machine = [list(_permutations(group)) for group in groups]
    total = 1
    for orders in orders_per_machine:
        total *= len(orders)
    if limit is not None and total > int(limit):
        raise ValueError(f"enumeration space {total} exceeds limit {limit}")
    best_value, best_orders, feasible = None, None, 0
    for combination in product(*orders_per_machine):
        result = mq.decode(inst, [list(order) for order in combination], validate=False)
        if not result["feasible"]:
            continue
        feasible += 1
        if best_value is None or result["makespan"] < best_value:
            best_value, best_orders = int(result["makespan"]), combination
    if best_value is None:
        raise ValueError("no feasible schedule found; instance is malformed")
    verified = mq.validate_schedule(inst, mq.decode(inst, [list(o) for o in best_orders])["starts"])
    if verified != best_value:
        raise RuntimeError("independent schedule verification disagrees with the decoder")
    return {
        "exact_optimum": int(best_value),
        "combinations": int(total),
        "feasible_combinations": int(feasible),
        "orders": [[int(v) for v in order] for order in best_orders],
        "independent_verification": True,
    }


def restricted_pool_optimum(inst, pool) -> dict:
    """Pool optimum ``C_Pi^*`` and its relation to the exact optimum.

    ``pool`` is a sequence of per-machine candidate order lists, as produced by
    ``medium_qjsp.build_pool`` or written by hand.  This exists to demonstrate
    the scope rule the breakdown document insists on: a pool bound is a bound
    on the pool, and it must not be reported as a global bound.
    """
    mq.validate_pool(inst, pool)
    best_value, best_choice = None, None
    for choice in product(*[range(len(candidates)) for candidates in pool]):
        result = mq.decode(inst, [list(pool[m][a]) for m, a in enumerate(choice)],
                           validate=False)
        if not result["feasible"]:
            continue
        if best_value is None or result["makespan"] < best_value:
            best_value, best_choice = int(result["makespan"]), tuple(choice)
    return {"pool_optimum": best_value, "choice": best_choice,
            "pool_hash": mq.content_hash([[list(map(int, order)) for order in candidates]
                                          for candidates in pool])}


def d0_manifest(entries) -> dict:
    """Manifest payload for a set of D0 records (pure data, no file paths)."""
    rows = []
    for jobs, machines, seed, split, inst in entries:
        rows.append({
            "dataset": "D0",
            "data_version": DATA_VERSION,
            "jobs": int(jobs),
            "machines": int(machines),
            "seed": int(seed),
            "split": split,
            "name": inst.name,
            **{key: value for key, value in instance_identity(inst).items()
               if key not in {"name", "jobs", "machines"}},
        })
    rows.sort(key=lambda row: (row["jobs"], row["machines"], row["seed"]))
    return {
        "schema_version": 1,
        "dataset": "D0",
        "data_version": DATA_VERSION,
        "sizes": [list(size) for size in D0_SIZES],
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "generator": "data_identity.build_d0",
        "duration_range": [DURATION_LOW, DURATION_HIGH],
        "instance_count": len(rows),
        "instances": rows,
        "manifest_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _permutations(values):
    """Deterministic lexicographic permutations (avoids import-time surprises)."""
    items = sorted(values)
    if len(items) <= 1:
        yield tuple(items)
        return
    for index, head in enumerate(items):
        rest = items[:index] + items[index + 1:]
        for tail in _permutations(rest):
            yield (head,) + tail
