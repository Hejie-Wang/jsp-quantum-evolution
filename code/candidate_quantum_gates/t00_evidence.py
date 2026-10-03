#!/usr/bin/env python3
"""T00 evidence: cost separation between the uniform control and its legacy path.

    python t00_evidence.py --report reports/t00_evidence.json

Measures, on the D0 family, the preparation cost the legacy uniform arm paid by
materialising the whole legal subspace, and re-runs the uniform arm end to end
through both paths on a fixed instance and seed.

Scope of the claim
------------------
This script measures *preparation cost* and *reproducibility*.  It deliberately
does not claim that the quantum arm is better or worse than the classical arm:
T00 only has to make the comparison fair, which means the uniform control must
be an independent sampler rather than a quantum simulation.  Any statement about
search quality belongs to T06/T10 and would need the frozen protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_search import solve
from circuits import CandidatePool
from compact_simulator import CompactSimulator
from data_identity import build_d0, instance_sha256
from search_loop import demo_instance, graph_witness_specs, propose_joint_actions
from uniform_sampler import UniformSampler


def _distinct_orders(width: int, count: int) -> tuple[tuple[int, ...], ...]:
    """``count`` distinct permutations of ``range(width)`` (order content is irrelevant here)."""
    base = list(range(width))
    orders = [tuple(base)]
    for shift in range(1, count):
        orders.append(tuple(base[shift:] + base[:shift]))
    return tuple(orders)


def pool_study(size: int, candidates: int, shots: int, repetitions: int = 20) -> dict:
    """Synthetic pool with ``size`` active machines and ``candidates`` each.

    Sizes are kept inside the ``CompactSimulator`` cap (65,536 states) so the two
    paths can be timed on the same pool; beyond the cap the legacy path cannot be
    constructed at all, which ``zero_statevector_probe`` reports separately.
    """
    pool = CandidatePool(tuple(_distinct_orders(size, candidates) for _ in range(size)))
    rng = np.random.default_rng(7)
    uniform = UniformSampler(pool)
    start = time.perf_counter()
    for _ in range(repetitions):
        uniform.sample(shots, rng)
    uniform_seconds = (time.perf_counter() - start) / repetitions

    # The legacy path allocates and fills the full legal subspace before it can
    # draw anything; time exactly that preparation, not the draw.
    start = time.perf_counter()
    for _ in range(repetitions):
        CompactSimulator(pool, (), (), 10 ** 9)
    legacy_seconds = (time.perf_counter() - start) / repetitions
    sampler = CompactSimulator(pool, (), (), 10 ** 9)
    rng = np.random.default_rng(7)
    start = time.perf_counter()
    for _ in range(repetitions):
        sampler.sample([], shots, rng, "uniform")
    legacy_sampling_seconds = (time.perf_counter() - start) / repetitions
    return {
        "machines": size,
        "candidates_per_machine": candidates,
        "product_space_states": candidates ** size,
        "shots": shots,
        "uniform_seconds": uniform_seconds,
        "legacy_preparation_seconds": legacy_seconds,
        "legacy_preparation_and_draw_seconds": legacy_seconds + legacy_sampling_seconds,
        "preparation_speedup": legacy_seconds / uniform_seconds if uniform_seconds else None,
    }


def instance_run(instance_path: Path, *, seconds: float, seed: int) -> dict:
    from medium_qjsp import Instance

    inst = Instance.read(instance_path)
    started = time.perf_counter()
    modern = solve(inst, seconds=seconds, seed=seed, mode="uniform", max_iterations=200000)
    modern_seconds = time.perf_counter() - started
    started = time.perf_counter()
    legacy = solve(inst, seconds=seconds, seed=seed, mode="uniform", max_iterations=200000,
                   legacy_uniform=True)
    legacy_seconds = time.perf_counter() - started
    return {
        "instance": inst.name,
        "instance_sha256": instance_sha256(inst),
        "seed": seed,
        "search_budget_seconds": seconds,
        "uniform": {
            "backend": modern["uniform_sampler_backend"],
            "compact_simulator_constructions": modern["compact_simulator_constructions"],
            "sampler_seconds": modern["uniform_sampler_construction_seconds"],
            "proposal_calls": modern["proposal_calls"],
            "iterations": modern["iterations"],
            "best_makespan": modern["best_makespan"],
            "wall_seconds": modern["wall_seconds"],
            "end_to_end_seconds": modern_seconds,
            "independent_schedule_verification": modern["independent_schedule_verification"],
        },
        "legacy_uniform": {
            "backend": legacy["uniform_sampler_backend"],
            "compact_simulator_constructions": legacy["compact_simulator_constructions"],
            "proposal_seconds": legacy["proposal_seconds"],
            "proposal_calls": legacy["proposal_calls"],
            "iterations": legacy["iterations"],
            "best_makespan": legacy["best_makespan"],
            "wall_seconds": legacy["wall_seconds"],
            "end_to_end_seconds": legacy_seconds,
            "independent_schedule_verification": legacy["independent_schedule_verification"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--instance", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "4x3" / "seed0.txt")
    args = parser.parse_args()

    pool_rows = [pool_study(size, 3, 64) for size in (4, 5, 6, 7, 8)]
    report = {
        "schema_version": 1,
        "task": "T00",
        "purpose": ("Independent uniform control: cost separation from the legacy "
                    "compact-simulator path, and end-to-end reproducibility"),
        "claim_scope": ("Preparation cost and reproducibility only. No search-quality or "
                        "quantum-advantage claim."),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "pool_preparation": pool_rows,
        "instance_runs": [],
        "zero_statevector_probe": None,
    }

    # Direct check that the uniform arm never builds the subspace: a pool whose
    # product space exceeds the CompactSimulator cap would raise there, while the
    # independent sampler stays cheap.
    big_pool = CandidatePool(tuple(_distinct_orders(9, 9) for _ in range(9)))
    try:
        CompactSimulator(big_pool, (), (), 10 ** 9)
        probe = {"legacy_path": "constructed", "note": "unexpected; pool under cap"}
    except ValueError as error:
        probe = {"legacy_path": "rejected", "error": str(error)}
    sampler = UniformSampler(big_pool)
    sample = sampler.sample(8, np.random.default_rng(1))
    probe.update({"product_space_states": sampler.dimension,
                  "independent_sampler_samples": int(sample.shape[0]),
                  "independent_sampler_rejected": False})
    report["zero_statevector_probe"] = probe

    if args.instance.exists():
        report["instance_runs"].append(instance_run(args.instance, seconds=args.seconds,
                                                    seed=args.seed))
    else:
        report["instance_runs"].append({"error": f"missing instance {args.instance}"})

    if args.report is not None:
        args.report = Path(args.report)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps({"pool_preparation": [{k: row[k] for k in
                                            ("machines", "product_space_states",
                                             "uniform_seconds", "legacy_preparation_seconds")}
                                           for row in pool_rows],
                      "zero_statevector_probe": report["zero_statevector_probe"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
