#!/usr/bin/env python3
"""Minimum run-record schema and unified runner wrapper (T00).

Section 6 of the breakdown document freezes a minimum record per run.  This
module is the single place that builds it, so T00-T10 cannot quietly invent
divergent field names, and every field is either filled from a measured value or
explicitly ``None`` -- a ``None`` here means "not measured in this run", never
"assumed good".

The wrapper adds three things the older ad-hoc result files lacked:

* ``code_sha`` and ``instance_sha256`` on every record, so a number can be tied
  to the exact code and instance that produced it;
* a separation between the *search* budget and the *end to end* wall clock
  including setup/JIT, which the old CUDA-accumulated timing mixed together;
* ``bound_scope``, which keeps a pool bound from being read as a global bound.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

_GATES_DIR = Path(__file__).resolve().parent
if str(_GATES_DIR) not in sys.path:
    sys.path.insert(0, str(_GATES_DIR))

from data_identity import instance_sha256  # noqa: E402

__all__ = [
    "REQUIRED_RECORD_FIELDS",
    "SCHEMA_VERSION",
    "build_record",
    "code_sha",
    "environment_record",
    "run_and_record",
    "validate_record",
]

SCHEMA_VERSION = 1

#: Field set required by the breakdown document, section 6.  ``validate_record``
#: is the enforcement point used by the tests and by downstream tasks.
REQUIRED_RECORD_FIELDS = (
    "schema_version",
    "code_sha",
    "instance_sha256",
    "split",
    "seed",
    "pool_sha256",
    "method",
    "quantum_execution",
    "bound_scope",
    "solver_status",
    "global_lower_bound",
    "pool_lower_bound",
    "verified_upper_bound",
    "bound_evidence",
    "wall_seconds",
    "cost_breakdown",
    "graph_evaluations",
    "raw_counts_path",
    "trace_path",
    "failure_reason",
)

BOUND_SCOPES = ("global", "relaxation", "pool", "none")
QUANTUM_EXECUTIONS = ("none", "simulator", "qpu")


def code_sha(workdir=None) -> str | None:
    """Current commit of the working tree, or ``None`` if git is unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(workdir or _GATES_DIR),
                             capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


def environment_record() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__}


def build_record(result: dict, *, method: str, split: str, trace_path: str | None,
                 code_revision: str | None = None, raw_counts_path: str | None = None,
                 bound_scope: str = "none", global_lower_bound=None,
                 pool_lower_bound=None, bound_evidence=None,
                 failure_reason: str | None = None, extra: dict | None = None) -> dict:
    """Turn one ``adaptive_search.solve`` result into a schema-complete record.

    ``method`` is the measured module (for example ``"tabu"``, ``"uniform"`` or
    ``"witness_xy_joint_simulation"``); it is recorded separately from
    ``quantum_execution`` so a classical simulation is never reported as QPU
    evidence.
    """
    if split not in {"development", "validation", "confirmation", "diagnostic"}:
        raise ValueError(f"unknown split {split!r}")
    if bound_scope not in BOUND_SCOPES:
        raise ValueError(f"unknown bound scope {bound_scope!r}")
    mode = result.get("mode")
    quantum_execution = "simulator" if mode == "quantum" else "none"
    # A verified incumbent is an upper bound on the *instance*, so it is always
    # safe to publish; lower bounds are only published with their scope.
    verified_upper = result.get("best_makespan")
    record = {
        "schema_version": SCHEMA_VERSION,
        "code_sha": code_revision or code_sha(),
        "instance_sha256": result.get("instance_sha256"),
        "instance": result.get("instance"),
        "jobs": result.get("jobs"),
        "machines": result.get("machines"),
        "split": split,
        "seed": result.get("seed"),
        "pool_sha256": None,  # dynamic pools are not yet content-addressed per run
        "method": method,
        "quantum_execution": quantum_execution,
        "bound_scope": bound_scope,
        "solver_status": "completed",
        "global_lower_bound": global_lower_bound,
        "pool_lower_bound": pool_lower_bound,
        "verified_upper_bound": verified_upper,
        "bound_evidence": bound_evidence,
        "wall_seconds": result.get("wall_seconds"),
        "search_seconds": result.get("search_seconds"),
        "setup_seconds": result.get("setup_seconds"),
        "cost_breakdown": {
            "setup_seconds": result.get("setup_seconds"),
            "search_seconds": result.get("search_seconds"),
            "batch_evaluation_seconds": result.get("batch_evaluation_seconds"),
            "proposal_seconds": result.get("proposal_seconds"),
            "training_evaluations": result.get("training_evaluations"),
            "proposal_calls": result.get("proposal_calls"),
        },
        "graph_evaluations": result.get("graph_evaluations"),
        "raw_counts_path": raw_counts_path,
        "trace_path": trace_path,
        "failure_reason": failure_reason,
        "hardware_jobs_submitted": int(result.get("hardware_jobs_submitted", 0)),
        "initial_makespan": result.get("initial_makespan"),
        "independent_schedule_verification": result.get("independent_schedule_verification"),
        "iterations": result.get("iterations"),
        "environment": environment_record(),
    }
    if extra:
        record.update(extra)
    return record


def validate_record(record: dict) -> dict:
    """Check the frozen field set and the scope/backend invariants.

    Raises ``ValueError`` on a missing field, an unknown scope, a QPU claim
    without a hardware job, or a pool-scoped value placed in the global field.
    """
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"record is missing required fields: {missing}")
    if record["bound_scope"] not in BOUND_SCOPES:
        raise ValueError(f"unknown bound scope {record['bound_scope']!r}")
    if record["quantum_execution"] not in QUANTUM_EXECUTIONS:
        raise ValueError(f"unknown quantum execution {record['quantum_execution']!r}")
    if record["quantum_execution"] == "qpu" and not record.get("hardware_jobs_submitted"):
        raise ValueError("a QPU record must carry at least one submitted job")
    if record["bound_scope"] == "pool" and record["global_lower_bound"] is not None:
        raise ValueError("a pool-scoped result must not populate global_lower_bound")
    if not record["independent_schedule_verification"]:
        raise ValueError("only independently verified schedules may be recorded")
    return record


def run_and_record(inst, *, split: str, workdir=None, trace_dir=None, **solve_kwargs) -> dict:
    """Run one search and return ``{"record", "trace", "result"}``.

    The optional ``trace_dir`` writes the per-iteration improvement trace to a
    file and records its path, because the trace is the artefact a reviewer
    replays; it is never embedded only inside the summary number.
    """
    from adaptive_search import solve  # local import keeps the schema import-light

    started = perf_counter()
    result = solve(inst, **solve_kwargs)
    end_to_end = perf_counter() - started
    trace_path = None
    if trace_dir is not None:
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        name = (f"{inst.name}_{solve_kwargs.get('mode', 'classical')}"
                f"_seed{solve_kwargs.get('seed', 0)}.trace.json")
        trace_path = str(trace_dir / name)
        Path(trace_path).write_text(
            json.dumps({"instance_sha256": instance_sha256(inst),
                        "mode": solve_kwargs.get("mode"),
                        "seed": solve_kwargs.get("seed"),
                        "trace": result["trace"]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    method = {"classical": "tabu", "uniform": "independent_uniform",
              "quantum": "witness_xy_joint_simulation"}.get(
                  solve_kwargs.get("mode", "classical"), str(solve_kwargs.get("mode")))
    record = build_record(result, method=method, split=split, trace_path=trace_path,
                          code_revision=code_sha(workdir))
    record["end_to_end_seconds"] = end_to_end
    return {"record": validate_record(record), "trace": result["trace"], "result": result}
