#!/usr/bin/env python3
"""D2 frozen-window adapter (T04 output -> T06/T10 arms).

T04 (`window_pool_builder.py`, Issue #19) publishes frozen D2 snapshots: for each
instance, seed, trajectory iteration and pool strategy it stores the complete
per-machine candidate pool, the incumbent labels, and the snapshot's raw witness
set, plus an offline evaluation block with the pool optimum and rho.  This module
turns those records into exactly the interface the T06/T10 arms already consume,
so the confirmation-stage runs do not need a second window format:

    (inst, pool, incumbent_labels, u0, target, witnesses)

Witness handling follows the T03/T05 discipline: the raw relations of the
snapshot are re-projected onto *this* pool with ``mq.witness_support`` (never a
stale mask), constant-false witnesses are dropped, and the dropped count is
reported so a pool change cannot silently keep a dead witness.

Only the offline evaluation block is read for the target; the arms never see the
enumeration answers themselves (``improving_combinations`` and ``d_imp`` are
diagnostics, not arm inputs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATES_DIR = Path(__file__).resolve().parent
MEDIUM_DIR = ROOT / "code" / "candidate_quantum_medium"
for _p in (str(GATES_DIR), str(MEDIUM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import medium_qjsp as mq  # noqa: E402
from window_diagnostics import check_choice  # noqa: E402

SCHEMA_VERSION = 1


def load_dataset(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    d2 = payload.get("d2") or {}
    if "windows" not in d2 or "evaluations" not in d2:
        raise ValueError("not a T04 D2 dataset: missing d2.windows / d2.evaluations")
    return payload


def _snapshot_key(instance, seed, iteration):
    return f"{instance}|s{seed}|it{iteration}"


def _load_instance(instance_name, task_data_dir=None):
    directory = Path(task_data_dir) if task_data_dir else ROOT / "task_data"
    path = directory / f"{instance_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"instance file for D2 window not found: {path}")
    return mq.Instance.read(path)


def project_witnesses(pool, raw_witnesses):
    """Re-project raw D2 witness relations onto this pool (T03/T05 discipline)."""
    kept, dropped = [], 0
    for raw in raw_witnesses or []:
        witness = mq.Witness(raw["kind"],
                             tuple(tuple(int(v) for v in r) for r in raw["relations"]),
                             int(raw.get("length", 0)), ())
        projected = mq.witness_support(witness, pool)
        if projected is None:
            dropped += 1
            continue
        support, allowed = projected
        kept.append({"kind": witness.kind,
                     "relations": [tuple(int(v) for v in r) for r in witness.relations],
                     "length": int(witness.length),
                     "support": [int(m) for m in support],
                     "allowed": {int(m): [int(a) for a in allowed[m]] for m in allowed}})
    return kept, dropped


def _load_witness_tables(path, d2):
    """Witness dedup table (T04 compact freeze); legacy sidecar supported."""
    tables = {t["trajectory_key"]: t["witnesses"]
              for t in d2.get("witness_table", [])}
    if tables:
        return tables
    sidecar = Path(path).with_name(Path(path).stem + "_witnesses.json.gz")
    if sidecar.exists():
        import gzip
        with gzip.open(sidecar, "rt", encoding="utf-8") as fh:
            return {t["trajectory_key"]: t["witnesses"] for t in json.load(fh)}
    return {}


def _snapshot_witnesses(snapshot, tables):
    """Resolve a snapshot's raw witnesses in either freeze format.

    Legacy format: witnesses embedded per snapshot.  Compact format (T04
    PR #40): per-trajectory dedup table + per-snapshot ``witness_indices``
    with flattened relations.
    """
    if snapshot is None:
        return []
    if "witnesses" in snapshot:
        return snapshot["witnesses"]
    table = tables.get(snapshot.get("trajectory_key"), [])
    out = []
    for index in snapshot.get("witness_indices", []):
        w = table[index]
        flat = w["relations_flat"]
        out.append({"kind": w["kind"], "length": w["length"],
                    "relations": [flat[i:i + 3]
                                  for i in range(0, len(flat), 3)]})
    return out


def iter_windows(dataset_path, *, only_improving=True, limit=None,
                 task_data_dir=None, instance_cache=None):
    """Yield dicts consumed by the T06/T10 arms.

    Each yielded record carries the frozen pool, the incumbent labels, the
    offline target (the pool optimum) and the snapshot's re-projected witness
    set, plus the T04 metadata needed to trace the run back to its snapshot.
    """
    payload = load_dataset(dataset_path)
    d2 = payload["d2"]
    evaluations = {(e["instance"], e["seed"], e["strategy"],
                    e["trajectory_iteration"]): e for e in d2["evaluations"]}
    snapshots = {s["snapshot_key"]: s for s in d2["snapshots"]}
    witness_tables = _load_witness_tables(Path(dataset_path), d2)
    cache = instance_cache if instance_cache is not None else {}
    produced = 0
    for window in d2["windows"]:
        key = (window["instance"], window["seed"], window["strategy"],
               window["trajectory_iteration"])
        evaluation = evaluations.get(key)
        if evaluation is None:
            continue
        if only_improving and evaluation.get("no_improvement", True):
            continue
        instance_name = window["instance"]
        if instance_name not in cache:
            cache[instance_name] = _load_instance(instance_name, task_data_dir)
        inst = cache[instance_name]
        pool = tuple(tuple(tuple(int(v) for v in order) for order in machine)
                     for machine in window["pool"])
        mq.validate_pool(inst, pool)
        incumbent = tuple(int(a) for a in window["incumbent"])
        u0 = check_choice(inst, pool, incumbent)["makespan"]
        if u0 is None:
            continue  # a frozen incumbent that is not feasible cannot seed an arm
        snapshot = snapshots.get(_snapshot_key(instance_name, window["seed"],
                                               window["trajectory_iteration"]))
        witnesses, dropped = project_witnesses(
            pool, _snapshot_witnesses(snapshot, witness_tables))
        target = evaluation.get("pool_optimum")
        produced += 1
        yield {
            "dataset": "D2", "source": str(dataset_path),
            "instance": instance_name, "inst": inst, "pool": pool,
            "incumbent": incumbent, "u0": int(u0),
            "target": int(target) if target is not None else int(u0) - 1,
            "pool_optimum": target,
            "rho": evaluation.get("rho"),
            "improving_combinations": evaluation.get("improving_combinations"),
            "d_imp_from_incumbent": evaluation.get("d_imp_from_incumbent"),
            "seed": int(window["seed"]), "strategy": window["strategy"],
            "trajectory_iteration": int(window["trajectory_iteration"]),
            "pool_sizes": [len(machine) for machine in pool],
            "pool_sha256": window.get("pool_sha256") or mq.content_hash(pool),
            "instance_sha256": window.get("instance_sha256"),
            "witnesses": witnesses,
            "witnesses_dropped_constant_false": dropped,
        }
        if limit is not None and produced >= limit:
            return


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="D2 frozen-window adapter (T04 -> T06/T10)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--only-improving", action="store_true", default=True)
    ap.add_argument("--all-windows", action="store_true",
                    help="also yield windows whose pool has no improving combination")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    rows = list(iter_windows(args.dataset, only_improving=not args.all_windows,
                             limit=args.limit))
    print(f"windows: {len(rows)}")
    for row in rows[:10]:
        print(f"  {row['instance']} s{row['seed']} {row['strategy']} "
              f"it{row['trajectory_iteration']}: pool={row['pool_sizes']} "
              f"u0={row['u0']} target={row['target']} rho={row['rho']} "
              f"d_imp={row['d_imp_from_incumbent']} "
              f"witnesses={len(row['witnesses'])} "
              f"(dropped {row['witnesses_dropped_constant_false']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
