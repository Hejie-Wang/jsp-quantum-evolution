#!/usr/bin/env python3
"""T00 data commands: build the D0 family and verify it from disk.

    python make_d0_data.py --out-dir data --report reports/d0_verification.json

``build``  writes the frozen D0 instances, their manifest and the exact optimal
           values into a diagnostic file that no search may read.
``verify`` re-reads everything from disk, re-derives each instance hash, and
           fails loudly if a file, a hash or the split does not match the
           manifest.  Exit code 0 means the D0 layer is intact.

The two commands are separated on purpose: T01-T10 consume the files, while
only this script is allowed to (re)generate them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_identity import (DATA_VERSION, D0_SIZES, build_d0, d0_manifest,
                           d0_records, d0_split_of, enumerate_exact_optimum,
                           instance_sha256, load_instance_file, write_text_instance)

#: Enumeration ceiling for the certificate step; 4x3 needs 24^3 = 13,824.
ENUMERATION_LIMIT = 200_000


def build(out_dir: Path, report_path: Path | None) -> dict:
    out_dir = Path(out_dir)
    records = list(d0_records())
    manifest = d0_manifest(records)
    started = perf_counter()
    optima, verification = [], {"files": 0, "hash_matches": 0, "failures": []}
    for jobs, machines, seed, split, inst in records:
        relative = Path(f"{jobs}x{machines}") / f"seed{seed}.txt"
        target = out_dir / relative
        write_text_instance(inst, target)
        verification["files"] += 1
        loaded = load_instance_file(target)
        if instance_sha256(loaded) != instance_sha256(inst):
            verification["failures"].append(f"round-trip hash mismatch: {relative}")
        else:
            verification["hash_matches"] += 1
        certificate = enumerate_exact_optimum(inst, limit=ENUMERATION_LIMIT)
        optima.append({
            "name": inst.name,
            "jobs": int(jobs), "machines": int(machines), "seed": int(seed),
            "split": split,
            "path": relative.as_posix(),
            "instance_sha256": instance_sha256(inst),
            "lower_bound": int(inst.lower_bound),
            **certificate,
        })
    for row in manifest["instances"]:
        relative = Path(f"{row['jobs']}x{row['machines']}") / f"seed{row['seed']}.txt"
        if not (out_dir / relative).exists():
            verification["failures"].append(f"missing data file: {relative}")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostic_dir = out_dir / "window_oracle_outputs"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "schema_version": 1,
        "purpose": ("Exact optimal machine orders for the D0 family. Diagnostic ground truth: "
                    "no sampler, no search arm and no online component may read this file."),
        "enumeration_limit": ENUMERATION_LIMIT,
        "instances": optima,
    }
    (diagnostic_dir / "d0_exact_optima.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema_version": 1,
        "dataset": "D0",
        "data_version": DATA_VERSION,
        "generated_by": "make_d0_data.py build",
        "instance_count": manifest["instance_count"],
        "sizes": [list(size) for size in D0_SIZES],
        "manifest_sha256": manifest["manifest_sha256"],
        "files_written": verification["files"],
        "round_trip_hash_matches": verification["hash_matches"],
        "correctness_gate_passed": not verification["failures"],
        "failures": verification["failures"],
        "exact_optima": [
            {"name": row["name"], "split": row["split"],
             "exact_optimum": row["exact_optimum"],
             "combinations": row["combinations"],
             "feasible_combinations": row["feasible_combinations"],
             "independent_verification": row["independent_verification"]}
            for row in optima
        ],
        "seconds": perf_counter() - started,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return report


def verify(out_dir: Path, report_path: Path | None = None) -> tuple[bool, dict]:
    out_dir = Path(out_dir)
    checks, failures = [], []

    def record(name: str, passed: bool, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            failures.append(name)

    manifest_path = out_dir / "manifest.json"
    record("manifest exists", manifest_path.exists())
    if not manifest_path.exists():
        return False, {"checks": checks, "failures": failures}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["instances"]
    record("data version", manifest.get("data_version") == DATA_VERSION,
           manifest.get("data_version"))
    record("instance count", manifest.get("instance_count") == len(rows), len(rows))
    record("sizes covered", {(row["jobs"], row["machines"]) for row in rows} == set(D0_SIZES))
    record("splits valid", all(row["split"] == d0_split_of(row["seed"]) for row in rows))

    rebuilt = []
    for row in rows:
        relative = Path(f"{row['jobs']}x{row['machines']}") / f"seed{row['seed']}.txt"
        path = out_dir / relative
        if not path.exists():
            record(f"file {relative}", False)
            continue
        loaded = load_instance_file(path)
        rebuilt.append((row, loaded))
    record("all files present", len(rebuilt) == len(rows), f"{len(rebuilt)}/{len(rows)}")
    hash_ok = all(row["instance_sha256"] == instance_sha256(loaded) for row, loaded in rebuilt)
    record("file hashes match manifest", hash_ok)
    regenerated_ok = all(
        instance_sha256(build_d0(row["jobs"], row["machines"], row["seed"])) == row["instance_sha256"]
        for row, _ in rebuilt)
    record("instances regenerate identically from the frozen seed", regenerated_ok)
    contents_ok = all(
        row["durations"] == loaded.durations.tolist() and row["routes"] == loaded.routes.tolist()
        for row, loaded in rebuilt)
    record("durations and routes match manifest", contents_ok)
    diagnostic = out_dir / "window_oracle_outputs" / "d0_exact_optima.json"
    record("exact-optimum diagnostic exists", diagnostic.exists())
    if diagnostic.exists():
        rows_out = json.loads(diagnostic.read_text(encoding="utf-8"))["instances"]
        record("diagnostic covers the family", len(rows_out) == len(rows), len(rows_out))
        record("diagnostic is independently verified",
               all(row["independent_verification"] for row in rows_out))
        record("optima respect the trivial lower bound",
               all(row["exact_optimum"] >= row["lower_bound"] for row in rows_out))
    result = {
        "schema_version": 1,
        "dataset": "D0",
        "data_version": DATA_VERSION,
        "verified_by": "make_d0_data.py verify",
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "correctness_gate": "100% required by the T00 acceptance rules",
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return not failures, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "build":
        report = build(args.out_dir, args.report)
        print(json.dumps({key: report[key] for key in
                          ("instance_count", "files_written", "round_trip_hash_matches",
                           "correctness_gate_passed", "seconds")}, ensure_ascii=False))
        return 0 if report["correctness_gate_passed"] else 1
    passed, result = verify(args.out_dir, args.report)
    print(json.dumps({"passed": passed, "checks": len(result["checks"]),
                      "failures": result["failures"]}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
