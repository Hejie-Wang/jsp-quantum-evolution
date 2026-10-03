#!/usr/bin/env python3
"""T03: witness correctness and makespan-consistent surrogate targets.

Implements the T03 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md as a read-only
diagnostic layer over frozen pools.  It reuses the production projection
primitive ``medium_qjsp.witness_support`` but independently re-verifies every
acceptance property with the ``window_diagnostics.check_choice`` graph
checker, so the correctness claims never rest on the decoder alone.

Surrogates compared on frozen pools (PR #18 section 5, T03):

  H_count   violation count  sum_W P_W(a)            (all witnesses)
  H_weight  path-severity weighted  sum_W L_W * P_W(a) (path witnesses)
  l_W       path-max surrogate  max(L0, max_{path W} L_W * P_W(a))

For every acyclic combination the battery enforces l_W(a) <= C(a); a
triggered cycle witness must imply an infeasible combination; a triggered
path witness on an acyclic combination must imply C(a) >= L_W; and the
surrogate used as a screen must exclude no combination that actually meets
the target ("zero false exclusions").  Screening quality of the lowest-10%
energy states (precision / recall for true improvements, ties reported
separately) and Kendall-tau ranking error against the true makespan close
the item left blank by the T01 pilot.

Scope discipline: l_W at a point is a POOL-side quantity, never a global
bound; summing path lengths is explicitly not done.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GATES_DIR = Path(__file__).resolve().parent
MEDIUM_DIR = ROOT / "code" / "candidate_quantum_medium"
for _p in (str(GATES_DIR), str(MEDIUM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import medium_qjsp as mq  # noqa: E402
from window_diagnostics import check_choice  # noqa: E402  (independent truth)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Witness collection, dedup and projection onto a frozen pool
# ---------------------------------------------------------------------------

def collect_witnesses(inst: mq.Instance, pool, incumbent, n_probe=16, seed=0):
    """Build a frozen witness set from the pool itself.

    Probes the incumbent plus random combinations, compiles each decode into
    a witness (mq.separate), dedups raw relations across witnesses, drops
    witnesses that can never activate inside this pool (constant-false) and
    records constant-true ones explicitly.  The frozen set is then what all
    surrogate targets use; nothing here is re-derived per combination.
    """
    rng = np.random.default_rng(seed)
    sizes = [len(p) for p in pool]
    probes = [tuple(incumbent)]
    for _ in range(n_probe):
        probes.append(tuple(int(rng.integers(k)) for k in sizes))
    raw = {}
    for choice in probes:
        orders = [pool[m][a] for m, a in enumerate(choice)]
        decoded = mq.decode(inst, orders, validate=False)
        w = mq.separate(inst, decoded)
        raw.setdefault(w.key(), w)
    witnesses = []
    stats = {"raw": len(raw), "constant_false_dropped": 0,
             "constant_true": 0, "kept": 0}
    for w in raw.values():
        proj = mq.witness_support(w, pool)
        if proj is None:
            stats["constant_false_dropped"] += 1
            continue
        support, allowed = proj
        if all(len(allowed[m]) == len(pool[m]) for m in allowed):
            stats["constant_true"] += 1
        witnesses.append({
            "kind": w.kind,
            "relations": [tuple(int(x) for x in r) for r in w.relations],
            "length": int(w.length),
            "support": [int(m) for m in support],
            "allowed": {int(m): [int(a) for a in allowed[m]]
                        for m in allowed},
        })
        stats["kept"] += 1
    return witnesses, stats


def witness_activates(w, choice) -> bool:
    """P_W(a) == 1 under the projected allowed-label sets."""
    return all(choice[m] in w["allowed"][m] for m in w["allowed"])


def surrogates(inst: mq.Instance, pool, witnesses, choice, l0=None):
    """(H_count, H_weight, l_W, feasible_by_witnesses) for one combination.

    H_weight sums L_W * P_W over PATH witnesses only; cycle witnesses enter
    H_count and the feasibility flag, never the weighted path term.  l_W is
    max(L0, max over path witnesses of L_W * P_W) exactly as specified.
    """
    if l0 is None:
        l0 = int(inst.lower_bound)
    h_count = 0
    h_weight = 0
    path_max = 0
    feasible = True
    for w in witnesses:
        if not witness_activates(w, choice):
            continue
        h_count += 1
        if w["kind"] == "cycle":
            feasible = False
        else:
            h_weight += w["length"]
            path_max = max(path_max, w["length"])
    return {"H_count": h_count, "H_weight": h_weight,
            "l_w": max(l0, path_max), "cycle_triggered": not feasible}


# ---------------------------------------------------------------------------
# Correctness battery over enumerated (D0) or sampled (D1) combinations
# ---------------------------------------------------------------------------

def correctness_battery(inst: mq.Instance, pool, witnesses, choices, u0,
                        target=None):
    """Full T03 acceptance battery.  Returns per-property violation counts
    (all must be zero) plus the screening/ranking report."""
    l0 = int(inst.lower_bound)
    checks = {"cycle_witness_without_cycle": 0,
              "path_witness_below_length": 0,
              "l_w_exceeds_makespan": 0,
              "false_exclusions": 0}
    rows = []
    t0 = time.perf_counter()
    for choice in choices:
        r = check_choice(inst, pool, choice)
        c = r["makespan"]
        s = surrogates(inst, pool, witnesses, choice, l0=l0)
        if s["cycle_triggered"] and r["feasible"]:
            checks["cycle_witness_without_cycle"] += 1
        if r["feasible"]:
            for w in witnesses:
                if w["kind"] == "path" and witness_activates(w, choice) \
                        and c < w["length"]:
                    checks["path_witness_below_length"] += 1
            if s["l_w"] > c:
                checks["l_w_exceeds_makespan"] += 1
        # zero false exclusions: anything meeting the target must survive
        # the surrogate screen (no triggered cycle, l_w <= target)
        if target is not None and c is not None and c <= target:
            if s["cycle_triggered"] or s["l_w"] > target:
                checks["false_exclusions"] += 1
        rows.append({"choice": choice, "makespan": c,
                     "H_count": s["H_count"], "H_weight": s["H_weight"],
                     "l_w": s["l_w"], "cycle": s["cycle_triggered"]})
    seconds = time.perf_counter() - t0

    # screening quality of the lowest-10%-energy states, per surrogate
    feasible_rows = [x for x in rows if isinstance(x["makespan"], int)]
    improving = [x for x in feasible_rows if x["makespan"] < u0]
    screening = {}
    for key in ("H_count", "H_weight", "l_w"):
        screening[key] = _screening_report(feasible_rows, improving, key, u0)
    return {"checks": checks, "screening": screening, "rows": rows,
            "seconds": seconds,
            "n_feasible": len(feasible_rows), "n_improving": len(improving)}


def _screening_report(rows, improving, key, u0, fraction=0.1):
    """Lowest-`fraction` energy states: precision/recall for true improves.

    Ties at the cut point are handled two ways and both are reported:
    `strict` takes exactly floor(n*fraction) after a stable sort;
    `with_ties` takes every state whose energy equals the strict cut value.
    """
    if not improving:
        return {"skipped": "no improving combinations in window"}
    ordered = sorted(rows, key=lambda x: (x[key], x["makespan"]))
    k = max(1, int(len(rows) * fraction))
    strict = ordered[:k]
    cut = strict[-1][key]
    with_ties = [x for x in rows if x[key] < cut] + \
                [x for x in ordered[len([x for x in rows if x[key] < cut]):]
                 if x[key] == cut]
    def pr(subset):
        hits = sum(1 for x in subset if x["makespan"] < u0)
        n = max(1, len(subset))
        return {"precision": hits / n, "recall": hits / len(improving),
                "selected": len(subset), "hits": hits}
    return {"strict": pr(strict), "with_ties": pr(with_ties),
            "baseline_improving_rate": len(improving) / len(rows)}


def rank_correlation(rows, key="l_w"):
    """Kendall tau between a surrogate and the true makespan over acyclic
    combinations (the T01-pilot metric left blank there)."""
    from scipy.stats import kendalltau
    pts = [(x[key], x["makespan"]) for x in rows
           if isinstance(x["makespan"], int) and not x["cycle"]]
    if len(pts) < 2:
        return None
    tau = kendalltau([a for a, _ in pts], [b for _, b in pts])
    return {"tau": float(tau.statistic), "p_value": float(tau.pvalue),
            "n": len(pts), "surrogate": key}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_d0(seeds, k_values, n_probe, sample_cap, seed_offset=0):
    import data_identity
    out = []
    for jobs, machines in data_identity.D0_SIZES:
        for seed in seeds:
            inst = data_identity.build_d0(jobs, machines, seed)
            c_star = data_identity.enumerate_exact_optimum(inst)["exact_optimum"]
            from window_diagnostics import build_window
            pool, incumbent, u0 = build_window(inst, 3, seed + seed_offset)
            witnesses, stats = collect_witnesses(inst, pool, incumbent,
                                                 n_probe=n_probe, seed=seed)
            sizes = [len(p) for p in pool]
            all_choices = list(itertools.product(*(range(k) for k in sizes)))
            battery = correctness_battery(inst, pool, witnesses, all_choices,
                                          u0, target=u0)
            rows = [x for x in battery["rows"] if "_seconds" not in x]
            out.append({
                "scale": f"{jobs}x{machines}", "seed": seed,
                "instance_sha256": data_identity.instance_sha256(inst),
                "c_star": c_star, "u0": u0,
                "pool_sizes": sizes,
                "witness_stats": stats,
                "checks": battery["checks"],
                "all_checks_zero": all(v == 0 for v in battery["checks"].values()),
                "screening": battery["screening"],
                "kendall_tau": {k: rank_correlation(rows, k)
                                for k in ("H_count", "H_weight", "l_w")},
            })
    return out


def run_d1(reports, n_samples, seed, n_probe=0):
    """D1 old fixed pools: witnesses from the saved witness_data, correctness
    verified on random combinations (8^15 is beyond enumeration)."""
    rng = np.random.default_rng(seed)
    out = []
    for path in reports:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        inst = mq.Instance(report["instance_data"]["durations"],
                           report["instance_data"]["routes_zero_based"],
                           name=str(report.get("instance", Path(path).stem)))
        pool = tuple(tuple(tuple(int(v) for v in order) for order in machine)
                     for machine in report["pool_data"])
        mq.validate_pool(inst, pool)
        witnesses = []
        for w in report["witness_data"]:
            proj = mq.witness_support(
                mq.Witness(w["kind"], tuple(tuple(r) for r in w["relations"]),
                           int(w["length"]), tuple(w.get("nodes", ()))),
                pool)
            if proj is None:
                continue
            support, allowed = proj
            witnesses.append({
                "kind": w["kind"],
                "relations": [tuple(int(x) for x in r) for r in w["relations"]],
                "length": int(w["length"]),
                "support": [int(m) for m in support],
                "allowed": {int(m): [int(a) for a in allowed[m]]
                            for m in allowed},
            })
        incumbent = tuple(int(a) for a in report["best_choice"])
        u0 = int(report["best_makespan"])
        sizes = [len(p) for p in pool]
        sample = [incumbent]
        for _ in range(n_samples):
            sample.append(tuple(int(rng.integers(k)) for k in sizes))
        battery = correctness_battery(inst, pool, witnesses, sample, u0,
                                      target=u0)
        out.append({
            "source": Path(path).name,
            "instance": report.get("instance"),
            "witnesses_kept": len(witnesses),
            "witnesses_raw": len(report["witness_data"]),
            "u0": u0,
            "sampled": len(sample),
            "checks": battery["checks"],
            "all_checks_zero": all(v == 0 for v in battery["checks"].values()),
            "screening": battery["screening"],
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="T03 witness/surrogate battery")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--pool-k", type=int, default=3)
    ap.add_argument("--d1", action="store_true",
                    help="also run the D1 old-pool sampled battery")
    ap.add_argument("--d1-samples", type=int, default=2000)
    ap.add_argument("--out", default=str(GATES_DIR /
                                         "results_witness_targets_20261003"
                                         / "witness_targets.json"))
    args = ap.parse_args(argv)

    lo, hi = args.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    result = {
        "schema_version": SCHEMA_VERSION,
        "work_package": "T03 (Issue #19, PR #18 docs TODO)",
        "d0": run_d0(seeds, [args.pool_k], n_probe=16, sample_cap=None,
                     seed_offset=1000),
    }
    if args.d1:
        result["d1"] = run_d1(
            sorted(str(p) for p in (ROOT / "code" / "candidate_quantum_medium"
                                    / "results" / "corrected_20261001")
                   .glob("*_milp.json")),
            args.d1_samples, seed=7)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    d0_ok = sum(1 for w in result["d0"] if w["all_checks_zero"])
    print(f"wrote {out}")
    print(f"D0 windows: {d0_ok}/{len(result['d0'])} all checks zero")
    if "d1" in result:
        d1_ok = sum(1 for w in result["d1"] if w["all_checks_zero"])
        print(f"D1 pools:   {d1_ok}/{len(result['d1'])} all checks zero "
              f"({args.d1_samples} samples each)")


if __name__ == "__main__":
    main()
