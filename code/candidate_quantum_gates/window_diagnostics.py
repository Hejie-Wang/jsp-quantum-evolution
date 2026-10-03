#!/usr/bin/env python3
"""T01 pilot: frozen-window pool coverage and joint-improvement diagnostics.

Implements the T01 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md as an offline, read-only
diagnostic layer.  For a frozen candidate pool Pi = (Pi_1, ..., Pi_M) with an
incumbent label a0 and verified upper bound U0 = C(a0) it computes:

  - C_Pi*      exact pool-inner optimum by full enumeration (windows with
               prod(K_m) <= --max-combos) using an independent graph checker;
  - rho        pool coverage card{a feasible: C(a) < U0} / prod(K_m);
  - d_imp(a0)  min number of machines whose label must change to reach any
               improving combination; None when the pool has none;
  - swap graph from a0 over feasible states, one machine re-labelled at a
               time: whether the improvement set is reachable, and the minimal
               bottleneck makespan along any such path (min over improving
               paths of the max C on the path, endpoints included), reported
               absolutely and as relative excess over U0; unreachable -> null;
  - MILP cross-check of the pool optimum through the existing certified
    pool_optimum_milp solver (T01 acceptance: enumeration == MILP optimal).

Full enumeration is diagnostic-only; its answers must never be fed to any
searched algorithm.  This module imports production code but never modifies
adaptive_search.py, compact_simulator.py or any other production file.

Two data sources:

  --demo          D0-style generated instances (per PR #18 section 3.1:
                  durations uniform 1..9, each job a random machine
                  permutation; generator, numpy version, seed and instance
                  hash recorded).  A full-permutation pool yields the exact
                  instance optimum C*; restricted pools (serial-SGS incumbent
                  as candidate 0 plus random permutations) get the complete
                  battery, including the pool-bound-vs-global-bound check.
  --pool-report   a saved report with instance_data + pool_data + best_choice
                  (code/candidate_quantum_medium/results/corrected_20261001/).
                  Those pools are 8^15, far beyond enumeration, so only the
                  one-swap/two-swap neighbourhood of the incumbent plus the
                  MILP pool optimum (status recorded; UNKNOWN claims nothing)
                  are computed there.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
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

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Independent graph checker.  Deliberately implemented separately from
# medium_qjsp.decode and from the batch evaluator; agreement with the
# production decoder is asserted at every window.
# ---------------------------------------------------------------------------

def check_choice(inst: mq.Instance, pool, choice) -> dict:
    """Feasibility and makespan of one combination via Kahn's algorithm.

    Job chains are the fixed job arcs; machine chains follow the selected
    candidate order on each machine.
    """
    n = inst.operations
    row_pos = {}
    for m, a in enumerate(choice):
        row = pool[m][a]
        for k, v in enumerate(row):
            row_pos[v] = (m, k)
    deg = [0] * n
    head = [0] * n
    adj = [[] for _ in range(n)]
    for u in range(n):
        if u % inst.machines < inst.machines - 1:
            adj[u].append(u + 1)          # fixed job chain
            deg[u + 1] += 1
        m, k = row_pos[u]
        row = pool[m][choice[m]]
        if k + 1 < len(row):
            adj[u].append(row[k + 1])     # chosen machine chain
            deg[row[k + 1]] += 1
    makespan = 0
    processed = 0
    queue = [v for v in range(n) if deg[v] == 0]
    while queue:
        nxt = []
        for u in queue:
            processed += 1
            finish = head[u] + inst.processing[u]
            if finish > makespan:
                makespan = finish
            for v in adj[u]:
                if finish > head[v]:
                    head[v] = finish
                deg[v] -= 1
                if deg[v] == 0:
                    nxt.append(v)
        queue = nxt
    if processed != n:
        return {"feasible": False, "makespan": None}
    return {"feasible": True, "makespan": makespan}


def all_choices(sizes):
    return itertools.product(*(range(k) for k in sizes))


def enumerate_pool(inst: mq.Instance, pool, u0=None):
    """Evaluate every combination of the pool.  prod(K_m) must be manageable."""
    sizes = [len(p) for p in pool]
    total = int(np.prod(sizes))
    makespans = []
    feasible = 0
    for choice in all_choices(sizes):
        r = check_choice(inst, pool, choice)
        makespans.append(r["makespan"])
        feasible += r["feasible"]
    values = [m for m in makespans if m is not None]
    best = min(values) if values else None
    improving = None
    rho = None
    if u0 is not None:
        improving = sum(1 for m in values if m < u0)
        rho = improving / total if total else None
    return {
        "total_combinations": total,
        "feasible_combinations": feasible,
        "feasible_rate": feasible / total if total else None,
        "pool_optimum": best,
        "u0": u0,
        "improving_combinations": improving,
        "rho": rho,
        "makespans": makespans,
    }


def d_imp_from_incumbent(incumbent, sizes, u0, makespans):
    """Min machines changed to reach any C < u0; None if no improvement exists."""
    best = None
    for c, m in zip(all_choices(sizes), makespans):
        if m is None or m >= u0:
            continue
        d = sum(1 for x, y in zip(incumbent, c) if x != y)
        if best is None or d < best:
            best = d
            if best <= 1:
                break
    return best


def swap_reachability(pool, incumbent, u0, makespans):
    """One-swap feasible-state graph from the incumbent (Dijkstra on the
    bottleneck makespan).  Returns reachability of the improvement set and the
    minimal bottleneck along any improving path, absolute and relative."""
    sizes = [len(p) for p in pool]
    index = {c: i for i, c in enumerate(all_choices(sizes))}
    states = list(all_choices(sizes))
    start = index[tuple(incumbent)]
    if makespans[start] is None:
        raise ValueError("incumbent must be feasible")
    dist = {start: makespans[start]}
    heap = [(makespans[start], start)]
    while heap:
        b, u = heapq.heappop(heap)
        if b != dist.get(u):
            continue
        cu = makespans[u]
        if u != start and cu is not None and cu < u0:
            return {"improvement_reachable": True,
                    "bottleneck_makespan": b,
                    "relative_excess": (b - u0) / u0}
        state = list(states[u])
        for m in range(len(sizes)):
            old = state[m]
            for a in range(sizes[m]):
                if a == old:
                    continue
                state[m] = a
                v = index[tuple(state)]
                cv = makespans[v]
                if cv is None:
                    continue
                nb = max(b, cv)
                if nb < dist.get(v, 1 << 62):
                    dist[v] = nb
                    heapq.heappush(heap, (nb, v))
            state[m] = old
    return {"improvement_reachable": False, "bottleneck_makespan": None,
            "relative_excess": None}


def milp_pool_optimum(inst_mq: mq.Instance, pool, u_bound, time_limit=300.0):
    """Cross-check through the existing certified one-hot MILP diagnostic."""
    import pool_optimum_milp
    return pool_optimum_milp.solve_pool_optimum_milp(
        inst_mq, pool, u_bound, time_limit=time_limit)


# ---------------------------------------------------------------------------
# D0-style generation and restricted windows (PR #18 section 3.1 rules)
# ---------------------------------------------------------------------------

def generate_d0(jobs: int, machines: int, seed: int) -> mq.Instance:
    """Durations uniform 1..9; each job visits machines in a random order."""
    rng = np.random.default_rng(seed)
    durations = rng.integers(1, 10, size=(jobs, machines))
    routes = np.array([rng.permutation(machines) for _ in range(jobs)])
    return mq.Instance(durations, routes, name=f"d0_{jobs}x{machines}_s{seed}")


def instance_hash(inst: mq.Instance) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(inst.durations, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(inst.routes, dtype=np.int64).tobytes())
    return h.hexdigest()


def build_window(inst: mq.Instance, k: int, seed: int):
    """Restricted pool: the serial-SGS (mwkr) incumbent order as candidate 0
    on every machine plus k-1 random permutations; a0 = all-candidate-0.
    Returns (pool, incumbent, u0) with u0 verified by both checkers."""
    starts, _ = mq.serial_sgs(inst, "mwr", np.random.default_rng(seed))
    rng = np.random.default_rng(1000 + seed)
    pool = []
    for m in range(inst.machines):
        inc = tuple(sorted(inst.groups[m], key=lambda v: starts[v]))
        perms = [tuple(p) for p in itertools.permutations(inst.groups[m])
                 if tuple(p) != inc]
        rng.shuffle(perms)
        pool.append([inc] + perms[:k - 1])
    pool = tuple(tuple(c) for c in pool)
    mq.validate_pool(inst, pool)
    incumbent = tuple(0 for _ in range(inst.machines))
    mine = check_choice(inst, pool, incumbent)
    prod = mq.decode(inst, [pool[m][0] for m in range(inst.machines)])
    if not mine["feasible"] or not prod["feasible"] \
            or mine["makespan"] != prod["makespan"]:
        raise RuntimeError("checkers disagree on the SGS incumbent")
    return pool, incumbent, mine["makespan"]


# ---------------------------------------------------------------------------
# Window-level batteries
# ---------------------------------------------------------------------------

def diagnose_enumerable_window(inst, pool, incumbent, u0, time_limit):
    sizes = [len(p) for p in pool]
    t0 = time.perf_counter()
    enum = enumerate_pool(inst, pool, u0=u0)
    makespans = enum.pop("makespans")
    enum["enumeration_seconds"] = time.perf_counter() - t0
    enum["d_imp_from_incumbent"] = d_imp_from_incumbent(
        incumbent, sizes, u0, makespans)
    t0 = time.perf_counter()
    swap = swap_reachability(pool, incumbent, u0, makespans)
    swap["seconds"] = time.perf_counter() - t0
    milp = None
    if enum["feasible_combinations"]:
        t0 = time.perf_counter()
        milp = milp_pool_optimum(inst, pool, enum["pool_optimum"],
                                 time_limit=time_limit)
        milp["seconds"] = time.perf_counter() - t0
        milp["matches_enumeration"] = (
            milp.get("status") == "optimal"
            and int(round(float(milp.get("objective", -1))))
            == enum["pool_optimum"])
    return {
        "total_combinations": enum["total_combinations"],
        "incumbent": list(incumbent),
        "u0": u0,
        "enumeration": enum,
        "swap_graph": swap,
        "milp_cross_check": milp,
        "acceptance_enumeration_equals_milp": (
            milp["matches_enumeration"] if milp else None),
    }


def diagnose_large_pool_report(path: Path, max_swap: int, time_limit):
    """D1-style saved pool report: incumbent neighbourhood + MILP only."""
    report = json.loads(path.read_text(encoding="utf-8"))
    inst = mq.Instance(report["instance_data"]["durations"],
                       report["instance_data"]["routes_zero_based"],
                       name=str(report.get("instance", path.stem)))
    pool = tuple(tuple(tuple(int(v) for v in order) for order in machine)
                 for machine in report["pool_data"])
    mq.validate_pool(inst, pool)
    incumbent = tuple(int(a) for a in report["best_choice"])
    u0 = int(report["best_makespan"])
    mine = check_choice(inst, pool, incumbent)
    prod = mq.decode(inst, [pool[m][incumbent[m]] for m in range(inst.machines)])
    sizes = [len(p) for p in pool]
    total = int(np.prod(sizes))
    out = {
        "source": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "instance": report.get("instance"),
        "instance_sha256_report": report.get("instance_sha256"),
        "pool_sizes": sizes,
        "total_combinations": total,
        "full_enumeration_feasible": total <= 200_000,
        "incumbent": list(incumbent),
        "u0_reported": u0,
        "u0_independent_check": mine,
        "u0_production_decoder_makespan": prod.get("makespan"),
        "checkers_agree": bool(mine["feasible"])
        and mine["makespan"] == prod.get("makespan"),
    }
    neigh = {}
    for swap_count in range(1, max_swap + 1):
        combos = []
        for machines_to_change in itertools.combinations(range(len(sizes)),
                                                         swap_count):
            alternatives = [[a for a in range(sizes[m]) if a != incumbent[m]]
                            for m in machines_to_change]
            for repl in itertools.product(*alternatives):
                c = list(incumbent)
                for m, a in zip(machines_to_change, repl):
                    c[m] = a
                combos.append(tuple(c))
        improving = 0
        feasible = 0
        best = None
        t0 = time.perf_counter()
        for c in combos:
            r = check_choice(inst, pool, c)
            if r["feasible"]:
                feasible += 1
                m = r["makespan"]
                if best is None or m < best:
                    best = m
                if m < u0:
                    improving += 1
        neigh[str(swap_count)] = {
            "neighbourhood": f"{swap_count}-machine swap",
            "explored": len(combos),
            "feasible": feasible,
            "improving": improving,
            "best_found": best,
            "best_relative_improvement": (u0 - best) / u0 if best else 0.0,
            "seconds": time.perf_counter() - t0,
        }
    out["incumbent_neighbourhood"] = neigh
    t0 = time.perf_counter()
    milp = milp_pool_optimum(inst, pool, u0, time_limit=time_limit)
    milp["seconds"] = time.perf_counter() - t0
    out["milp_pool_optimum"] = {
        "status": milp.get("status"),
        "objective": milp.get("objective"),
        "dual_bound": milp.get("mip_dual_bound"),
        "seconds": milp.get("seconds"),
    }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_demo(scales, seeds, k_values, time_limit):
    rows = []
    for jobs, machines in scales:
        for seed in seeds:
            inst = generate_d0(jobs, machines, seed)
            base = {
                "scale": f"{jobs}x{machines}",
                "seed": seed,
                "generator": "numpy default_rng(seed); durations uniform 1..9; "
                             "per-job random machine permutation",
                "numpy_version": np.__version__,
                "instance_sha256": instance_hash(inst),
            }
            # full-permutation pool = exact instance optimum C*
            full_pool = tuple(
                tuple(itertools.permutations(inst.groups[m]))
                for m in range(inst.machines))
            first = tuple(0 for _ in range(inst.machines))
            u0_full = check_choice(inst, full_pool, first)["makespan"]
            full = enumerate_pool(inst, full_pool, u0=u0_full)
            full.pop("makespans")
            c_star = full["pool_optimum"]
            rows.append({**base, "window": "full_permutations",
                         "pool_sizes": [len(p) for p in full_pool],
                         "instance_optimum": c_star,
                         "feasible_rate": full["feasible_rate"],
                         "note": "full-permutation pool optimum equals the "
                                 "instance optimum by construction"})
            for k in k_values:
                pool, incumbent, u0 = build_window(inst, k, seed)
                diag = diagnose_enumerable_window(inst, pool, incumbent, u0,
                                                  time_limit)
                rows.append({**base, "window": f"restricted_k{k}", "k": k,
                             "pool_sizes": [len(p) for p in pool],
                             "total_combinations": diag["total_combinations"],
                             "u0": u0,
                             "pool_optimum":
                                 diag["enumeration"]["pool_optimum"],
                             "rho": diag["enumeration"]["rho"],
                             "improving_combinations":
                                 diag["enumeration"]["improving_combinations"],
                             "d_imp_from_incumbent":
                                 diag["enumeration"]["d_imp_from_incumbent"],
                             "swap_graph": diag["swap_graph"],
                             "milp_matches_enumeration":
                                 diag["acceptance_enumeration_equals_milp"],
                             "global_vs_pool_bound": {
                                 "instance_optimum": c_star,
                                 "pool_optimum":
                                     diag["enumeration"]["pool_optimum"],
                                 "pool_bound_is_not_global":
                                     diag["enumeration"]["pool_optimum"] > c_star,
                             }})
    return rows


def parse_seeds(text):
    if text.count("-") == 1:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def main():
    ap = argparse.ArgumentParser(description="T01 pilot window diagnostics")
    ap.add_argument("--demo", action="store_true",
                    help="run the D0-style generated window battery")
    ap.add_argument("--scales", default="2x2,3x3,4x3")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--pool-k", default="2,3")
    ap.add_argument("--pool-report", action="append", default=[],
                    help="saved pool report JSON (repeatable)")
    ap.add_argument("--max-swap", type=int, default=2,
                    help="neighbourhood radius for --pool-report windows")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--out", default=str(GATES_DIR / "results_window_diag_20261003"
                                         / "window_diagnostics.json"))
    args = ap.parse_args()

    result = {
        "schema_version": SCHEMA_VERSION,
        "work_package": "T01 pilot (Issue #19, PR #18 docs TODO)",
        "numpy_version": np.__version__,
        "windows": [],
    }
    if args.demo:
        scales = [tuple(map(int, s.split("x"))) for s in args.scales.split(",")]
        result["windows"].extend(
            run_demo(scales, parse_seeds(args.seeds),
                     [int(k) for k in args.pool_k.split(",")],
                     args.time_limit))
    for raw in args.pool_report:
        result["windows"].append(
            diagnose_large_pool_report(Path(raw), args.max_swap, args.time_limit))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    demo = [w for w in result["windows"] if "milp_matches_enumeration" in w]
    large = [w for w in result["windows"] if "milp_pool_optimum" in w]
    print(f"wrote {out}")
    print(f"demo windows: {len(demo)}; milp==enumeration: "
          f"{sum(1 for w in demo if w['milp_matches_enumeration'])}/{len(demo)}")
    print(f"large windows: {len(large)}; checkers agree: "
          f"{sum(1 for w in large if w['checkers_agree'])}/{len(large)}")


if __name__ == "__main__":
    main()
