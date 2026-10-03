#!/usr/bin/env python3
"""T04: dynamic candidate windows and local-relevance pool construction.

Implements the T04 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md.  Two strictly separated
layers live here:

ONLINE builder (never reads exact answers)
    ``classical_trajectory`` runs a pure-classical tabu trajectory (the same
    primitives as ``adaptive_search``: Giffler-Thompson restarts, critical-block
    insertion moves, pairwise tabu) and freezes a snapshot at fixed iterations.
    A snapshot carries the incumbent complete orders, its makespan, a bounded
    historical witness set (raw machine precedence relations, the only
    structural feedback the builder sees) and the last evaluated move batch.

    ``build_pool`` turns a snapshot into a candidate pool under a budget
    (s_max active machines, k_max candidates each).  Four strategies:

      random                   uniform machine choice, move-derived alternatives
                               when the machine has any (the production
                               ``refreshed_pool`` control)
      critical_path            machines ranked by critical-block tight pairs
      witness                  machines ranked by how many historical witnesses
                               mention them (multi-witness co-support)
      critical_path_randomfill critical-path machines, random-permutation
                               alternatives only (attribution ablation:
                               machine selection vs candidate quality)

    Every pool keeps the incumbent order as candidate 0 on every machine, and
    machines outside the window are frozen to the incumbent (recorded as fixed
    conditions).  A small predetermined share of alternatives may be random
    permutations that are singly infeasible; only the complete graph may reject
    them, and the diagnostic layer counts exactly that.  The builder consumes
    nothing from any enumeration or MILP oracle.

OFFLINE diagnostic (enumeration allowed, isolated)
    ``evaluate_pool`` enumerates every combination of the frozen pool with the
    independent ``window_diagnostics.check_choice`` Kahn checker and reports
    rho, d_imp, the no-improvement flag and the per-source rejection counts.
    ``reprojection_battery`` re-projects the frozen witnesses onto the new pool
    (``medium_qjsp.witness_support``) and re-verifies every acceptance property
    with an independent allowed-set recomputation and the graph checker:
    triggered cycle witness => infeasible, triggered path witness => C >= L_W,
    and zero false exclusions for combinations meeting the target.

Acceptance gates (PR #18 section 5, T04):
    incumbent retention 100%, candidate completeness 100% (validate_pool),
    witness reprojection correctness 100%, and the coverage gate: a structured
    strategy must reduce the no-improvement pool ratio by >= 20% relative
    against the random control measured on the same frozen states with the
    same checker.  The complete-classical dynamic search must not regress
    against the production classical mainline (adaptive_search.solve,
    classical mode) under the same budget; the benefit is attributed to
    classical candidate design alone.
"""
from __future__ import annotations

import argparse
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
from window_diagnostics import check_choice, enumerate_pool  # noqa: E402

SCHEMA_VERSION = 1
STRATEGIES = ("random", "critical_path", "witness", "critical_path_randomfill")
# Bounded historical evidence, mirroring the production witness cap.
WITNESS_CAP = 96


def _stable_rng(*parts):
    """Deterministic generator from integer parts (no per-process hash salt)."""
    return np.random.default_rng([int(p) for p in parts])


def _strategy_index(strategy):
    return STRATEGIES.index(strategy)


# ---------------------------------------------------------------------------
# Witness memory (raw machine precedence relations, bounded)
# ---------------------------------------------------------------------------

def _remember(inst, witnesses, seen, decoded):
    """Deduplicated bounded witness memory over raw relations."""
    if decoded is None:
        return
    w = mq.separate(inst, decoded)
    key = (w.kind, w.relations, w.length)
    if key in seen:
        return
    seen.add(key)
    witnesses.append({"kind": w.kind,
                      "relations": [tuple(int(x) for x in r) for r in w.relations],
                      "length": int(w.length)})
    if len(witnesses) > WITNESS_CAP:
        old = witnesses.pop(0)
        seen.discard((old["kind"], tuple(map(tuple, old["relations"])),
                      old["length"]))


# ---------------------------------------------------------------------------
# Pure-classical trajectory with frozen checkpoints (the D2 collection layer)
# ---------------------------------------------------------------------------

class Snapshot:
    """A frozen trajectory state: everything the online builder may consume."""

    __slots__ = ("iteration", "orders", "value", "head", "tail", "witnesses",
                 "move_machines", "move_orders", "move_values", "seconds")

    def __init__(self, iteration, orders, value, head, tail, witnesses,
                 move_machines, move_orders, move_values, seconds):
        self.iteration = iteration
        self.orders = orders
        self.value = value
        self.head = head
        self.tail = tail
        self.witnesses = witnesses
        self.move_machines = move_machines
        self.move_orders = move_orders
        self.move_values = move_values
        self.seconds = seconds


def classical_trajectory(inst, seed, checkpoints, *, initial_schedules=240,
                         stagnation=600, verbose=False):
    """Run the classical arm and freeze snapshots at the given iterations.

    Faithful in spirit to ``adaptive_search.solve`` (classical mode): the same
    Giffler-Thompson restarts, critical-block insertion moves and pairwise
    tabu, simplified only in the stagnation-restart bookkeeping.  Deterministic
    under ``seed``; the state is captured before the move of that iteration is
    applied.  Snapshot makespans are cross-checked between the fast evaluator
    and the production decoder.
    """
    from adaptive_search import apply_moves, critical_moves, gt_initial
    from batch_evaluator import BatchEvaluator, INVALID, evaluate_one

    checkpoints = sorted(set(int(c) for c in checkpoints))
    rng = np.random.default_rng(seed)
    evaluator = BatchEvaluator(inst, "cpu")
    processing = evaluator.processing
    routes = np.array(inst.machine_of, dtype=np.int32)
    _, info = mq.build_pool(inst, 8, initial_schedules, seed)
    current = np.asarray(info["orders"], dtype=np.int32).copy()
    decoded = mq.decode(inst, current)
    if not decoded["feasible"]:
        raise ValueError("initial schedule is infeasible")
    current_value = best_value = int(decoded["makespan"])
    evaluator.evaluate(current[None])          # warm the JIT before timing
    gt_initial(processing, routes, inst.jobs, inst.machines, seed)
    tabu, witnesses, seen = {}, [], set()
    _remember(inst, witnesses, seen, decoded)
    snapshots = []
    last_improvement = 0
    started = time.perf_counter()
    move_state = {"machines": [], "batch": None, "values": []}

    def snap(it):
        value, head, tail = evaluate_one(current, processing, inst.machines)
        if value != current_value:
            raise RuntimeError("evaluator and decoder disagree at snapshot")
        machines, orders, values = [], [], []
        if move_state["batch"] is not None:
            for i, m in enumerate(move_state["machines"]):
                if move_state["values"][i] < INVALID:
                    machines.append(int(m))
                    orders.append(tuple(int(v) for v in move_state["batch"][i, m]))
                    values.append(int(move_state["values"][i]))
        # Small perturbation probes harvest extra raw witnesses (independent
        # of the builder; purely historical feedback for the witness strategy).
        probe_rng = _stable_rng(seed, it, 999)
        for _ in range(4):
            m = int(probe_rng.integers(inst.machines))
            if len(inst.groups[m]) < 2:
                continue
            row = current[m].tolist()
            k = int(probe_rng.integers(len(row) - 1))
            row[k], row[k + 1] = row[k + 1], row[k]
            perturbed = current.copy()
            perturbed[m] = row
            _remember(inst, witnesses, seen,
                      mq.decode(inst, perturbed, validate=False))
        snapshots.append(Snapshot(it, current.copy(), current_value,
                                  np.asarray(head), np.asarray(tail),
                                  [dict(w) for w in witnesses],
                                  machines, orders, values,
                                  time.perf_counter() - started))

    iteration = 0
    while True:
        if iteration in checkpoints:
            snap(iteration)
            if verbose:
                print(f"  snapshot seed={seed} iter={iteration} "
                      f"value={current_value} witnesses={len(witnesses)}",
                      file=sys.stderr)
        if iteration == checkpoints[-1]:
            break
        iteration += 1
        current_value, head, tail = evaluate_one(current, processing, inst.machines)
        insertion = iteration % 10 == 0
        moves = critical_moves(current, processing, head, tail, current_value,
                               insertion=insertion)
        if not moves:
            current = gt_initial(processing, routes, inst.jobs, inst.machines,
                                 int(rng.integers(1, 2 ** 30)))
            current_value = evaluate_one(current, processing, inst.machines)[0]
            tabu.clear()
            last_improvement = iteration
            continue
        rng.shuffle(moves)
        batch = apply_moves(current, moves)
        values = evaluator.evaluate(batch, validate=False)
        move_state["machines"] = [m for m, _, _ in moves]
        move_state["batch"] = batch
        move_state["values"] = values
        chosen, score = -1, INVALID
        for i, (m, a, b) in enumerate(moves):
            value = int(values[i])
            if value >= INVALID:
                continue
            u = int(current[m, a])
            crossed = current[m, a + 1:b + 1] if a < b else current[m, b:a]
            created = [(int(v), u) if a < b else (u, int(v)) for v in crossed]
            if any(tabu.get(pair, 0) > iteration for pair in created) \
                    and value >= current_value:
                continue
            if value < score:
                score, chosen = value, i
        if chosen >= 0:
            m, a, b = moves[chosen]
            u = int(current[m, a])
            crossed = current[m, a + 1:b + 1] if a < b else current[m, b:a]
            tenure = int(rng.integers(7, 19))
            for v in crossed:
                old_pair = (u, int(v)) if a < b else (int(v), u)
                tabu[old_pair] = iteration + tenure
            current, current_value = batch[chosen].copy(), score
            checked = mq.decode(inst, current, validate=False)
            if checked["feasible"]:
                _remember(inst, witnesses, seen, checked)
        if current_value < best_value:
            best_value = current_value
            last_improvement = iteration
        if iteration - last_improvement >= stagnation:
            current = gt_initial(processing, routes, inst.jobs, inst.machines,
                                 int(rng.integers(1, 2 ** 30)))
            current_value = evaluate_one(current, processing, inst.machines)[0]
            tabu.clear()
            last_improvement = iteration
        if iteration % 100 == 0:
            tabu = {k: v for k, v in tabu.items() if v > iteration}
        if verbose and iteration % 1000 == 0:
            print(f"  trajectory seed={seed} iter={iteration} "
                  f"value={current_value}", file=sys.stderr)
    return snapshots


# ---------------------------------------------------------------------------
# ONLINE builder: snapshot -> candidate pool (never touches any oracle)
# ---------------------------------------------------------------------------

def _machine_scores(inst, snap, strategy):
    """Per-machine selection scores for the structured strategies."""
    scores = np.zeros(inst.machines)
    if strategy in ("critical_path", "critical_path_randomfill"):
        head, tail, makespan = snap.head, snap.tail, snap.value
        for m, row in enumerate(snap.orders):
            tight = 0
            for k in range(len(row) - 1):
                u, v = int(row[k]), int(row[k + 1])
                if head[u] + inst.processing[u] + inst.processing[v] + tail[v] \
                        == makespan:
                    tight += 1
            scores[m] = tight
    elif strategy == "witness":
        for w in snap.witnesses:
            for m, _u, _v in w["relations"]:
                scores[m] += 1
    else:
        raise ValueError(f"no machine score for strategy {strategy!r}")
    return scores


def _alternatives_for(inst, snap, m, k_max, rng, fill):
    """Up to k_max-1 alternatives for machine m.

    fill="move_then_perm": move-derived orders first (ranked by their
    evaluated full-graph value), then random permutations up to budget —
    the predetermined share of singly-unverified candidates.
    fill="perm_only":     random permutations only (attribution ablation:
    machine-selection quality without move-derived candidate quality).
    """
    incumbent = tuple(int(v) for v in snap.orders[m])
    alts, sources = [], []
    if fill == "move_then_perm":
        found = {}
        for mm, order, value in zip(snap.move_machines, snap.move_orders,
                                    snap.move_values):
            if mm == m and order != incumbent and (order not in found
                                                   or value < found[order]):
                found[order] = value
        alts = sorted(found, key=lambda o: (found[o], o))[:k_max - 1]
        sources = ["move"] * len(alts)
    elif fill != "perm_only":
        raise ValueError(f"unknown alternative fill {fill!r}")
    if len(alts) < k_max - 1 and len(inst.groups[m]) > 1:
        have = {incumbent, *alts}
        tries = 0
        while len(alts) < k_max - 1 and tries < 64:
            tries += 1
            perm = tuple(int(v) for v in rng.permutation(list(inst.groups[m])))
            if perm not in have:
                have.add(perm)
                alts.append(perm)
                sources.append("random_perm")
    return [incumbent] + alts, sources


def build_pool(inst, snap, strategy, s_max, k_max, rng):
    """Candidate pool for one snapshot under one strategy.

    Returns (pool, meta).  The incumbent order is always candidate 0 on every
    machine (retention by construction); unselected machines carry only the
    incumbent, i.e. they are the frozen window complement.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    scores = None if strategy == "random" \
        else _machine_scores(inst, snap, strategy)
    if strategy == "random":
        selected = [int(m) for m in rng.choice(inst.machines,
                                               size=min(s_max, inst.machines),
                                               replace=False)]
        fill = "move_then_perm"
    else:
        order = np.argsort(-scores, kind="stable")
        want = min(s_max, inst.machines)
        selected = [int(m) for m in order[:want] if scores[m] > 0]
        if len(selected) < want:
            rest = [int(m) for m in order if scores[m] == 0]
            rng.shuffle(rest)
            selected += rest[:want - len(selected)]
        fill = "perm_only" if strategy == "critical_path_randomfill" \
            else "move_then_perm"
    pool, sources, perm_slots = [], {}, []
    for m in range(inst.machines):
        if m in selected:
            alts, src = _alternatives_for(inst, snap, m, k_max, rng, fill)
            pool.append(tuple(alts))
            sources[m] = src
            for a, s in enumerate(src):
                if s == "random_perm":
                    perm_slots.append([m, a])
        else:
            pool.append((tuple(int(v) for v in snap.orders[m]),))
            sources[m] = ["incumbent_only"]
    mq.validate_pool(inst, pool)
    meta = {
        "selected_machines": [int(m) for m in selected],
        "alternative_sources": {str(m): s for m, s in sources.items()},
        "random_perm_slots": perm_slots,
        "random_perm_candidates": len(perm_slots),
        "move_candidates": sum(1 for m in sources for s in sources[m]
                               if s == "move"),
    }
    return pool, meta


# ---------------------------------------------------------------------------
# OFFLINE diagnostic: enumeration + reprojection correctness (oracle layer)
# ---------------------------------------------------------------------------

def freeze_window(inst, snap, strategy, pool, meta, *, dataset, seed,
                  target_t=None):
    """JSON-ready frozen window record (complete orders, candidates, witnesses,
    fixed machines, target, collection iteration, pool hash)."""
    incumbent_choice = tuple(0 for _ in range(inst.machines))
    mine = check_choice(inst, pool, incumbent_choice)
    prod = mq.decode(inst, [pool[m][0] for m in range(inst.machines)])
    if not mine["feasible"] or not prod["feasible"] \
            or mine["makespan"] != snap.value or prod["makespan"] != snap.value:
        raise RuntimeError("incumbent verification failed in freeze_window")
    pool_hash = mq.content_hash([[list(map(int, order)) for order in candidates]
                                 for candidates in pool])
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "instance": inst.name,
        "instance_sha256": mq.content_hash({
            "durations": np.asarray(inst.durations).tolist(),
            "routes": np.asarray(inst.routes).tolist()}),
        "seed": int(seed),
        "strategy": strategy,
        "trajectory_iteration": int(snap.iteration),
        "pool_sizes": [len(p) for p in pool],
        "pool": [[[int(v) for v in order] for order in candidates]
                 for candidates in pool],
        "pool_sha256": pool_hash,
        "incumbent": [0] * inst.machines,
        "incumbent_orders": [[int(v) for v in row] for row in snap.orders],
        "u0": int(snap.value),
        "witnesses": [dict(w) for w in snap.witnesses],
        "fixed_machines": [m for m in range(inst.machines)
                           if m not in meta["selected_machines"]],
        "target_t": int(target_t if target_t is not None else snap.value),
        "builder_meta": meta,
        "trajectory_seconds": snap.seconds,
    }


def _independent_allowed(inst, pool, relations):
    """Recompute allowed candidate sets without mq.witness_support."""
    per_machine = {}
    for m, u, v in relations:
        per_machine.setdefault(int(m), []).append((int(u), int(v)))
    allowed = {}
    for m, reqs in per_machine.items():
        ok = []
        for a, order in enumerate(pool[m]):
            pos = {v: i for i, v in enumerate(order)}
            if all(pos[u] < pos[v] for u, v in reqs):
                ok.append(a)
        if not ok:
            return None
        allowed[m] = ok
    return allowed


def reprojection_battery(inst, pool, raw_witnesses, choices, makespans, target):
    """Verify witness projection correctness on the NEW pool.

    - the projection primitive and an independent recomputation must agree;
    - a triggered cycle witness implies an infeasible combination;
    - a triggered path witness on a feasible combination implies C >= L_W;
    - zero false exclusions: every combination with C <= target survives the
      surrogate screen (no triggered cycle witness, l_W <= target).
    Unsupported witnesses are counted, not an error: dropping a witness whose
    relations no candidate of the new pool satisfies is the documented
    constant-false path of the T03 layer.
    """
    checks = {"projection_mismatch": 0, "unsupported_dropped": 0,
              "cycle_witness_without_cycle": 0, "path_witness_below_length": 0,
              "false_exclusions": 0}
    projected = []
    l0 = int(inst.lower_bound)
    for w in raw_witnesses:
        rels = tuple(tuple(int(x) for x in r) for r in w["relations"])
        mine = _independent_allowed(inst, pool, rels)
        theirs = mq.witness_support(
            mq.Witness(w["kind"], rels, int(w["length"]), ()), pool)
        if (mine is None) != (theirs is None):
            checks["projection_mismatch"] += 1
            continue
        if mine is None:
            checks["unsupported_dropped"] += 1
            continue
        if {m: sorted(a) for m, a in mine.items()} != \
                {m: sorted(a) for m, a in theirs[1].items()}:
            checks["projection_mismatch"] += 1
            continue
        projected.append({"kind": w["kind"], "length": int(w["length"]),
                          "allowed": theirs[1]})
    for choice, c in zip(choices, makespans):
        triggered, path_max = False, 0
        for w in projected:
            if all(choice[m] in w["allowed"][m] for m in w["allowed"]):
                if w["kind"] == "cycle":
                    triggered = True
                else:
                    path_max = max(path_max, w["length"])
        if triggered and c is not None:
            checks["cycle_witness_without_cycle"] += 1
        if not triggered and c is not None:
            for w in projected:
                if w["kind"] == "path" and c < w["length"] and \
                        all(choice[m] in w["allowed"][m] for m in w["allowed"]):
                    checks["path_witness_below_length"] += 1
        if c is not None and c <= target:
            if triggered or max(l0, path_max) > target:
                checks["false_exclusions"] += 1
    return checks, len(projected)


def evaluate_pool(inst, record, *, witnesses=None, store_makespans=False):
    """Offline oracle evaluation of one frozen window (enumeration allowed).

    ``witnesses`` supplies the raw witness set for records that reference a
    shared snapshot (large D2 collections deduplicate witnesses across the
    strategies built from the same trajectory state)."""
    pool = tuple(tuple(tuple(int(v) for v in order) for order in candidates)
                 for candidates in record["pool"])
    mq.validate_pool(inst, pool)
    incumbent = tuple(int(a) for a in record["incumbent"])
    incumbent_orders = [tuple(int(v) for v in row)
                        for row in record["incumbent_orders"]]
    retention = all(pool[m][incumbent[m]] == incumbent_orders[m]
                    for m in range(inst.machines))
    u0 = int(record["u0"])
    mine = check_choice(inst, pool, incumbent)
    prod = mq.decode(inst, [pool[m][incumbent[m]] for m in range(inst.machines)])
    if not mine["feasible"] or not prod["feasible"] \
            or mine["makespan"] != u0 or prod["makespan"] != u0:
        raise RuntimeError("frozen incumbent does not verify against u0")
    t0 = time.perf_counter()
    enum = enumerate_pool(inst, pool, u0=u0)
    makespans = enum.pop("makespans")
    sizes = [len(p) for p in pool]
    choices = list(itertools.product(*(range(k) for k in sizes)))
    d_imp = None
    for choice, c in zip(choices, makespans):
        if c is None or c >= u0:
            continue
        d = sum(1 for x, y in zip(incumbent, choice) if x != y)
        if d_imp is None or d < d_imp:
            d_imp = d
            if d_imp <= 1:
                break
    checks, n_projected = reprojection_battery(
        inst, pool,
        record["witnesses"] if "witnesses" in record else witnesses,
        choices, makespans, int(record["target_t"]))
    # rejection accounting: does any improving combination use a candidate
    # that was a singly-unverified random permutation?
    perm_slots = [tuple(int(x) for x in s)
                  for s in record["builder_meta"]["random_perm_slots"]]
    improving_needs_perm = None
    if perm_slots:
        hits = [any(choice[m] == a for m, a in perm_slots)
                for choice, c in zip(choices, makespans)
                if c is not None and c < u0]
        improving_needs_perm = [sum(hits), len(hits)]
    out = {
        "strategy": record["strategy"],
        "dataset": record["dataset"],
        "instance": record["instance"],
        "seed": record["seed"],
        "trajectory_iteration": record["trajectory_iteration"],
        "total_combinations": enum["total_combinations"],
        "feasible_rate": enum["feasible_rate"],
        "pool_optimum": enum["pool_optimum"],
        "rho": enum["rho"],
        "improving_combinations": enum["improving_combinations"],
        "d_imp_from_incumbent": d_imp,
        "no_improvement": enum["improving_combinations"] == 0,
        "incumbent_retention": bool(retention),
        "candidate_completeness": True,   # validate_pool raised otherwise
        "reprojection": {"checks": checks, "witnesses_kept": n_projected,
                         "witnesses_raw": len(record.get("witnesses", witnesses)
                                              or []),
                         "all_zero": all(v == 0 for k, v in checks.items()
                                         if k != "unsupported_dropped")},
        "improving_needs_random_perm": improving_needs_perm,
        "enumeration_seconds": time.perf_counter() - t0,
    }
    if store_makespans:
        out["makespans"] = [int(m) if m is not None else None
                            for m in makespans]
    return out


# ---------------------------------------------------------------------------
# Complete-classical dynamic search with pluggable windows (D1 layer)
# ---------------------------------------------------------------------------

def dynamic_search(inst, strategy, *, seconds=10.0, seed=7, s_max=6, k_max=3,
                   proposal_every=40, max_iterations=100000):
    """Classical tabu + periodic whole-pool joint evaluation.

    The pool is built by the T04 builder from the live trajectory state; every
    ``proposal_every`` iterations every combination of the pool (exhaustive
    over <= k_max**s_max <= 729 states) is evaluated by the same batch
    evaluator and the best improving combination is accepted.  This is the
    classical joint consumer the T04 deliverable owes T06: the quantum arm
    would sample the identical pool/witness/action interface.
    """
    from adaptive_search import apply_moves, critical_moves, gt_initial
    from batch_evaluator import BatchEvaluator, INVALID, evaluate_one

    started_setup = time.perf_counter()
    rng = np.random.default_rng(seed)
    evaluator = BatchEvaluator(inst, "cpu")
    processing = evaluator.processing
    routes = np.array(inst.machine_of, dtype=np.int32)
    _, info = mq.build_pool(inst, 8, 240, seed)
    current = np.asarray(info["orders"], dtype=np.int32).copy()
    decoded = mq.decode(inst, current)
    if not decoded["feasible"]:
        raise ValueError("initial schedule is infeasible")
    current_value = best_value = int(decoded["makespan"])
    best = current.copy()
    evaluator.evaluate(current[None])
    gt_initial(processing, routes, inst.jobs, inst.machines, seed)
    # Same budget accounting as adaptive_search.solve: the search clock starts
    # after setup (pool construction + JIT warm-up), never across it.
    setup_seconds = time.perf_counter() - started_setup
    started = time.perf_counter()
    tabu, witnesses, seen = {}, [], set()
    _remember(inst, witnesses, seen, decoded)
    pool_calls = pool_evaluations = pool_improvements = 0
    pool_seconds = 0.0
    iteration = 0
    deadline = started + seconds
    last_improvement = 0
    while iteration < max_iterations and time.perf_counter() < deadline:
        iteration += 1
        current_value, head, tail = evaluate_one(current, processing, inst.machines)
        insertion = iteration % 10 == 0
        moves = critical_moves(current, processing, head, tail, current_value,
                               insertion=insertion)
        rng.shuffle(moves)
        batch = apply_moves(current, moves)
        values = evaluator.evaluate(batch, validate=False)
        chosen, score = -1, INVALID
        for i, (m, a, b) in enumerate(moves):
            value = int(values[i])
            if value >= INVALID:
                continue
            u = int(current[m, a])
            crossed = current[m, a + 1:b + 1] if a < b else current[m, b:a]
            created = [(int(v), u) if a < b else (u, int(v)) for v in crossed]
            if any(tabu.get(pair, 0) > iteration for pair in created) \
                    and value >= best_value:
                continue
            if value < score:
                score, chosen = value, i
        if iteration % proposal_every == 0 and time.perf_counter() < deadline:
            tic = time.perf_counter()
            # Fresh evidence for the witness strategy: decode the current state
            # once per proposal cycle (not per accepted move, matching the
            # production cost profile where decode runs on improvements only).
            _remember(inst, witnesses, seen,
                      mq.decode(inst, current, validate=False))
            value, head2, tail2 = evaluate_one(current, processing, inst.machines)
            move_m, move_o, move_v = [], [], []
            for i, (m, _a, _b) in enumerate(moves):
                if values[i] < INVALID:
                    move_m.append(int(m))
                    move_o.append(tuple(int(v) for v in batch[i, m]))
                    move_v.append(int(values[i]))
            snap = Snapshot(iteration, current, int(value),
                            np.asarray(head2), np.asarray(tail2),
                            [dict(w) for w in witnesses],
                            move_m, move_o, move_v, 0.0)
            pool, _meta = build_pool(inst, snap, strategy, s_max, k_max,
                                     _stable_rng(seed, iteration))
            combo = np.array(
                [[pool[m][a] for m, a in enumerate(c)] for c in
                 itertools.product(*(range(len(p)) for p in pool))],
                dtype=np.int32).reshape((-1, inst.machines, inst.jobs))
            scores = evaluator.evaluate(combo, validate=False)
            pool_calls += 1
            pool_evaluations += len(scores)
            feasible = np.where(scores < INVALID)[0]
            if len(feasible):
                q = int(feasible[np.argmin(scores[feasible])])
                if int(scores[q]) < min(current_value, score):
                    current = combo[q].copy()
                    current_value = int(scores[q])
                    chosen = -2
                    if current_value < best_value:
                        pool_improvements += 1
            pool_seconds += time.perf_counter() - tic
        if chosen >= 0:
            m, a, b = moves[chosen]
            u = int(current[m, a])
            crossed = current[m, a + 1:b + 1] if a < b else current[m, b:a]
            tenure = int(rng.integers(7, 19))
            for v in crossed:
                old_pair = (u, int(v)) if a < b else (int(v), u)
                tabu[old_pair] = iteration + tenure
            current, current_value = batch[chosen].copy(), score
        elif chosen != -2:
            current = gt_initial(processing, routes, inst.jobs, inst.machines,
                                 int(rng.integers(1, 2 ** 30)))
            current_value = evaluate_one(current, processing, inst.machines)[0]
            tabu.clear()
            last_improvement = iteration
        if current_value < best_value:
            checked = mq.decode(inst, current)
            if not checked["feasible"] or \
                    mq.validate_schedule(inst, checked["starts"]) != current_value:
                raise RuntimeError("independent schedule verification failed")
            best_value, best = current_value, current.copy()
            last_improvement = iteration
            _remember(inst, witnesses, seen, checked)
        if iteration - last_improvement >= 600:
            current = gt_initial(processing, routes, inst.jobs, inst.machines,
                                 int(rng.integers(1, 2 ** 30)))
            current_value = evaluate_one(current, processing, inst.machines)[0]
            tabu.clear()
            last_improvement = iteration
        if iteration % 100 == 0:
            tabu = {k: v for k, v in tabu.items() if v > iteration}
    return {
        "schema_version": SCHEMA_VERSION,
        "instance": inst.name, "strategy": strategy, "seed": int(seed),
        "budget_seconds": seconds,
        "best_makespan": int(best_value),
        "independent_verification": True,
        "iterations": iteration,
        "setup_seconds": setup_seconds,
        "search_seconds": time.perf_counter() - started,
        "pool_calls": pool_calls,
        "pool_evaluations": pool_evaluations,
        "pool_improvements": pool_improvements,
        "pool_seconds": pool_seconds,
        "wall_seconds": time.perf_counter() - started,
        "graph_evaluations": evaluator.evaluations,
        "s_max": s_max, "k_max": k_max, "proposal_every": proposal_every,
    }


def d1_instances():
    """The six frozen D1 instance files (data layer of the breakdown doc)."""
    names = ["tai15_15_01_test.txt", "tai20_15_01_test.txt", "ta21.txt",
             "tai50_20_01.txt", "tai50_20_02.txt", "tai50_20_03.txt"]
    out = []
    for name in names:
        path = ROOT / "task_data" / name
        if path.exists():
            out.append(mq.Instance.read(path))
    if not out:
        raise FileNotFoundError("no D1 instance files under task_data/")
    return out


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def run_d0(seeds, s_max, k_max, checkpoints=(0,)):
    """Correctness battery on D0 (small, fully enumerable windows)."""
    import data_identity
    windows, evals = [], []
    for jobs, machines in data_identity.D0_SIZES:
        for seed in seeds:
            inst = data_identity.build_d0(jobs, machines, seed)
            for snap in classical_trajectory(inst, seed, checkpoints):
                for strategy in STRATEGIES:
                    rng = _stable_rng(seed, snap.iteration, _strategy_index(strategy))
                    pool, meta = build_pool(inst, snap, strategy, s_max, k_max, rng)
                    rec = freeze_window(inst, snap, strategy, pool, meta,
                                        dataset="D0", seed=seed)
                    evals.append(evaluate_pool(inst, rec, store_makespans=True))
                    windows.append(rec)
    return windows, evals


def _witness_key(w):
    return (w["kind"], tuple(map(tuple, w["relations"])), int(w["length"]))


def _flat_witness(w):
    """JSON-compact witness: flat [m,u,v,...] relations plus kind and length."""
    flat = [int(x) for r in w["relations"] for x in r]
    return {"kind": w["kind"], "length": int(w["length"]), "relations_flat": flat}


def run_d2_collection(seeds, checkpoints, s_max, k_max):
    """Freeze D2 windows from pure-classical D1 trajectories (dev seeds).

    The four strategy windows of one trajectory state share the same raw
    witness set and the same incumbent orders.  For a file that stays
    reviewable, witnesses are deduplicated once per trajectory (sliding-window
    memory overlaps heavily between checkpoints) and referenced by index;
    witness relations are flattened; incumbent orders live on the snapshot.
    ``evaluate_pool`` receives the witness objects in memory during the run,
    so the numbers are identical to a fully self-contained freeze; the shipped
    file is an exact, lossless re-encoding, and re-running this collection
    reproduces every pool hash bit-for-bit (determinism is unit-tested).
    """
    windows, evals, snapshots, witness_tables = [], [], {}, {}
    for inst in d1_instances():
        for seed in seeds:
            traj_key = f"{inst.name}|s{seed}"
            table, table_keys = [], {}
            for snap in classical_trajectory(inst, seed, checkpoints):
                indices = []
                for w in snap.witnesses:
                    key = _witness_key(w)
                    if key not in table_keys:
                        table_keys[key] = len(table)
                        table.append(_flat_witness(w))
                    indices.append(table_keys[key])
                snap_key = f"{traj_key}|it{snap.iteration}"
                snapshots[snap_key] = {
                    "trajectory_key": traj_key,
                    "trajectory_iteration": int(snap.iteration),
                    "incumbent_orders": [[int(v) for v in row]
                                         for row in snap.orders],
                    "witness_indices": indices,
                }
                for strategy in STRATEGIES:
                    rng = _stable_rng(seed, snap.iteration,
                                      _strategy_index(strategy))
                    pool, meta = build_pool(inst, snap, strategy, s_max, k_max,
                                            rng)
                    rec = freeze_window(inst, snap, strategy, pool, meta,
                                        dataset="D2", seed=seed)
                    witnesses = rec.pop("witnesses")
                    rec.pop("incumbent_orders", None)
                    rec.pop("trajectory_seconds", None)
                    rec["snapshot_key"] = snap_key
                    evals.append(evaluate_pool(inst, rec,
                                               witnesses=witnesses))
                    windows.append(rec)
            witness_tables[traj_key] = table
    snap_out = [{"snapshot_key": k, **v} for k, v in sorted(snapshots.items())]
    table_out = [{"trajectory_key": k, "witnesses": w}
                 for k, w in witness_tables.items()]
    return windows, evals, snap_out, table_out


def summarize(evals, label):
    """Aggregate the acceptance numbers for one dataset."""
    rows = {}
    for e in evals:
        rows.setdefault(e["strategy"], []).append(e)
    summary = {}
    for strategy, group in rows.items():
        no_imp = sum(1 for e in group if e["no_improvement"])
        rho = [e["rho"] for e in group if e["rho"] is not None]
        summary[strategy] = {
            "windows": len(group),
            "no_improvement_ratio": no_imp / len(group),
            "improving_windows": len(group) - no_imp,
            "rho_mean": float(np.mean(rho)) if rho else None,
            "rho_max": float(np.max(rho)) if rho else None,
            "d_imp_1": sum(1 for e in group
                           if e["d_imp_from_incumbent"] == 1),
            "d_imp_inf": no_imp,
            "retention_ok": all(e["incumbent_retention"] for e in group),
            "completeness_ok": all(e["candidate_completeness"] for e in group),
            "reprojection_ok": all(
                e["reprojection"]["checks"]["projection_mismatch"] == 0
                and e["reprojection"]["checks"]["cycle_witness_without_cycle"] == 0
                and e["reprojection"]["checks"]["path_witness_below_length"] == 0
                and e["reprojection"]["checks"]["false_exclusions"] == 0
                for e in group),
        }
    if "random" in rows:
        base = summary["random"]["no_improvement_ratio"]
        for strategy in rows:
            if strategy == "random":
                continue
            ratio = summary[strategy]["no_improvement_ratio"]
            summary[strategy]["relative_reduction_vs_random"] = \
                (base - ratio) / base if base > 0 else None
            summary[strategy]["gate_20pct"] = bool(
                base > 0 and (base - ratio) / base >= 0.20)
    return {"label": label, "strategies": summary}


def main(argv=None):
    ap = argparse.ArgumentParser(description="T04 dynamic window/pool builder")
    ap.add_argument("--d0", action="store_true", help="D0 correctness battery")
    ap.add_argument("--d2", action="store_true",
                    help="D2 frozen-window collection on D1 trajectories")
    ap.add_argument("--d1-dynamic", action="store_true",
                    help="complete-classical dynamic search comparison on D1")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--checkpoints", default="0,100,500,1000,5000")
    ap.add_argument("--s-max", type=int, default=6)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--dynamic-seconds", type=float, default=10.0)
    ap.add_argument("--out", default=str(GATES_DIR / "results_window_pool_20261003"
                                         / "window_pool.json"))
    args = ap.parse_args(argv)
    lo, hi = args.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    checkpoints = [int(c) for c in args.checkpoints.split(",")]
    result = {"schema_version": SCHEMA_VERSION,
              "work_package": "T04 (Issue #19, PR #18 docs TODO)",
              "s_max": args.s_max, "k_max": args.k_max}
    if args.d0:
        windows, evals = run_d0(seeds, args.s_max, args.k_max)
        result["d0"] = {"windows": windows, "evaluations": evals,
                        "summary": summarize(evals, "D0")}
    if args.d2:
        windows, evals, snap_out, table_out = run_d2_collection(
            seeds, checkpoints, args.s_max, args.k_max)
        result["d2"] = {"witness_table": table_out, "snapshots": snap_out,
                        "windows": windows, "evaluations": evals,
                        "summary": summarize(evals, "D2-dev")}
    if args.d1_dynamic:
        rows = []
        for inst in d1_instances():
            for seed in seeds:
                from adaptive_search import solve
                r = solve(inst, seconds=args.dynamic_seconds, seed=seed,
                          mode="classical")
                rows.append({"instance": inst.name, "strategy": "mainline",
                             "seed": seed, "best_makespan": r["best_makespan"],
                             "wall_seconds": r["wall_seconds"],
                             "setup_seconds": r["setup_seconds"],
                             "search_seconds": r["search_seconds"],
                             "iterations": r["iterations"],
                             "graph_evaluations": r["graph_evaluations"]})
                print(f"  {inst.name} s{seed} mainline: {r['best_makespan']}",
                      file=sys.stderr)
                for strategy in STRATEGIES:
                    r = dynamic_search(inst, strategy,
                                       seconds=args.dynamic_seconds, seed=seed,
                                       s_max=args.s_max, k_max=args.k_max)
                    rows.append(r)
                    print(f"  {inst.name} s{seed} {strategy}: "
                          f"{r['best_makespan']}", file=sys.stderr)
        result["d1_dynamic"] = rows
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False,
                              separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out}")
    for key in ("d0", "d2"):
        if key in result:
            print(key, json.dumps(result[key]["summary"],
                  ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
