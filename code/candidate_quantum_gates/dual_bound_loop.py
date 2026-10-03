#!/usr/bin/env python3
"""T08: dual-bound closed loop with scope-safe witness feedback.

Implements the T08 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md: one loop that keeps a
non-increasing verified upper bound U(t) and a non-decreasing *certified* global
lower bound L(t), with a replaceable proposer on the good-solution channel and
the T02 certification channel on the other side.

Invariants enforced (and re-checked after every event):

* ``U`` moves only to the makespan of a schedule that passed both the
  independent graph checker (``window_diagnostics.check_choice``) and the
  production validator (``medium_qjsp.validate_schedule``);
* ``L`` moves only from a certificate whose scope is ``global`` or
  ``relaxation`` - the trivial load bound, a certified relaxation bound, or a
  *proved infeasible* threshold test of the complete model;
* sampling failure, an empty proposal, a solver ``UNKNOWN`` or a timeout never
  change ``L`` (they are logged as failures and discarded);
* raw witnesses are extracted from every verified decode and re-projected when
  the pool changes; pool-scope bounds are dropped on a pool change and their
  stale values are never reused;
* failures are evidence for nothing: a rejected combination only contributes a
  witness, never a bound.

Every event is appended to a replayable log with timestamps, cost accounting
(wall seconds and graph evaluations), the bound before/after and the solver
status, so the update chain can be audited (T08 acceptance) and consumed by T10.
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

import global_bounds as gb  # noqa: E402
import medium_qjsp as mq  # noqa: E402
from window_diagnostics import build_window, check_choice  # noqa: E402

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Proposers (replaceable good-solution channel)
# ---------------------------------------------------------------------------

class UniformProposer:
    """Independent uniform proposal over the legal candidate pool.

    The strongest classical control measured in T06/T07: one sample plus its
    graph check, no training, no hidden state.
    """

    name = "uniform"

    def __init__(self, sizes, seed=7):
        self.sizes = tuple(int(s) for s in sizes)
        self.rng = np.random.default_rng(seed)

    def propose(self):
        return tuple(int(self.rng.integers(k)) for k in self.sizes)

    def cost(self):
        return {"graph_evaluations": 1, "training_seconds": 0.0, "shots": 0}


class FixedProposer:
    """Deterministic proposal stream, used to inject total failure in tests."""

    name = "fixed"

    def __init__(self, proposal=None):
        self.proposal = proposal

    def propose(self):
        return self.proposal

    def cost(self):
        return {"graph_evaluations": 0, "training_seconds": 0.0, "shots": 0}


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

class DualBoundLoop:
    """One instance, one pool, two channels, one auditable event log."""

    def __init__(self, inst, pool, *, initial_choice=None, time_budget=10.0,
                 cert_time_limit=1.0, proposer=None, seed=7, log=None):
        self.inst = inst
        self.pool = pool
        self.sizes = tuple(len(machine) for machine in pool)
        mq.validate_pool(inst, pool)
        self.pool_hash = mq.content_hash(pool)
        self.time_budget = float(time_budget)
        self.cert_time_limit = float(cert_time_limit)
        self.proposer = proposer if proposer is not None else UniformProposer(self.sizes, seed)
        self.log = log
        self.seed = seed
        self.events = []
        self.graph_evaluations = 0
        self.certificate_calls = 0
        self.failures = {"empty_proposal": 0, "sampling_failure": 0,
                         "certificate_unproven": 0, "certificate_error": 0}
        self.witnesses = []
        self.witness_keys = set()
        self.pool_scope_bounds = {}      # dropped on pool change
        self.start = time.perf_counter()
        self.U = None   # set by the verified initial schedule below
        self.L = None   # set by the certified trivial bound below

        # --- lower bound: certified primitives only -------------------------
        self.L = gb.trivial_bound(inst)
        self.L_scope = "global"
        self.L_evidence = "trivial load bound max(job load, machine load)"
        self._event("bound", {"bound_kind": "trivial_bound", "value": self.L,
                              "scope": self.L_scope, "evidence": self.L_evidence},
                    {"lower_before": None, "lower_after": self.L})

        # --- upper bound: an explicit verified choice, else serial-SGS ------
        if initial_choice is not None:
            choice = tuple(int(a) for a in initial_choice)
            if len(choice) != len(self.pool):
                raise ValueError("initial_choice must name one candidate per machine")
            source = "caller_supplied_verified_choice"
        else:
            starts, _makespan = mq.serial_sgs(inst, "mwr",
                                              np.random.default_rng(seed))
            orders = [tuple(sorted(inst.groups[m], key=lambda v: starts[v]))
                      for m in range(inst.machines)]
            choice = self._choice_from_orders(orders)
            source = "serial_sgs_mwr"
        verified = self._verify(choice)
        if not verified["feasible"]:
            raise RuntimeError("initial schedule must be feasible")
        self.U = verified["makespan"]
        self.U_choice = choice
        self.initial_U = self.U
        self._add_witness(choice)
        self._event("init", {"choice": list(choice), "makespan": self.U,
                             "source": source},
                    {"upper_before": None, "upper_after": self.U})
        self.trajectory = [(0.0, self.U, self.L)]

    # -- helpers ------------------------------------------------------------

    def _choice_from_orders(self, orders):
        choice = []
        for machine, row in enumerate(orders):
            for label, candidate in enumerate(self.pool[machine]):
                if tuple(candidate) == tuple(row):
                    choice.append(label)
                    break
            else:
                raise ValueError("order not present in the pool")
        return tuple(choice)

    def _verify(self, choice):
        """Double verification: independent checker plus production validator."""
        self.graph_evaluations += 1
        checked = check_choice(self.inst, self.pool, choice)
        if not checked["feasible"]:
            return {"feasible": False, "makespan": None}
        decoded = mq.decode(self.inst, mq.choice_orders(self.pool, list(choice)))
        if not decoded["feasible"]:
            raise RuntimeError("checkers disagree on feasibility")
        validated = mq.validate_schedule(self.inst, decoded["starts"])
        if validated != checked["makespan"] or validated != decoded["makespan"]:
            raise RuntimeError("checkers disagree on makespan")
        return {"feasible": True, "makespan": validated,
                "starts": decoded["starts"]}

    def _add_witness(self, choice):
        decoded = mq.decode(self.inst, mq.choice_orders(self.pool, list(choice)),
                            validate=False)
        witness = mq.separate(self.inst, decoded)
        if witness.key() not in self.witness_keys:
            self.witness_keys.add(witness.key())
            self.witnesses.append(witness)
        return witness

    def _event(self, kind, payload, bounds):
        event = {
            "index": len(self.events),
            "elapsed_seconds": time.perf_counter() - self.start,
            "kind": kind,
            "lower_bound": self.L,
            "upper_bound": self.U,
            "bounds_before_after": bounds,
            "graph_evaluations": self.graph_evaluations,
            "witnesses": len(self.witnesses),
            "pool_hash": self.pool_hash,
            **payload,
        }
        self.events.append(event)
        if self.log:
            self.log(event)
        return event

    # -- good-solution channel ---------------------------------------------

    def propose_and_verify(self):
        """One proposal round: sample, verify, maybe improve U, extract witness."""
        proposal = self.proposer.propose()
        if proposal is None:
            self.failures["empty_proposal"] += 1
            self._event("proposal_failure", {"reason": "empty_proposal",
                                             "proposer": self.proposer.name}, {})
            return False
        cost = self.proposer.cost()
        before = self.U
        result = self._verify(proposal)
        self.failures["sampling_failure"] += int(not result["feasible"])
        self._add_witness(proposal)
        if result["feasible"] and result["makespan"] < self.U:
            self.U = result["makespan"]
            self.U_choice = tuple(proposal)
            self._event("improvement", {"choice": list(proposal),
                                        "makespan": result["makespan"],
                                        "proposer": self.proposer.name,
                                        "cost": cost},
                        {"upper_before": before, "upper_after": self.U})
            self.trajectory.append((time.perf_counter() - self.start, self.U, self.L))
            return True
        self._event("rejection" if result["feasible"] else "infeasible_sample",
                    {"choice": list(proposal), "makespan": result["makespan"],
                     "proposer": self.proposer.name, "cost": cost},
                    {"upper_before": before, "upper_after": self.U})
        return False

    # -- certification channel ---------------------------------------------

    def certify(self, time_limit=None):
        """Threshold test at U-1; only a proof of infeasibility may raise L."""
        limit = self.cert_time_limit if time_limit is None else float(time_limit)
        before = self.L
        if limit <= 0:
            self.failures["certificate_unproven"] += 1
            self._event("certificate", {"certificate_kind": "threshold_infeasibility",
                                        "threshold": self.U - 1,
                                        "verdict": "skipped_no_budget",
                                        "scope": None}, {"lower_before": before,
                                                         "lower_after": self.L})
            return False
        self.certificate_calls += 1
        try:
            record = gb.threshold_infeasibility(self.inst, self.U - 1,
                                                time_limit=limit)
        except Exception as exc:  # pragma: no cover - reported, never raises L
            self.failures["certificate_error"] += 1
            self._event("certificate", {"certificate_kind": "threshold_infeasibility",
                                        "threshold": self.U - 1,
                                        "verdict": "error",
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "scope": None},
                        {"lower_before": before, "lower_after": self.L})
            return False
        verdict = record["verdict"]
        if verdict == "infeasible":
            self.L = max(self.L, int(record["raises_lower_bound_to"]))
            self.L_scope = "global"
            self.L_evidence = (f"complete model proved C_max <= {self.U - 1} "
                               f"infeasible (status {record['status']})")
        else:
            self.failures["certificate_unproven"] += 1
        self.trajectory.append((time.perf_counter() - self.start, self.U, self.L))
        self._event("certificate", {"certificate_kind": "threshold_infeasibility",
                                    "threshold": self.U - 1,
                                    "verdict": verdict,
                                    "solver_status": record["status"],
                                    "scope": "global" if verdict == "infeasible" else None,
                                    "evidence": self.L_evidence},
                    {"lower_before": before, "lower_after": self.L})
        return verdict == "infeasible"

    def relax_bound(self, machine_subset, time_limit=1.0):
        """Certified relaxation bound; only a proved optimum of the relaxation
        (or its dual bound) is a valid global lower bound."""
        before = self.L
        self.certificate_calls += 1
        record = gb.subset_relaxation_bound(self.inst, machine_subset,
                                            time_limit=time_limit)
        value = record.get("certified_bound")
        if value is not None:
            self.L = max(self.L, int(value))
            self.L_scope = "relaxation"
            self.L_evidence = (f"relaxation R_S on machines {list(machine_subset)} "
                               f"status {record['status']}")
        self._event("certificate", {"certificate_kind": "subset_relaxation",
                                    "machine_subset": list(machine_subset),
                                    "solver_status": record["status"],
                                    "certified_bound": value,
                                    "scope": "relaxation" if value is not None else None},
                    {"lower_before": before, "lower_after": self.L})
        return value

    # -- pool change (witness re-projection) -------------------------------

    def change_pool(self, new_pool):
        """Swap the pool, re-project witnesses and drop pool-scope bounds."""
        from witness_surrogate import collect_witnesses
        mq.validate_pool(self.inst, new_pool)
        raw = [w for w in self.witnesses]
        dropped, kept = 0, []
        for witness in raw:
            projected = mq.witness_support(witness, new_pool)
            if projected is None:
                dropped += 1
                continue
            kept.append(witness)
        self.pool = new_pool
        self.pool_hash = mq.content_hash(new_pool)
        self.witnesses = kept
        self.witness_keys = {w.key() for w in kept}
        stale = len(self.pool_scope_bounds)
        self.pool_scope_bounds = {}
        self._event("pool_change",
                    {"witnesses_kept": len(kept), "witnesses_dropped": dropped,
                     "pool_scope_bounds_dropped": stale,
                     "lower_bound_unchanged": self.L, "lower_scope": self.L_scope},
                    {"lower_bound_kept": self.L, "scope": self.L_scope,
                     "reason": "pool-scope bounds are never carried across pools"})
        return {"kept": len(kept), "dropped": dropped, "pool_scope_bounds_dropped": stale}

    # -- runs --------------------------------------------------------------

    def run(self, max_rounds=200, cert_every=5):
        """Alternate proposal rounds and certification slices inside the budget."""
        rounds = 0
        while (time.perf_counter() - self.start < self.time_budget
               and self.L < self.U and rounds < max_rounds):
            self.propose_and_verify()
            if rounds % cert_every == cert_every - 1:
                self.certify()
            rounds += 1
        if self.L < self.U:
            self.certify(time_limit=min(self.cert_time_limit,
                                        max(0.0, self.time_budget - (time.perf_counter() - self.start))))
        self.trajectory.append((time.perf_counter() - self.start, self.U, self.L))
        return self.summary()

    def gap_integral(self, budget=None):
        """A_cert(B) = B^-1 * integral_0^B (U-L)/U dt on the recorded trajectory."""
        budget = budget or (self.trajectory[-1][0] or 1.0)
        area = 0.0
        for (t0, u0, l0), (t1, _u1, _l1) in zip(self.trajectory, self.trajectory[1:]):
            span = max(0.0, min(t1, budget) - t0)
            area += span * ((u0 - l0) / u0 if u0 else 0.0)
        return area / budget if budget > 0 else None

    def summary(self):
        elapsed = time.perf_counter() - self.start
        return {
            "instance": self.inst.name,
            "initial_upper_bound": self.initial_U,
            "upper_bound": self.U,
            "lower_bound": self.L,
            "lower_bound_scope": self.L_scope,
            "lower_bound_evidence": self.L_evidence,
            "proved_pool_optimal": False,
            "gap_closed": self.L >= self.U,
            "wall_seconds": elapsed,
            "graph_evaluations": self.graph_evaluations,
            "certificate_calls": self.certificate_calls,
            "witnesses": len(self.witnesses),
            "failures": dict(self.failures),
            "gap_integral": self.gap_integral(budget=elapsed),
            "events": list(self.events),
        }


# ---------------------------------------------------------------------------
# Invariant checking against the known D0 optimum
# ---------------------------------------------------------------------------

def check_invariants(loop, c_star):
    """L <= C* <= U at every recorded event, plus legal accepted schedules."""
    violations = []
    for event in loop.events:
        lower, upper = event["lower_bound"], event["upper_bound"]
        if lower is not None and lower > c_star:
            violations.append({"event": event["index"], "kind": "L_above_optimum",
                               "L": lower, "c_star": c_star})
        if upper is not None and upper < c_star:
            violations.append({"event": event["index"], "kind": "U_below_optimum",
                               "U": upper, "c_star": c_star})
    if loop.L > loop.U:
        violations.append({"kind": "bounds_crossed", "L": loop.L, "U": loop.U})
    return violations


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_d0(seeds, *, pool_k=3, time_budget=3.0, cert_time_limit=0.5,
           cert_every=5, max_rounds=200, seed_offset=1000):
    import data_identity
    rows = []
    for jobs, machines in data_identity.D0_SIZES:
        for seed in seeds:
            inst = data_identity.build_d0(jobs, machines, seed)
            pool, _incumbent, _u0 = build_window(inst, pool_k, seed + seed_offset)
            loop = DualBoundLoop(inst, pool, time_budget=time_budget,
                                 cert_time_limit=cert_time_limit, seed=seed,
                                 log=None)
            summary = loop.run(max_rounds=max_rounds, cert_every=cert_every)
            exact = data_identity.enumerate_exact_optimum(inst)["exact_optimum"]
            violations = check_invariants(loop, exact)
            rows.append({
                "scale": f"{jobs}x{machines}", "seed": seed,
                "instance_sha256": data_identity.instance_sha256(inst),
                "exact_optimum": exact,
                "policy": {"pool_k": pool_k, "time_budget": time_budget,
                           "cert_time_limit": cert_time_limit,
                           "cert_every": cert_every, "max_rounds": max_rounds},
                "summary": summary,
                "invariant_violations": violations,
                "trajectory": [[float(t), int(u), int(l)]
                               for t, u, l in loop.trajectory],
            })
            print(f"  {jobs}x{machines} seed {seed}: C*={exact} "
                  f"U: {summary['initial_upper_bound']} -> {summary['upper_bound']} "
                  f"L={summary['lower_bound']} ({summary['lower_bound_scope']}) "
                  f"gap={summary['gap_integral']:.4f} "
                  f"violations={len(violations)} events={len(summary['events'])}",
                  flush=True)
    return rows


def run_failure_injections(seed=0, pool_k=3):
    """Adversarial cases the work package asks for: L must never be raised."""
    import data_identity
    inst = data_identity.build_d0(3, 3, seed)
    pool, _incumbent, _u0 = build_window(inst, pool_k, seed + 1000)
    rows = []

    # 1. total sampling failure: the proposer never returns anything
    loop = DualBoundLoop(inst, pool, time_budget=1.0, cert_time_limit=0.0,
                         proposer=FixedProposer(None), seed=seed)
    before = loop.L
    for _ in range(5):
        loop.propose_and_verify()
    rows.append({"case": "empty_proposal", "lower_before": before,
                 "lower_after": loop.L, "raised": loop.L > before,
                 "failures": dict(loop.failures)})

    # 2. samples that are always infeasible (a bad but non-empty proposer)
    loop = DualBoundLoop(inst, pool, time_budget=1.0, cert_time_limit=0.0,
                         proposer=FixedProposer(None), seed=seed)
    infeasible = _an_infeasible_choice(inst, pool)
    loop.proposer = FixedProposer(infeasible)
    before = loop.L
    for _ in range(10):
        loop.propose_and_verify()
    rows.append({"case": "all_samples_infeasible", "lower_before": before,
                 "lower_after": loop.L, "raised": loop.L > before,
                 "failures": dict(loop.failures)})

    # 3. solver timeout: certificate with a zero/insufficient budget proves nothing
    loop = DualBoundLoop(inst, pool, time_budget=1.0, cert_time_limit=1e-9, seed=seed)
    before = loop.L
    loop.certify(time_limit=1e-9)
    rows.append({"case": "certificate_timeout", "lower_before": before,
                 "lower_after": loop.L, "raised": loop.L > before,
                 "failures": dict(loop.failures)})

    # 4. pool change: stale pool-scope bounds are dropped, L keeps its scope
    loop = DualBoundLoop(inst, pool, time_budget=1.0, cert_time_limit=0.0, seed=seed)
    loop.pool_scope_bounds = {"old_pool": 123456}
    before, scope = loop.L, loop.L_scope
    other = build_window(inst, pool_k, seed + 2000)[0]
    change = loop.change_pool(other)
    rows.append({"case": "pool_change", "lower_before": before,
                 "lower_after": loop.L, "lower_scope_before": scope,
                 "lower_scope_after": loop.L_scope, "raised": loop.L > before,
                 "pool_scope_bounds_dropped": change["pool_scope_bounds_dropped"],
                 "witnesses_kept": change["kept"], "witnesses_dropped": change["dropped"]})
    return rows


def _an_infeasible_choice(inst, pool):
    for choice in itertools.product(*(range(len(m)) for m in pool)):
        if not check_choice(inst, pool, choice)["feasible"]:
            return choice
    raise RuntimeError("no infeasible combination in this pool")


def main(argv=None):
    ap = argparse.ArgumentParser(description="T08 dual-bound closed loop")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--pool-k", type=int, default=3)
    ap.add_argument("--time-budget", type=float, default=3.0)
    ap.add_argument("--cert-time-limit", type=float, default=0.5)
    ap.add_argument("--cert-every", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=200)
    ap.add_argument("--out", default=str(GATES_DIR / "results_dual_bound_20261003"
                                         / "dual_bound_loop.json"))
    args = ap.parse_args(argv)
    lo, hi = args.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    result = {
        "schema_version": SCHEMA_VERSION,
        "work_package": "T08 (Issue #19, PR #18 docs TODO)",
        "scope_discipline": [
            "U only from double-verified schedules",
            "L only from global/relaxation certificates",
            "sampling failure, empty proposal, UNKNOWN and timeouts never raise L",
            "pool-scope bounds are dropped on pool change",
        ],
        "d0": run_d0(seeds, pool_k=args.pool_k, time_budget=args.time_budget,
                     cert_time_limit=args.cert_time_limit,
                     cert_every=args.cert_every, max_rounds=args.max_rounds),
        "failure_injections": run_failure_injections(seed=seeds[0], pool_k=args.pool_k),
    }
    for row in result["d0"]:
        row["events"] = row["summary"].pop("events", [])  # full log for the JSONL
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=_json_default),
                   encoding="utf-8")
    events_path = out.parent / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for row in result["d0"]:
            handle.write(json.dumps({"scale": row["scale"], "seed": row["seed"],
                                     "events": row.pop("events", [])},
                                    ensure_ascii=False, default=_json_default) + "\n")
    result["event_log_path"] = events_path.name
    violations = sum(len(row["invariant_violations"]) for row in result["d0"])
    raised = sum(1 for row in result["failure_injections"] if row["raised"])
    print(f"wrote {out} and {events_path.name}")
    print(f"D0 windows: {len(result['d0'])}; invariant violations: {violations}; "
          f"failure injections that wrongly raised L: {raised}")
    return 0


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj).__name__)


if __name__ == "__main__":
    sys.exit(main())
