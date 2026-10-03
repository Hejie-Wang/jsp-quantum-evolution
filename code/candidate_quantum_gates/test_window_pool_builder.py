#!/usr/bin/env python3
"""T04 unit tests: pool builder correctness and oracle-layer invariants."""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "candidate_quantum_medium"))

import data_identity
import medium_qjsp as mq
import window_pool_builder as wpb
from window_diagnostics import check_choice


class BuilderBasics(unittest.TestCase):
    def test_pool_invariants_all_strategies(self):
        """Incumbent retention, completeness and budget bounds on a 3x3."""
        inst = data_identity.build_d0(3, 3, 0)
        snaps = wpb.classical_trajectory(inst, 0, [12])
        snap = snaps[0]
        for strategy in wpb.STRATEGIES:
            pool, meta = wpb.build_pool(
                inst, snap, strategy, s_max=6, k_max=3,
                rng=wpb._stable_rng(0, 12, wpb._strategy_index(strategy)))
            mq.validate_pool(inst, pool)
            self.assertEqual(len(pool), inst.machines)
            for m in range(inst.machines):
                self.assertEqual(len(pool[m]) <= 3, True)
                self.assertEqual(
                    pool[m][0], tuple(int(v) for v in snap.orders[m]),
                    f"incumbent not candidate 0: strategy={strategy} m={m}")
            selected = set(meta["selected_machines"])
            self.assertLessEqual(len(selected), 6)
            for m in range(inst.machines):
                if m not in selected:
                    self.assertEqual(meta["alternative_sources"][str(m)],
                                     ["incumbent_only"])

    def test_window_selection_matches_strategy(self):
        """critical_path picks machines with tight pairs; witness uses evidence."""
        inst = data_identity.build_d0(4, 3, 1)
        snap = wpb.classical_trajectory(inst, 1, [10])[0]
        scores = wpb._machine_scores(inst, snap, "critical_path")
        pool, meta = wpb.build_pool(inst, snap, "critical_path", 3, 3,
                                    wpb._stable_rng(1, 10, 1))
        positive = {int(m) for m in range(inst.machines) if scores[m] > 0}
        chosen = set(meta["selected_machines"])
        self.assertTrue(positive <= chosen,
                        "critical machines must be inside the window")
        # witness score comes from raw relations; every machine with a
        # witness relation must be ranked, and the strategy must be able
        # to select it when it is among the highest scorers.
        w_scores = wpb._machine_scores(inst, snap, "witness")
        self.assertEqual(int(w_scores.sum()),
                         sum(len(w["relations"]) for w in snap.witnesses))

    def test_frozen_record_roundtrip_and_oracle(self):
        """freeze -> evaluate: retention, u0 verification, reprojection zero."""
        inst = data_identity.build_d0(3, 3, 2)
        snap = wpb.classical_trajectory(inst, 2, [5])[0]
        pool, meta = wpb.build_pool(inst, snap, "witness", 4, 3,
                                    wpb._stable_rng(2, 5, 2))
        rec = wpb.freeze_window(inst, snap, "witness", pool, meta,
                                dataset="D0", seed=2)
        self.assertEqual(rec["pool_sha256"], mq.content_hash(rec["pool"]))
        out = wpb.evaluate_pool(inst, rec, store_makespans=True)
        self.assertTrue(out["incumbent_retention"])
        self.assertTrue(out["reprojection"]["all_zero"])
        self.assertEqual(len(out["makespans"]), int(np.prod(rec["pool_sizes"])))
        # enumeration == brute-force decode on the same pool (checker sanity)
        brute = min((r["makespan"] for r in
                     (mq.decode(inst, [pool[m][a] for m, a in enumerate(c)],
                                validate=False)
                      for c in itertools.product(*(range(len(p))
                                                   for p in pool)))
                     if r["feasible"]), default=None)
        self.assertEqual(out["pool_optimum"], brute)

    def test_witness_cycle_implies_infeasible_on_new_pool(self):
        """Any raw cycle witness from history must reject combinations on a
        rebuilt pool (the T04 reprojection property, verified directly)."""
        inst = data_identity.build_d0(4, 3, 3)
        snap = wpb.classical_trajectory(inst, 3, [8])[0]
        cycles = [w for w in snap.witnesses if w["kind"] == "cycle"]
        pool, _meta = wpb.build_pool(inst, snap, "critical_path", 4, 3,
                                     wpb._stable_rng(3, 8, 1))
        for w in cycles:
            allowed = wpb._independent_allowed(
                inst, pool, [tuple(int(x) for x in r) for r in w["relations"]])
            if allowed is None:
                continue
            for choice in itertools.product(*(range(len(p)) for p in pool)):
                if all(choice[m] in allowed[m] for m in allowed):
                    self.assertFalse(
                        check_choice(inst, pool, choice)["feasible"],
                        f"cycle witness triggered on a feasible combination: {w}")

    def test_projection_primitive_matches_independent_recomputation(self):
        """mq.witness_support agrees with the independent allowed-set check."""
        inst = data_identity.build_d0(3, 3, 4)
        snap = wpb.classical_trajectory(inst, 4, [3])[0]
        pool, _meta = wpb.build_pool(inst, snap, "random", 5, 3,
                                     wpb._stable_rng(4, 3, 0))
        for w in snap.witnesses:
            rels = tuple(tuple(int(x) for x in r) for r in w["relations"])
            mine = wpb._independent_allowed(inst, pool, rels)
            theirs = mq.witness_support(
                mq.Witness(w["kind"], rels, int(w["length"]), ()), pool)
            self.assertEqual(mine is None, theirs is None)
            if mine is not None:
                self.assertEqual({m: sorted(a) for m, a in mine.items()},
                                 {m: sorted(a) for m, a in theirs[1].items()})

    def test_determinism_of_frozen_record(self):
        """Same seed -> identical pool hash (cross-process reproducibility)."""
        inst = data_identity.build_d0(2, 2, 0)
        hashes = set()
        for _ in range(2):
            snap = wpb.classical_trajectory(inst, 7, [4])[0]
            pool, meta = wpb.build_pool(inst, snap, "witness", 2, 3,
                                        wpb._stable_rng(7, 4, 2))
            rec = wpb.freeze_window(inst, snap, "witness", pool, meta,
                                    dataset="D0", seed=7)
            hashes.add(rec["pool_sha256"])
        self.assertEqual(len(hashes), 1)

    def test_no_improvement_gate_summary(self):
        """summarize computes the >=20% relative-reduction gate vs random."""
        evals = []
        for strategy, ratio, n in (("random", 0.8, 10), ("critical_path", 0.6, 10),
                                   ("witness", 0.7, 10)):
            for i in range(n):
                no_imp = i < round(ratio * n)
                evals.append({
                    "strategy": strategy, "dataset": "D2-dev", "instance": "x",
                    "seed": i, "trajectory_iteration": 0,
                    "total_combinations": 27, "feasible_rate": 1.0,
                    "pool_optimum": 10, "rho": 0.0,
                    "improving_combinations": 0 if no_imp else 1,
                    "d_imp_from_incumbent": None if no_imp else 1,
                    "no_improvement": no_imp, "incumbent_retention": True,
                    "candidate_completeness": True,
                    "reprojection": {"checks": {
                        "projection_mismatch": 0, "unsupported_dropped": 0,
                        "cycle_witness_without_cycle": 0,
                        "path_witness_below_length": 0,
                        "false_exclusions": 0},
                        "witnesses_kept": 1, "witnesses_raw": 1,
                        "all_zero": True},
                    "improving_needs_random_perm": None,
                    "enumeration_seconds": 0.0,
                })
        summary = wpb.summarize(evals, "synthetic")["strategies"]
        self.assertAlmostEqual(summary["random"]["no_improvement_ratio"], 0.8)
        self.assertAlmostEqual(
            summary["critical_path"]["relative_reduction_vs_random"], 0.25)
        self.assertTrue(summary["critical_path"]["gate_20pct"])
        self.assertAlmostEqual(
            summary["witness"]["relative_reduction_vs_random"], 0.125)
        self.assertFalse(summary["witness"]["gate_20pct"])


if __name__ == "__main__":
    unittest.main()
