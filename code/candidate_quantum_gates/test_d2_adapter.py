#!/usr/bin/env python3
"""Unit tests for the D2 frozen-window adapter (T04 -> T06/T10)."""
from __future__ import annotations

import os  # noqa: E402
import sys  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

GATES_DIR = Path(__file__).resolve().parent
if str(GATES_DIR) not in sys.path:
    sys.path.insert(0, str(GATES_DIR))

import d2_adapter  # noqa: E402
from window_diagnostics import check_choice  # noqa: E402

#: The T04 developer dataset is published from glm's clone (docs/agent_collaboration.md
#: workspace layout).  Set ``D2_DATASET`` to point elsewhere; the tests skip when
#: the file is absent so the suite stays runnable in a clean checkout.
D2_DATASET = Path(os.environ.get(
    "D2_DATASET",
    GATES_DIR.parents[2] / "improvement-glm" / "code" / "candidate_quantum_gates"
    / "results_window_pool_20261003" / "window_pool_d2_dev.json"))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        if not D2_DATASET.exists():
            self.skipTest(f"D2 dataset not available at {D2_DATASET}")

    def test_only_improving_windows_are_yielded_and_targets_are_pool_optima(self):
        rows = list(d2_adapter.iter_windows(D2_DATASET, only_improving=True))
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertFalse(row["pool_optimum"] is None)
            self.assertLess(row["target"], row["u0"])
            self.assertLessEqual(row["improving_combinations"], 729)

    def test_every_window_is_self_consistent(self):
        row = next(iter(d2_adapter.iter_windows(D2_DATASET, limit=1)))
        inst, pool, incumbent = row["inst"], row["pool"], row["incumbent"]
        self.assertEqual(len(pool), inst.machines)
        self.assertEqual(check_choice(inst, pool, incumbent)["makespan"], row["u0"])
        self.assertEqual(row["pool_sizes"], [len(machine) for machine in pool])

    def test_witnesses_are_reprojected_onto_this_pool(self):
        row = next(iter(d2_adapter.iter_windows(D2_DATASET, limit=1)))
        import medium_qjsp as mq
        for witness in row["witnesses"]:
            self.assertTrue(witness["support"] is not None)
            for machine in witness["support"]:
                self.assertLess(len(witness["allowed"][machine]),
                                len(row["pool"][machine]))
            projected = mq.witness_support(
                mq.Witness(witness["kind"],
                           tuple(tuple(r) for r in witness["relations"]),
                           witness["length"], ()), row["pool"])
            self.assertIsNotNone(projected)
        self.assertGreaterEqual(row["witnesses_dropped_constant_false"], 0)

    def test_dropped_witnesses_account_for_the_snapshot_set(self):
        row = next(iter(d2_adapter.iter_windows(D2_DATASET, limit=1)))
        payload = d2_adapter.load_dataset(D2_DATASET)
        key = f"{row['instance']}|s{row['seed']}|it{row['trajectory_iteration']}"
        snapshot = next(s for s in payload["d2"]["snapshots"]
                        if s["snapshot_key"] == key)
        tables = d2_adapter._load_witness_tables(D2_DATASET, payload["d2"])
        self.assertEqual(len(row["witnesses"])
                         + row["witnesses_dropped_constant_false"],
                         len(d2_adapter._snapshot_witnesses(snapshot, tables)))

    def test_rejects_a_non_d2_dataset(self):
        with self.assertRaises(ValueError):
            d2_adapter.load_dataset(GATES_DIR / "test_d2_adapter.py")

    def test_out_of_range_witness_index_fails_loudly(self):
        tables = {"inst|s0": [{"kind": "path", "length": 3,
                               "relations_flat": [0, 1, 2]}]}
        snapshot = {"snapshot_key": "inst|s0|it0", "trajectory_key": "inst|s0",
                    "witness_indices": [0]}
        self.assertEqual(len(d2_adapter._snapshot_witnesses(snapshot, tables)), 1)
        with self.assertRaises(ValueError):
            d2_adapter._snapshot_witnesses(dict(snapshot, witness_indices=[1]), tables)

    def test_missing_trajectory_table_fails_loudly(self):
        snapshot = {"snapshot_key": "inst|s0|it0", "trajectory_key": "other|s9",
                    "witness_indices": [0]}
        with self.assertRaises(ValueError):
            d2_adapter._snapshot_witnesses(snapshot, {"inst|s0": []})
        # the legacy embedded format is unaffected by the guard
        legacy = {"snapshot_key": "inst|s0|it0", "witnesses": [{"kind": "path"}]}
        self.assertEqual(d2_adapter._snapshot_witnesses(legacy, {}),
                         [{"kind": "path"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
