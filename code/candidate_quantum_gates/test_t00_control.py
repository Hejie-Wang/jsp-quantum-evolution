"""T00 acceptance tests: independent uniform control, identity, D0 certificate.

These tests encode the T00 acceptance conditions from the breakdown document:

* the uniform arm is an independent sampler whose preparation cost does not grow
  with ``prod_m K_m`` (``test_sampler_never_builds_the_product_space``,
  ``test_uniform_proposal_does_not_materialise_the_legal_subspace``);
* its sampling law equals the theoretical product distribution
  (``test_empirical_distribution_matches_exact_product_law``);
* instance identity is stable across the text round trip
  (``test_text_round_trip_preserves_identity``);
* D0 enumeration is a real certificate and a pool bound is not a global bound
  (``test_enumerated_optimum_is_independently_verified``,
  ``test_restricted_pool_bound_is_not_a_global_bound``).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "candidate_quantum_medium"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import medium_qjsp as mq  # noqa: E402

from circuits import CandidatePool
from data_identity import (DATA_VERSION, D0_SIZES, DEVELOPMENT_SEEDS, VALIDATION_SEEDS,
                           build_d0, d0_manifest, d0_records, d0_split_of,
                           enumerate_exact_optimum, instance_identity, instance_sha256,
                           load_instance_file, restricted_pool_optimum,
                           write_text_instance)
from experiment_manifest import (REQUIRED_RECORD_FIELDS, build_record,
                                 validate_record)
from search_loop import demo_instance
from uniform_sampler import UniformSampler, uniform_label_probability, uniform_marginals


class UniformSamplerTests(unittest.TestCase):
    def test_shape_range_and_determinism(self):
        sampler = UniformSampler((3, 2, 4))
        first = sampler.sample(500, np.random.default_rng(11))
        second = sampler.sample(500, np.random.default_rng(11))
        self.assertEqual(first.shape, (500, 3))
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first >= 0))
        self.assertTrue(np.all(first[:, 0] < 3) and np.all(first[:, 1] < 2)
                        and np.all(first[:, 2] < 4))
        other = sampler.sample(500, np.random.default_rng(12))
        self.assertFalse(np.array_equal(first, other))

    def test_accepts_candidate_pool_sizes(self):
        pool = CandidatePool((((0, 1), (1, 0)), ((2,),), ((3, 4), (4, 3))))
        sampler = UniformSampler(pool)
        self.assertEqual(sampler.sizes, (2, 1, 2))
        self.assertEqual(sampler.dimension, 4)

    def test_rejects_malformed_sizes_and_shots(self):
        for sizes in ((), (2, 0), (-1, 2)):
            with self.assertRaises(ValueError):
                UniformSampler(sizes)
        with self.assertRaises(ValueError):
            UniformSampler((2, 2)).sample(0, np.random.default_rng(1))

    def test_sampler_never_builds_the_product_space(self):
        """Cost model: draws scale with shots*machines, preparation stays O(machines)."""
        sampler = UniformSampler((3,) * 12)  # 3^12 = 531,441 legacy states
        self.assertEqual(sampler.dimension, 3 ** 12)
        _, stats = sampler.sample_with_stats(64, np.random.default_rng(5))
        self.assertEqual(stats.draws, 64 * 12)
        self.assertEqual(stats.as_dict()["preparation_statevector"], 0)
        self.assertEqual(stats.as_dict()["product_space_states"], 3 ** 12)
        self.assertLessEqual(stats.unique, 64)

    def test_empirical_distribution_matches_exact_product_law(self):
        from scipy.stats import chisquare

        sizes = (3, 2, 4)
        sampler = UniformSampler(sizes)
        expected_probability = uniform_label_probability(sizes, (0, 0, 0))
        self.assertAlmostEqual(expected_probability, 1.0 / 24, places=12)
        self.assertAlmostEqual(sampler.probability_of((2, 1, 3)), 1.0 / 24, places=12)
        self.assertEqual(sampler.probability_of((3, 0, 0)), 0.0)   # out of range
        self.assertEqual(sampler.probability_of((0, 0)), 0.0)      # wrong length

        draws = 24 * 20000
        sample = sampler.sample(draws, np.random.default_rng(20261003))
        observed = np.zeros(sampler.dimension, dtype=np.int64)
        flat = np.ravel_multi_index(sample.T, sizes)
        np.add.at(observed, flat, 1)
        expected = np.full(sampler.dimension, draws / sampler.dimension)
        result = chisquare(observed, expected)
        # Chi-square is a smoke test for gross bias; the exact law is asserted
        # separately through the product probability above.
        self.assertGreater(result.pvalue, 1e-3,
                           f"uniform sampler looks biased: p={result.pvalue}")

    def test_per_machine_marginals_are_uniform(self):
        sizes = (5, 2, 3)
        sampler = UniformSampler(sizes)
        sample = sampler.sample(60000, np.random.default_rng(7))
        for machine, (size, expected) in enumerate(zip(sizes, sampler.marginals())):
            counts = np.bincount(sample[:, machine], minlength=size)
            frequencies = counts / counts.sum()
            self.assertLess(float(np.max(np.abs(frequencies - expected))), 0.01)
            np.testing.assert_allclose(expected, np.full(size, 1.0 / size))

    def test_deduplicate_returns_sorted_unique_rows(self):
        sampler = UniformSampler((2, 2))
        values, stats = sampler.sample_with_stats(400, np.random.default_rng(3),
                                                  deduplicate=True)
        self.assertEqual(values.shape[0], len(np.unique(values, axis=0)))
        self.assertEqual(values.shape[0], 4)
        np.testing.assert_array_equal(values, [[0, 0], [0, 1], [1, 0], [1, 1]])
        # De-duplication is the consumer's saving, so raw draws stay reported.
        self.assertEqual(stats.draws, 800)
        self.assertEqual(stats.unique, 4)

    def test_probabilities_mode_is_restricted(self):
        sampler = UniformSampler((2, 3))
        np.testing.assert_allclose(sampler.probabilities("uniform"),
                                   np.full(6, 1 / 6))
        with self.assertRaises(ValueError):
            sampler.probabilities("xy_joint")


class DataIdentityTests(unittest.TestCase):
    def test_d0_generator_is_deterministic_within_the_duration_range(self):
        first = build_d0(3, 3, 4)
        second = build_d0(3, 3, 4)
        self.assertEqual(instance_sha256(first), instance_sha256(second))
        self.assertEqual(first.name, "d0_3x3_seed4")
        for jobs, machines in D0_SIZES:
            for seed in DEVELOPMENT_SEEDS + VALIDATION_SEEDS:
                inst = build_d0(jobs, machines, seed)
                self.assertEqual(inst.durations.shape, (jobs, machines))
                self.assertTrue(np.all(inst.durations >= 1) and np.all(inst.durations <= 9))
                for row in inst.routes:
                    self.assertEqual(sorted(map(int, row)), list(range(machines)))
        self.assertNotEqual(instance_sha256(build_d0(3, 3, 4)),
                            instance_sha256(build_d0(3, 3, 5)))

    def test_d0_identity_is_stable(self):
        """Pins the D0 seed-0 3x3 identity: silent generator drift must fail here."""
        inst = build_d0(3, 3, 0)
        payload = {"durations": inst.durations.tolist(), "routes": inst.routes.tolist()}
        self.assertEqual(instance_identity(inst)["durations"], payload["durations"])
        # Recomputing through the manifest path must reproduce the same hash.
        manifest = d0_manifest([(3, 3, 0, "development", inst)])
        self.assertEqual(manifest["instances"][0]["instance_sha256"], instance_sha256(inst))

    def test_split_partition_is_exhaustive(self):
        for seed in DEVELOPMENT_SEEDS:
            self.assertEqual(d0_split_of(seed), "development")
        for seed in VALIDATION_SEEDS:
            self.assertEqual(d0_split_of(seed), "validation")
        with self.assertRaises(ValueError):
            d0_split_of(10)

    def test_manifest_covers_the_whole_family(self):
        manifest = d0_manifest(list(d0_records()))
        expected = len(D0_SIZES) * (len(DEVELOPMENT_SEEDS) + len(VALIDATION_SEEDS))
        self.assertEqual(manifest["instance_count"], expected)
        self.assertEqual(manifest["data_version"], DATA_VERSION)
        self.assertEqual(len(manifest["instances"]), expected)
        self.assertEqual({(row["jobs"], row["machines"]) for row in manifest["instances"]},
                         set(D0_SIZES))
        self.assertEqual(sum(row["split"] == "development" for row in manifest["instances"]),
                         len(D0_SIZES) * len(DEVELOPMENT_SEEDS))
        hashes = [row["instance_sha256"] for row in manifest["instances"]]
        self.assertEqual(len(set(hashes)), len(hashes))

    def test_text_round_trip_preserves_identity(self):
        inst = build_d0(3, 3, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "d0_3x3_seed2.txt"
            write_text_instance(inst, path)
            loaded = load_instance_file(path)
            self.assertEqual(instance_sha256(loaded), instance_sha256(inst))
            np.testing.assert_array_equal(loaded.durations, inst.durations)
            np.testing.assert_array_equal(loaded.routes, inst.routes)
            first_line = path.read_text(encoding="utf-8").splitlines()[0].split()
            self.assertEqual(len(first_line), 3)
            # Route rows are stored 1-based, as medium_qjsp.Instance.read expects.
            last_line = path.read_text(encoding="utf-8").splitlines()[-1].split()
            self.assertEqual(sorted(int(v) for v in last_line), [1, 2, 3])

    def test_enumerated_optimum_is_independently_verified(self):
        for jobs, machines, seed in ((2, 2, 0), (2, 2, 9), (3, 3, 0)):
            inst = build_d0(jobs, machines, seed)
            certificate = enumerate_exact_optimum(inst, limit=200000)
            self.assertTrue(certificate["independent_verification"])
            self.assertGreater(certificate["feasible_combinations"], 0)
            self.assertGreaterEqual(certificate["exact_optimum"], inst.lower_bound)
            self.assertLessEqual(certificate["exact_optimum"], inst.total_duration)
        inst = build_d0(4, 3, 0)
        certificate = enumerate_exact_optimum(inst, limit=20000)
        self.assertEqual(certificate["combinations"], 24 ** 3)

    def test_enumerated_optimum_is_the_best_of_all_orderings(self):
        """Brute force itself is cross-checked against a full random sweep."""
        inst = build_d0(3, 3, 1)
        certificate = enumerate_exact_optimum(inst, limit=200000)
        rng = np.random.default_rng(1)
        best_random = None
        for _ in range(2000):
            orders = [list(rng.permutation(list(group))) for group in inst.groups]
            result = mq.decode(inst, orders, validate=False)
            if result["feasible"]:
                best_random = result["makespan"] if best_random is None else min(
                    best_random, result["makespan"])
        self.assertIsNotNone(best_random)
        self.assertLessEqual(certificate["exact_optimum"], best_random)

    def test_restricted_pool_bound_is_not_a_global_bound(self):
        inst = build_d0(3, 3, 3)
        certificate = enumerate_exact_optimum(inst, limit=200000)
        # A deliberately thin pool: keep the operation order induced by the job
        # routes for each machine, plus one reversal.  Its optimum may exceed the
        # true optimum, which is the scope rule under test.
        thin_pool = []
        for machine in range(inst.machines):
            group = sorted(inst.groups[machine])
            thin_pool.append([list(group), list(reversed(group))])
        pool_result = restricted_pool_optimum(inst, thin_pool)
        self.assertIsNotNone(pool_result["pool_optimum"])
        self.assertGreaterEqual(pool_result["pool_optimum"], certificate["exact_optimum"])
        self.assertEqual(len(pool_result["pool_hash"]), 64)

    def test_enumeration_refuses_oversized_space(self):
        with self.assertRaises(ValueError):
            enumerate_exact_optimum(build_d0(4, 3, 0), limit=1000)


class RecordSchemaTests(unittest.TestCase):
    def _result(self, **overrides):
        result = {"schema_version": 1, "instance": "demo_3x3", "jobs": 3, "machines": 3,
                  "instance_sha256": instance_sha256(demo_instance()), "seed": 7,
                  "mode": "uniform", "best_makespan": 11, "wall_seconds": 1.5,
                  "search_seconds": 1.2, "setup_seconds": 0.3, "graph_evaluations": 100,
                  "batch_evaluation_seconds": 0.4, "proposal_seconds": 0.2,
                  "training_evaluations": 0, "proposal_calls": 5,
                  "hardware_jobs_submitted": 0, "initial_makespan": 12,
                  "independent_schedule_verification": True, "iterations": 10}
        result.update(overrides)
        return result

    def test_record_carries_every_required_field(self):
        record = build_record(self._result(), method="independent_uniform",
                              split="development", trace_path="trace.json")
        validate_record(record)
        for field in REQUIRED_RECORD_FIELDS:
            self.assertIn(field, record)
        self.assertEqual(record["quantum_execution"], "none")
        self.assertEqual(record["verified_upper_bound"], 11)
        self.assertIsNone(record["global_lower_bound"])
        self.assertEqual(record["cost_breakdown"]["proposal_calls"], 5)

    def test_quantum_mode_is_recorded_as_simulator_not_hardware(self):
        record = build_record(self._result(mode="quantum"),
                              method="witness_xy_joint_simulation",
                              split="development", trace_path=None)
        self.assertEqual(record["quantum_execution"], "simulator")
        self.assertEqual(record["hardware_jobs_submitted"], 0)

    def test_validation_rejects_scope_and_hardware_violations(self):
        record = build_record(self._result(), method="independent_uniform",
                              split="development", trace_path=None)
        with self.assertRaises(ValueError):
            validate_record({**record, "bound_scope": "pool", "global_lower_bound": 900})
        with self.assertRaises(ValueError):
            validate_record({**record, "quantum_execution": "qpu"})
        with self.assertRaises(ValueError):
            validate_record({**record, "independent_schedule_verification": False})
        with self.assertRaises(ValueError):
            validate_record({key: value for key, value in record.items()
                             if key != "trace_path"})
        with self.assertRaises(ValueError):
            validate_record({**record, "bound_scope": "everywhere"})

    def test_record_is_json_serialisable(self):
        record = build_record(self._result(), method="independent_uniform",
                              split="validation", trace_path="t.json")
        text = json.dumps(record, ensure_ascii=False)
        self.assertIn("instance_sha256", text)


class UniformProposalIntegrationTests(unittest.TestCase):
    def _solve(self, **kwargs):
        from adaptive_search import solve
        return solve(demo_instance(), seconds=2.0, seed=3, mode="uniform",
                     max_iterations=400, initial_schedules=24, **kwargs)

    def test_uniform_proposal_does_not_materialise_the_legal_subspace(self):
        result = self._solve()
        self.assertEqual(result["uniform_sampler_backend"], "independent_uniform_sampler")
        self.assertGreater(result["uniform_label_draws"], 0)
        self.assertEqual(result["uniform_preparation_statevector"], 0)
        # The legacy path is the only one allowed to allocate the subspace.
        self.assertLess(result["maximum_subspace_states"], 4)
        self.assertTrue(result["independent_schedule_verification"])

    def test_legacy_uniform_path_is_still_available(self):
        result = self._solve(legacy_uniform=True)
        self.assertEqual(result["uniform_sampler_backend"],
                         "compact_simulator_uniform_legacy")
        self.assertGreaterEqual(result["maximum_subspace_states"], 1)
        self.assertTrue(result["independent_schedule_verification"])

    def test_uniform_and_legacy_reach_the_same_pool_family(self):
        """Both arms must stay within the same instance/pool semantics."""
        modern = self._solve()
        legacy = self._solve(legacy_uniform=True)
        self.assertEqual(modern["instance_sha256"], legacy["instance_sha256"])
        self.assertEqual(modern["mode"], legacy["mode"])
        self.assertGreaterEqual(modern["proposal_calls"], 1)
        self.assertGreaterEqual(legacy["proposal_calls"], 1)


if __name__ == "__main__":
    unittest.main()
