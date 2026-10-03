#!/usr/bin/env python3
"""T06 unit tests: arm parity, raw-frequency discipline, ablation toggles."""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "candidate_quantum_medium"))

import data_identity
import medium_qjsp as mq
import sampling_ablation as sa
import window_pool_builder as wpb
from compact_simulator import CompactSimulator
from search_loop import graph_witness_specs
from witness_surrogate import collect_witnesses
from window_diagnostics import build_window, check_choice


def small_context(seed=0):
    inst = data_identity.build_d0(3, 3, seed)
    pool, incumbent, u0 = build_window(inst, 3, seed + 1000)
    dicts, _ = collect_witnesses(inst, pool, incumbent, n_probe=8, seed=seed)
    witnesses = [mq.Witness(w["kind"],
                            tuple(tuple(r) for r in w["relations"]),
                            int(w["length"]), ()) for w in dicts]
    target = u0 - 1
    return inst, pool, witnesses, target, incumbent


class ArmParityTests(unittest.TestCase):
    def test_all_arms_same_information(self):
        """Every arm is built from the same pool/specs/transitions/target."""
        inst, pool, witnesses, target, incumbent = small_context()
        pool_obj = sa.CandidatePool(pool)
        specs = [s for s in graph_witness_specs(pool_obj, witnesses)
                 if any(0 < len(a) < size for a, size in
                        zip(s.allowed, pool_obj.sizes))]
        sim = CompactSimulator(pool_obj, specs, (), target)
        surrogate = sa.surrogate_view(pool_obj, specs, target, 1.0)
        self.assertEqual(sim.sizes, pool_obj.sizes)
        self.assertEqual(len(surrogate["labels"]), sim.dimension)
        for labels, energy in zip(surrogate["labels"], sim.energy):
            self.assertAlmostEqual(surrogate["energy_of"](tuple(labels)),
                                   float(energy), places=9)

    def test_uniform_arm_is_independent_labels(self):
        """The uniform arm draws raw per-machine labels, never a subspace."""
        sizes = (3, 3, 3)
        samples = sa.arm_uniform(sizes, 256, np.random.default_rng(0))
        self.assertEqual(len(samples), 256)
        self.assertTrue(all(len(s) == 3 for s in samples))
        self.assertTrue(all(0 <= a < sizes[m] for s in samples
                            for m, a in enumerate(s)))
        self.assertGreater(len(set(samples)), 1)  # raw draws, not deduplicated


class AblationToggleTests(unittest.TestCase):
    def test_mixers_only_has_zero_surrogate_energy(self):
        """The phase ablation empties the witness set: energy all zero."""
        inst, pool, witnesses, target, _inc = small_context()
        pool_obj = sa.CandidatePool(pool)
        specs = graph_witness_specs(pool_obj, witnesses)
        full = CompactSimulator(pool_obj, specs, (), target)
        empty = CompactSimulator(pool_obj, (), (), target)
        self.assertGreater(float(full.energy.max()), 0.0)
        self.assertEqual(float(empty.energy.max()), 0.0)

    def test_warm_start_changes_the_sampling_distribution(self):
        """Same params, basis initial state != uniform initial state."""
        inst, pool, witnesses, target, _inc = small_context()
        pool_obj = sa.CandidatePool(pool)
        specs = [s for s in graph_witness_specs(pool_obj, witnesses)
                 if any(0 < len(a) < size for a, size in
                        zip(s.allowed, pool_obj.sizes))]
        sim = CompactSimulator(pool_obj, specs, (), target)
        trained = sim.train(mode="xy", budget=2, seed=3)
        params = np.array(trained["params"])
        warm = np.abs(sim.state(params, "xy", initial_state="basis")) ** 2
        cold = np.abs(sim.state(params, "xy", initial_state="uniform")) ** 2
        self.assertGreater(float(np.max(np.abs(warm - cold))), 1e-9)

    def test_finite_shots_training_counts_shots(self):
        inst, pool, witnesses, target, _inc = small_context()
        pool_obj = sa.CandidatePool(pool)
        specs = [s for s in graph_witness_specs(pool_obj, witnesses)
                 if any(0 < len(a) < size for a, size in
                        zip(s.allowed, pool_obj.sizes))]
        sim = CompactSimulator(pool_obj, specs, (), target)
        params, evals, seconds, shots = sa._train_finite_shots(
            sim, 3, np.random.default_rng(0), "xy_joint")
        self.assertEqual(evals, 3)
        self.assertEqual(shots, 3 * 64)


class RawFrequencyTests(unittest.TestCase):
    def test_run_window_keeps_raw_frequencies_and_oracle_separate(self):
        """raw_p_imp denominators are shots; oracle is reported, not fed back."""
        inst, pool, witnesses, target, incumbent = small_context()
        inst = data_identity.build_d0(3, 3, 0)
        pool_t, incumbent, u0 = build_window(inst, 3, 1000)
        witnesses, _ = collect_witnesses(inst, pool_t, incumbent, n_probe=8,
                                         seed=0)
        record = {
            "pool": pool_t, "incumbent": [0] * 3, "u0": int(u0),
            "target_t": int(u0) - 1, "snapshot_key": "test|s0|it0",
            "strategy": "critical_path", "instance": inst.name, "seed": 0,
            "trajectory_iteration": 0,
        }
        result = sa.run_window(
            inst, record, {"test|s0|it0": {"trajectory_key": "t",
                                           "trajectory_iteration": 0,
                                           "incumbent_orders": pool_t,
                                           "witness_indices": list(
                                               range(len(witnesses)))}},
            {"t": [{"kind": w["kind"], "length": int(w["length"]),
                    "relations_flat": [int(x) for r in w["relations"]
                                       for x in r]} for w in witnesses]},
            shots=64)
        for arm in result["arms"]:
            self.assertAlmostEqual(arm["raw_p_imp"],
                                   arm["raw_improving_mass"] / 64, places=12)
        self.assertEqual(len(result["arms"]), len(sa.ARMS))
        # oracle must equal a brute-force enumeration on the same pool
        brute = sum(
            1 for c in itertools.product(*(range(len(p)) for p in pool_t))
            if (r := check_choice(inst, pool_t, c))["feasible"]
            and r["makespan"] < u0)
        self.assertEqual(result["oracle"]["improving"], brute)


if __name__ == "__main__":
    unittest.main()
