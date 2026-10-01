"""Mathematical gates for the reference model, not production performance tests."""
import unittest
import numpy as np
import demo as d


class CandidateHamiltonianTests(unittest.TestCase):
    def test_decoder_fixture_and_independent_validation(self):
        observed = []
        for c in d.CHOICES:
            r = d.decode(c)
            observed.append(r.get("makespan"))
            if r["feasible"]:
                self.assertEqual(d.verify_starts(r["starts"]), r["makespan"])
        self.assertEqual(observed, [16, 16, 16, 11, None, None, None, 21])

    def test_witnesses_are_valid_across_all_choices(self):
        for w in d.covering_witnesses():
            for c in d.CHOICES:
                if not w.active(c):
                    continue
                r = d.decode(c)
                if w.kind == "cycle":
                    self.assertFalse(r["feasible"])
                elif r["feasible"]:
                    self.assertGreaterEqual(r["makespan"], w.length)

    def test_threshold_boundary_and_ground_correspondence(self):
        table = d.energy_table(d.covering_witnesses())
        self.assertEqual(int(table.min()), 11)
        for i, c in enumerate(d.CHOICES):
            r = d.decode(c)
            if r["feasible"]:
                t = r["makespan"]
                self.assertEqual(int(table[i, t]), t)
                self.assertGreater(int(table[i, t - 1]), d.U)
                self.assertEqual(int(table[i].min()), t)
            else:
                self.assertTrue(np.all(table[i] > d.U))

    def test_incomplete_model_and_exact_cut_certificate(self):
        self.assertEqual(int(d.energy_table([]).min()), 0)
        cuts, trace = d.exact_cut_loop()
        bounds = [row["pool_master_lower_bound"] for row in trace]
        self.assertEqual(bounds, sorted(bounds))
        self.assertGreater(len(cuts), 0)
        self.assertEqual(bounds[-1], 11)
        self.assertEqual(trace[-1]["makespan"], 11)

    def test_joint_driver_hermiticity_locality_and_uniform_initial_ground(self):
        specs = d.joint_specs(d.covering_witnesses())
        self.assertTrue(specs)
        self.assertTrue(all(2 <= len(support) <= 3 for support, _, _ in specs))
        h0, joint = d.drivers(specs)
        self.assertTrue(np.allclose(h0, h0.T))
        self.assertTrue(np.allclose(joint, joint.T))
        self.assertGreater(np.count_nonzero(joint), 0)
        uniform = np.ones(len(h0)) / np.sqrt(len(h0))
        self.assertTrue(np.allclose(h0 @ uniform, -uniform))
        self.assertLessEqual(np.linalg.norm(joint, ord=np.inf), 1.0 + 1e-12)

    def test_unitary_midpoint_evolution_and_endpoint_target(self):
        table = d.energy_table(d.covering_witnesses())
        h0, joint = d.drivers(d.joint_specs(d.covering_witnesses()))
        for strength in (0.0, 0.5):
            result = d.evolve(table, h0, joint, strength, tau=2, steps=8)
            self.assertLess(result["norm_error"], 1e-10)
            self.assertGreaterEqual(result["ground_probability"], 0)
            self.assertLessEqual(result["ground_probability"], 1 + 1e-12)


if __name__ == "__main__":
    unittest.main()
