import unittest
from unittest.mock import patch

from bench_cuda_baseline import best_of


class _Evaluator:
    def __init__(self):
        self.seconds = 100.0
        self.calls = []

    def evaluate(self, batch, validate=True):
        self.calls.append((batch, validate))
        self.seconds += 10.0


class BenchmarkTimingTests(unittest.TestCase):
    def test_best_of_uses_per_call_wall_time(self):
        evaluator = _Evaluator()
        with patch("bench_cuda_baseline.perf_counter", side_effect=(
            1.0, 1.8, 2.0, 4.5, 5.0, 5.4
        )):
            elapsed = best_of(evaluator, "batch", runs=3, validate=False)

        self.assertAlmostEqual(elapsed, 0.4)
        self.assertEqual(evaluator.calls, [("batch", False)] * 3)
        self.assertEqual(evaluator.seconds, 130.0)


if __name__ == "__main__":
    unittest.main()
