"""CUDA batch-evaluator benchmark.

Usage: python bench_cuda_baseline.py [batch_size] [chunk_size]
Generates a random 20x20 JSP instance and random valid machine orderings,
times CPU (numba) vs CUDA (CuPy RawKernel) evaluation, and checks that both
backends agree. chunk_size=0 uses the memory-aware auto default.

Timing notes: run each configuration in a fresh process (thermal drift on
laptops skews repeated in-process loops) and report best-of-3. Host-side
validation cost is reported separately from device evaluation.
"""
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "../candidate_quantum_medium")

from batch_evaluator import BatchEvaluator
from medium_qjsp import Instance


def make_instance(jobs=20, machines=20, seed=7):
    rng = np.random.default_rng(seed)
    durations = rng.integers(1, 100, size=(jobs, machines))
    routes = np.array([rng.permutation(machines) for _ in range(jobs)])
    return Instance(durations, routes, f"random_{jobs}x{machines}")


def make_batch(inst, size, seed=11):
    """Random valid candidates: each machine row is a permutation of its group."""
    rng = np.random.default_rng(seed)
    batch = np.empty((size, inst.machines, inst.jobs), dtype=np.int32)
    for m, group in enumerate(inst.groups):
        rows = np.array([rng.permutation(group) for _ in range(size)])
        batch[:, m, :] = rows
    return batch


def best_of(ev, batch, runs=3, validate=False):
    """Return the best wall-clock duration of one isolated evaluation.

    ``BatchEvaluator.seconds`` is cumulative by design, so reading it after a
    CUDA warmup would include earlier calls in the benchmark result.
    """
    times = []
    for _ in range(runs):
        started = perf_counter()
        ev.evaluate(batch, validate=validate)
        times.append(perf_counter() - started)
    return min(times)


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    inst = make_instance()
    batch = make_batch(inst, b)

    ev_cpu = BatchEvaluator(inst, backend="cpu")
    ref = ev_cpu.evaluate(batch)
    t_cpu = best_of(ev_cpu, batch)
    print(f"CPU  (numba): {t_cpu*1e3:9.1f} ms  {b/t_cpu/1e6:6.2f} M evals/s")

    ev_cuda = BatchEvaluator(inst, backend="cuda",
                             chunk_size=chunk if chunk > 0 else None)
    ev_cuda.evaluate(batch[: min(1024, b)])  # warmup / JIT, not counted
    t_cuda = best_of(ev_cuda, batch)
    assert np.array_equal(ref, ev_cuda.evaluate(batch, validate=False)), "CPU/CUDA mismatch!"
    print(f"CUDA chunk={ev_cuda.chunk_size}: {t_cuda*1e3:9.1f} ms  "
          f"{b/t_cuda/1e6:6.2f} M evals/s  ({t_cpu/t_cuda:5.1f}x vs CPU)")
    print("correctness: CUDA == CPU ✓")

    t_val = best_of(ev_cuda, batch, validate=True)
    print(f"host-side validate=True adds ~{t_val - t_cuda:.0f} ms (numpy sort, "
          "search loop uses validate=False)")


if __name__ == "__main__":
    main()
