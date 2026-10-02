"""Exact batched DAG makespans, CPU/Numba or optional CuPy CUDA kernel."""
from pathlib import Path
from time import perf_counter
import numpy as np
from numba import njit

INVALID = 1 << 30


@njit(cache=True)
def evaluate_one(orders, processing, machines):
    n = len(processing)
    successor = np.full(n, -1, np.int32)
    degree = np.zeros(n, np.int32)
    head = np.zeros(n, np.int32)
    tail = np.zeros(n, np.int32)
    queue = np.empty(n, np.int32)
    for v in range(n):
        if v % machines:
            degree[v] = 1
    for row in orders:
        for k in range(1, len(row)):
            successor[row[k - 1]] = row[k]
            degree[row[k]] += 1
    size = 0
    for v in range(n):
        if degree[v] == 0:
            queue[size] = v
            size += 1
    i, makespan = 0, 0
    while i < size:
        u = queue[i]
        i += 1
        finish = head[u] + processing[u]
        makespan = max(makespan, finish)
        for edge in range(2):
            v = u + 1 if edge == 0 and u % machines < machines - 1 else -1
            if edge == 1:
                v = successor[u]
            if v >= 0:
                head[v] = max(head[v], finish)
                degree[v] -= 1
                if degree[v] == 0:
                    queue[size] = v
                    size += 1
    if size != n:
        return INVALID, head, tail
    for i in range(n - 1, -1, -1):
        u = queue[i]
        for edge in range(2):
            v = u + 1 if edge == 0 and u % machines < machines - 1 else -1
            if edge == 1:
                v = successor[u]
            if v >= 0:
                tail[u] = max(tail[u], processing[v] + tail[v])
    return makespan, head, tail


@njit(cache=True)
def evaluate_batch_cpu(batch, processing, machines):
    values = np.empty(len(batch), np.int32)
    for i in range(len(batch)):
        values[i] = evaluate_one(batch[i], processing, machines)[0]
    return values


class BatchEvaluator:
    def __init__(self, inst, backend="cpu", chunk_size=1024):
        if backend not in {"cpu", "cuda", "auto"} or chunk_size < 1:
            raise ValueError("invalid backend or chunk size")
        if inst.total_duration >= INVALID:
            raise ValueError("instance exceeds int32 time budget")
        self.inst, self.chunk_size = inst, chunk_size
        self.processing = np.asarray(inst.processing, dtype=np.int32)
        self.backend, self.fallback_reason = "cpu", None
        self.seconds, self.evaluations, self.batches = 0.0, 0, 0
        if backend != "cpu":
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("no CUDA device")
                self.cp = cp
                self.kernel = cp.RawKernel(Path(__file__).with_name("evaluate_batch.cu").read_text(),
                                           "evaluate_batch")
                self.kernel.compile()
                self.gpu_processing = cp.asarray(self.processing)
                self.backend = "cuda"
            except Exception as exc:
                if backend == "cuda":
                    raise RuntimeError("CUDA requested but unavailable") from exc
                self.fallback_reason = str(exc)

    def evaluate(self, batch, validate=True):
        array = np.asarray(batch)
        if array.ndim != 3 or array.shape[1:] != (self.inst.machines, self.inst.jobs):
            raise ValueError("expected batch x machines x jobs")
        if array.dtype.kind not in "iu":
            raise ValueError("operation IDs must be integers")
        if validate:
            for m, group in enumerate(self.inst.groups):
                if np.any(np.sort(array[:, m, :], axis=1) != np.array(group)):
                    raise ValueError("invalid machine order")
        array = np.ascontiguousarray(array, dtype=np.int32)
        started = perf_counter()
        outputs = []
        for i in range(0, len(array), self.chunk_size):
            part = array[i:i + self.chunk_size]
            if self.backend == "cpu":
                values = evaluate_batch_cpu(part, self.processing, self.inst.machines)
            else:
                cp, b, n = self.cp, len(part), self.inst.operations
                orders = cp.asarray(np.ascontiguousarray(part.transpose(1, 2, 0)))
                work = cp.empty((4, n, b), dtype=cp.int32)
                out = cp.empty(b, dtype=cp.int32)
                self.kernel(((b + 127) // 128,), (128,),
                            (orders, self.gpu_processing, work, out,
                             np.int32(b), np.int32(self.inst.jobs), np.int32(self.inst.machines)))
                values = cp.asnumpy(out)  # synchronizes; includes transfer in timing
            outputs.append(values)
        self.seconds += perf_counter() - started
        self.evaluations += len(array)
        self.batches += 1
        return np.concatenate(outputs) if outputs else np.empty(0, np.int32)
