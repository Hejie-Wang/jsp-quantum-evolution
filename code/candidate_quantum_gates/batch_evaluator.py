"""Exact batched DAG makespans, CPU/Numba or optional CuPy CUDA kernel."""
from pathlib import Path
from time import perf_counter
import numpy as np
from numba import njit

INVALID = 1 << 30

# CUDA memory budgets (bytes) for the whole-block upload and the per-chunk
# scratch buffer; the scratch is additionally clamped by device free memory.
# 64 MB keeps the (4, n, b) working set close to L2 on laptop GPUs, which the
# chunk sweep shows is the throughput sweet spot.
MAX_ORDERS_BYTES = 1 << 30
MAX_WORK_BYTES = 64 << 20


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


def _auto_chunk(operations, free_bytes):
    """Chunk keeping the (4, n, b) scratch buffer under MAX_WORK_BYTES."""
    per_candidate = 4 * operations * 4
    by_memory = max(1, min(free_bytes // 2, MAX_WORK_BYTES) // per_candidate)
    return max(4096, by_memory)


class BatchEvaluator:
    def __init__(self, inst, backend="cpu", chunk_size=None):
        if backend not in {"cpu", "cuda", "auto"}:
            raise ValueError("invalid backend")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("invalid chunk size")
        if inst.total_duration >= INVALID:
            raise ValueError("instance exceeds int32 time budget")
        self.inst = inst
        self.processing = np.asarray(inst.processing, dtype=np.int32)
        self.backend, self.fallback_reason = "cpu", None
        self.seconds, self.evaluations, self.batches = 0.0, 0, 0
        self._dev_bufs = {}
        if backend != "cpu":
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("no CUDA device")
                self.cp = cp
                self.kernel = cp.RawKernel(Path(__file__).with_name("evaluate_batch.cu").read_text(),
                                           "evaluate_batch")
                self.kernel.compile()
                self.transpose = cp.RawKernel(Path(__file__).with_name("transpose_bm_mj.cu").read_text(),
                                              "transpose_bm_mj")
                self.transpose.compile()
                self.gpu_processing = cp.asarray(self.processing)
                free = cp.cuda.runtime.memGetInfo()[0]
                self.chunk_size = (chunk_size if chunk_size is not None
                                   else _auto_chunk(inst.operations, free))
                self.backend = "cuda"
            except Exception as exc:
                if backend == "cuda":
                    raise RuntimeError("CUDA requested but unavailable") from exc
                self.fallback_reason = str(exc)
        if self.backend == "cpu":
            self.chunk_size = chunk_size if chunk_size is not None else 1024

    def _validate(self, array):
        for m, group in enumerate(self.inst.groups):
            if np.any(np.sort(array[:, m, :], axis=1) != np.array(group)):
                raise ValueError("invalid machine order")

    def _dev_buffers(self, n, b):
        """Reuse scratch buffers across chunks and evaluate() calls."""
        bufs = self._dev_bufs.get((n, b))
        if bufs is None:
            bufs = self.cp.empty((4, n, b), dtype=self.cp.int32)
            self._dev_bufs[(n, b)] = bufs
        return bufs

    def _eval_cuda(self, array):
        """Upload in native layout, transpose on device, chunked kernel launches.

        The host does no data movement at all: the (B, M, J) batch goes to the
        device as-is, the tiled transpose kernel converts it to the (M, J, B)
        layout the evaluator expects, and results come back with a single
        device-to-host copy per block. The kernel indexes candidates through
        (base, stride), so one upload of a block serves many chunked launches
        without re-uploads.
        """
        cp = self.cp
        n, m, j = self.inst.operations, self.inst.machines, self.inst.jobs
        result = np.empty(len(array), np.int32)
        block = max(1, min(MAX_ORDERS_BYTES // (4 * m * j * 4), len(array)))
        for start in range(0, len(array), block):
            part = array[start:start + block]
            b = len(part)
            src = cp.asarray(part)
            orders = cp.empty((m * j, b), dtype=cp.int32)
            self.transpose(((m * j + 31) // 32, (b + 31) // 32), (32, 32),
                           (src, orders, np.int32(b), np.int32(m * j)))
            out = cp.empty(b, dtype=cp.int32)
            for i in range(0, b, self.chunk_size):
                end = min(i + self.chunk_size, b)
                work = self._dev_buffers(n, end - i)
                self.kernel(((end - i + 127) // 128,), (128,),
                            (orders, self.gpu_processing, work, out[i:end],
                             np.int32(end - i), np.int32(j), np.int32(m),
                             np.int32(i), np.int32(b)))
            out.get(out=result[start:start + b],
                    stream=cp.cuda.get_current_stream())
        # Synchronize before returning: timed sections must not push queued
        # work into the next call (WDDM otherwise accumulates command backlog).
        cp.cuda.get_current_stream().synchronize()
        return result

    def evaluate(self, batch, validate=True):
        array = np.asarray(batch)
        if array.ndim != 3 or array.shape[1:] != (self.inst.machines, self.inst.jobs):
            raise ValueError("expected batch x machines x jobs")
        if array.dtype.kind not in "iu":
            raise ValueError("operation IDs must be integers")
        if validate:
            self._validate(array)
        array = np.ascontiguousarray(array, dtype=np.int32)
        started = perf_counter()
        if self.backend == "cpu":
            outputs = []
            for i in range(0, len(array), self.chunk_size):
                outputs.append(evaluate_batch_cpu(array[i:i + self.chunk_size],
                                                  self.processing, self.inst.machines))
            values = np.concatenate(outputs) if outputs else np.empty(0, np.int32)
        else:
            values = self._eval_cuda(array)
        self.seconds += perf_counter() - started
        self.evaluations += len(array)
        self.batches += 1
        return values
