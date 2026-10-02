"""Exact CPU simulation restricted to the legal candidate subspace.

This is a classical simulator, not a quantum device or a speedup claim. It
simulates the existing fixed-T, complete-graph XY, joint ansatz, without
allocating illegal one-hot states or clean work qubits. No schedule cost table
is built. Exponential memory is explicitly capped.
"""
from math import prod
from time import perf_counter
import numpy as np


class CompactSimulator:
    def __init__(self, pool, specs, transitions, fixed_t, penalty=1.0,
                 max_states=65536):
        self.sizes = tuple(pool.sizes)
        self.dimension = prod(self.sizes)
        if not self.sizes or self.dimension > max_states:
            raise ValueError("candidate subspace exceeds simulation budget")
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("penalty must be finite and positive")
        self.labels = np.array(np.unravel_index(np.arange(self.dimension), self.sizes)).T
        self.energy = np.zeros(self.dimension)
        self.xy_pairs, self.joint_pairs = [], []
        stride = [prod(self.sizes[m + 1:]) for m in range(len(self.sizes))]
        for spec in specs:
            spec.validate(pool)
            if spec.kind == "path" and spec.length <= fixed_t:
                continue
            active = np.ones(self.dimension, dtype=bool)
            for m, allowed in enumerate(spec.allowed):
                active &= np.isin(self.labels[:, m], allowed)
            self.energy += penalty * active
        for m, size in enumerate(self.sizes):
            for a in range(size):
                for b in range(a + 1, size):
                    left = np.flatnonzero(self.labels[:, m] == a)
                    self.xy_pairs.append((left, left + (b - a) * stride[m]))
        for action in transitions:
            action.validate(pool)
            active = np.ones(self.dimension, dtype=bool)
            offset = 0
            for m, a, b in zip(action.support, action.left, action.right):
                active &= self.labels[:, m] == a
                offset += (b - a) * stride[m]
            left = np.flatnonzero(active)
            self.joint_pairs.append((left, left + offset, action.weight))

    @staticmethod
    def _rotate(state, left, right, angle):
        a, b = state[left].copy(), state[right].copy()
        c, s = np.cos(angle), -1j * np.sin(angle)
        state[left], state[right] = c * a + s * b, s * a + c * b

    def state(self, params, mode="xy_joint", initial_state="uniform"):
        if mode not in {"uniform", "xy", "xy_joint"}:
            raise ValueError("unknown mode")
        if initial_state not in {"uniform", "basis"}:
            raise ValueError("unknown initial state")
        state = np.zeros(self.dimension, dtype=complex)
        if mode == "uniform" or initial_state == "uniform":
            state[:] = 1 / np.sqrt(self.dimension)
        else:
            state[0] = 1
        if mode != "uniform":
            for gamma, beta, joint in params:
                state *= np.exp(-1j * gamma * self.energy)
                for left, right in self.xy_pairs:
                    self._rotate(state, left, right, beta)
                if mode == "xy_joint":
                    for left, right, weight in self.joint_pairs:
                        self._rotate(state, left, right, joint * weight)
        return state

    def probabilities(self, params, mode="xy_joint"):
        p = np.abs(self.state(params, mode)) ** 2
        return p / p.sum()

    def train(self, *, mode="xy_joint", budget=24, seed=7, layers=2,
              deadline=None, warm_params=None):
        if budget < 1 or layers < 1:
            raise ValueError("positive training budget and layer count required")
        start = perf_counter()
        rng = np.random.default_rng(seed)
        # Normalize ONLY the surrogate in parameter selection, not its minima.
        scale = max(1.0, float(self.energy.max()))
        params = np.tile([0.7 / scale, 0.3, 0.2], (layers, 1))
        if warm_params is not None and np.shape(warm_params) == params.shape:
            params = np.array(warm_params, dtype=float)
        best = float("inf")
        used = 0
        for i in range(budget):
            if i and deadline is not None and perf_counter() >= deadline:
                break
            trial = params.copy()
            if i:
                k, axis = rng.integers(layers), rng.integers(3 if mode == "xy_joint" else 2)
                trial[k, axis] = rng.uniform(0, 2 * np.pi / scale if axis == 0 else np.pi)
            value = float(self.probabilities(trial, mode) @ self.energy)
            used += 1
            if value < best:
                best, params = value, trial
        return {"params": params.tolist(), "expected_energy": best,
                "evaluations": used, "seconds": perf_counter() - start,
                "backend": "exact_classical_candidate_subspace"}

    def sample(self, params, shots, rng, mode="xy_joint"):
        if shots < 1:
            raise ValueError("positive shots required")
        draws = rng.choice(self.dimension, shots, p=self.probabilities(params, mode))
        return self.labels[np.unique(draws)]
