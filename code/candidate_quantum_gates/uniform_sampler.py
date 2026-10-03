#!/usr/bin/env python3
"""Independent uniform control sampler over machine candidates (T00).

Why this module exists
----------------------
``adaptive_search.solve`` compares three arms that must differ in exactly one
place: the proposal module.  The classical arm is a tabu search, the quantum
arm is the exact simulation of the witness-phase/XY/joint ansatz, and the
uniform arm is the independent reference distribution
``q(a) = prod_m 1/K_m``.

Before T00 the uniform arm reused ``CompactSimulator``: it materialised the
whole legal subspace (``dimension = prod_m K_m`` amplitudes), built the witness
energy table over all of it, and only then sampled.  That attaches a
``O(prod_m K_m)`` preparation cost to the *classical* control, so the control
is not a cheap independent sampler and any comparison against it is biased.
This module draws one label per machine directly.  Preparation and memory are
``O(machines)``; the only cost that grows with the request is ``O(shots)``.

Sampling semantics
------------------
Each machine is drawn **independently and uniformly** from its candidate
labels, so the induced distribution over the product space is exactly uniform
on ``prod_m {0..K_m-1}`` and equals ``product.marginals``.  There is no
rejection, no repair and no coupling between machines; this is intentional and
is the mathematical reference point, not an approximation of it.  In
particular this sampler must never be described as a simulation of a quantum
circuit: it has no phases, no XY ring and no joint transition.

Cost accounting
---------------
Two quantities are reported separately, because they answer different
questions:

``draws``
    raw random labels requested by the caller (``shots * machines`` integer
    draws), the work a real device would also have to perform;
``unique``
    distinct label combinations after de-duplication, the work the full graph
    evaluator actually receives.

De-duplication is a property of the *consumer*, not of the sampler, so the
caller decides it via ``deduplicate``.  A cheap independent control must not be
charged for a full-table preparation it never performs, and it must not be
credited for evaluator savings it did not cause.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

__all__ = [
    "UniformSampler",
    "UniformSampleStats",
    "uniform_label_probability",
    "uniform_marginals",
]


def uniform_marginals(sizes) -> tuple[np.ndarray, ...]:
    """Exact per-machine marginal distribution (uniform on each machine)."""
    validated = _validated_sizes(sizes)
    return tuple(np.full(size, 1.0 / size) for size in validated)


def uniform_label_probability(sizes, labels) -> float:
    """Exact probability of one label combination under the reference law.

    ``prod_m 1/K_m`` for any in-range combination; no enumeration is needed,
    which is the point of the independent control.
    """
    validated = _validated_sizes(sizes)
    label = _validated_labels(labels, validated)
    probability = prod(1.0 / size for size in validated)
    return float(probability) if label is not None else 0.0


def _validated_sizes(sizes) -> tuple[int, ...]:
    validated = tuple(int(size) for size in sizes)
    if not validated:
        raise ValueError("at least one machine is required")
    if any(size < 1 for size in validated):
        raise ValueError("every machine needs at least one candidate")
    return validated


def _validated_labels(labels, sizes: tuple[int, ...]):
    array = np.asarray(labels, dtype=np.int64)
    if array.ndim != 1 or array.size != len(sizes):
        return None
    if np.any(array < 0) or np.any(array >= np.asarray(sizes, dtype=np.int64)):
        return None
    return array


@dataclass(frozen=True)
class UniformSampleStats:
    """Cost model of one independent uniform control draw."""

    shots: int
    machines: int
    draws: int
    unique: int
    # Recorded for the report: the subspace the legacy uniform path would have
    # materialised, listed only to show that this sampler never builds it.
    sizes: tuple[int, ...] = ()

    @property
    def product_space(self) -> int:
        return 0 if not self.sizes else prod(self.sizes)

    def as_dict(self) -> dict:
        return {
            "shots": int(self.shots),
            "machines": int(self.machines),
            "draws": int(self.draws),
            "unique": int(self.unique),
            "product_space_states": int(self.product_space),
            "preparation_statevector": 0,
            "backend": "independent_uniform_sampler",
        }


class UniformSampler:
    """Stateless independent uniform sampler over ``prod_m {0..K_m-1}``.

    ``sizes`` may be a sequence of per-machine candidate counts or any object
    exposing a ``sizes`` attribute (``CandidatePool`` does), which keeps the
    call sites free of adapter code.
    """

    def __init__(self, sizes):
        if hasattr(sizes, "sizes"):
            sizes = sizes.sizes
        self.sizes = _validated_sizes(sizes)
        self.machines = len(self.sizes)
        self.dimension = prod(self.sizes)
        self.label_probability = 1.0 / self.dimension

    def sample(self, shots, rng, *, deduplicate=False):
        """Draw ``shots`` independent label combinations.

        Returns an integer array of shape ``(shots, machines)`` ordered as
        drawn, or -- with ``deduplicate=True`` -- only the distinct rows, sorted
        lexicographically.  The same generator always produces the same rows,
        so a fixed seed makes a fixed trace reproducible.
        """
        shots = int(shots)
        if shots < 1:
            raise ValueError("positive shots required")
        choices = np.empty((shots, self.machines), dtype=np.int64)
        for machine, size in enumerate(self.sizes):
            choices[:, machine] = rng.integers(0, size, size=shots)
        if deduplicate:
            choices = np.unique(choices, axis=0)
        return choices

    def sample_with_stats(self, shots, rng, *, deduplicate=False):
        """``sample`` plus the cost model of the draw it just performed."""
        choices = self.sample(shots, rng, deduplicate=deduplicate)
        stats = UniformSampleStats(shots=int(shots), machines=self.machines,
                                   draws=int(shots) * self.machines,
                                   unique=int(len(choices)), sizes=self.sizes)
        return choices, stats

    def probabilities(self, mode: str = "uniform"):
        """Exact distribution on the product space.

        ``mode`` exists so callers that share one interface with
        ``CompactSimulator`` can pass ``"uniform"``; anything else is a
        programming error, because this sampler has exactly one law.
        """
        if mode != "uniform":
            raise ValueError("the independent uniform sampler only realises mode 'uniform'")
        return np.full(self.dimension, self.label_probability)

    def marginals(self):
        return uniform_marginals(self.sizes)

    def probability_of(self, labels) -> float:
        return uniform_label_probability(self.sizes, labels)

    def describe(self) -> dict:
        return {"backend": "independent_uniform_sampler",
                "machines": self.machines,
                "sizes": list(self.sizes),
                "product_space_states": self.dimension,
                "label_probability": self.label_probability,
                "preparation_statevector": 0}

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"UniformSampler(sizes={self.sizes})"
