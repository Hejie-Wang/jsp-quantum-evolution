"""Offline Qiskit circuits for witness-guided JSP candidate search."""

from .circuits import (
    CandidatePool,
    JointTransition,
    WitnessSpec,
    allowed_labels,
    build_phase_circuit,
    build_xy_mixer,
    build_joint_mixer,
    circuit_resources,
    decode_one_hot_counts,
    derive_witness,
    ensure_simulation_budget,
    prepare_legal_basis,
    prepare_uniform_legal,
)

__all__ = [
    "CandidatePool",
    "JointTransition",
    "WitnessSpec",
    "allowed_labels",
    "build_phase_circuit",
    "build_xy_mixer",
    "build_joint_mixer",
    "circuit_resources",
    "decode_one_hot_counts",
    "derive_witness",
    "ensure_simulation_budget",
    "prepare_legal_basis",
    "prepare_uniform_legal",
]
