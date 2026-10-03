#!/usr/bin/env python3
"""T05: QUBO, native-constraint and higher-order phase equivalence and resources.

Implements the T05 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md for one frozen predicate
set: the cycle witnesses plus the path witnesses with L_W > T of a frozen
candidate pool (exactly the cut set ``medium_qjsp.compile_cuts`` would build).

Four encodings of the *same* predicate set are compared:

  1. ``linear``   one-hot equalities plus one linear cut per witness
                  (sum_{m in S} z_{mW} <= |S|-1): the classical master problem.
  2. ``qubo``     squared-residual QUBO, built sparsely; the dense matrix is
                  cross-checked entry-by-entry against the production
                  ``medium_qjsp.build_qubo`` on the same cut set, so this module
                  never silently redefines the production encoding.
  3. ``and``      exact quadratization with AND auxiliary variables: the
                  k-ary product P_W = prod_{m in S} z_{mW} of the T03 surrogate
                  is replaced by a chain of binary AND gadgets, each with a
                  penalty that is zero exactly on y = p AND q, so existentially
                  eliminating the auxiliaries restores the original constraint.
  4. ``phase``    the native conditional phase of ``circuits.build_phase_circuit``
                  (compute membership flags -> multi-controlled phase ->
                  uncompute), verified on the legal one-hot subspace.

Claims verified (all counters must be zero):

  * zero-energy set equality: for every combination x, the minimum over
    auxiliaries of the QUBO energy and of the AND-quadratization energy is zero
    exactly when x is one-hot legal and satisfies every predicate of the set;
  * auxiliary elimination: the closed-form / DP minimum over auxiliaries equals
    brute-force enumeration of every auxiliary assignment wherever the
    auxiliary count is small enough to enumerate;
  * native phase: on the legal one-hot subspace the ideal-statevector phase of
    the built circuit equals (-1)^(number of triggered witnesses) with absolute
    error <= 1e-10, and every work qubit is uncomputed back to |0>;
  * exact cross-check at sizes that cannot be enumerated: CP-SAT and the
    production MILP master minimise the same energies and agree on whether the
    minimum is zero.

Scope discipline: zero energy is a statement about the FROZEN predicate set,
never a global JSP bound; the QUBO/AND minima are pool-side quantities; no
sampler, backend or hardware is involved anywhere in this module.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GATES_DIR = Path(__file__).resolve().parent
MEDIUM_DIR = ROOT / "code" / "candidate_quantum_medium"
for _p in (str(GATES_DIR), str(MEDIUM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import circuits  # noqa: E402
import medium_qjsp as mq  # noqa: E402
from window_diagnostics import check_choice  # noqa: E402
from witness_surrogate import collect_witnesses  # noqa: E402

SCHEMA_VERSION = 1
PHASE_TOLERANCE = 1e-10
ZERO_TOLERANCE = 1e-12
ONE_HOT_WEIGHT = 1.0
CUT_WEIGHT = 1.0
GADGET_WEIGHT = 1.0
PRODUCT_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Predicate set: exactly the cuts the production compiler would build at T
# ---------------------------------------------------------------------------

def _allowed_map(witness):
    """T03 witness dicts use int keys in memory and str keys after JSON."""
    allowed = witness.get("allowed") or {}
    out = {}
    for key, labels in allowed.items():
        out[int(key)] = tuple(int(a) for a in labels)
    return out


def compile_predicates(sizes, witnesses, target_t):
    """Cycle witnesses plus path witnesses with L_W > T, already projected.

    ``witnesses`` uses the T03 frozen format (``kind``, ``length``,
    ``support``, ``allowed``); constant-false witnesses were dropped upstream by
    ``witness_surrogate.collect_witnesses``.  A witness whose support is empty is
    constant true inside this pool, which makes the frozen predicate set
    infeasible by itself; that case is recorded explicitly instead of being
    silently dropped or treated as a satisfiable constraint.
    """
    cuts, constant_true = [], []
    for index, w in enumerate(witnesses):
        if w["kind"] == "path" and int(w["length"]) <= int(target_t):
            continue
        allowed = _allowed_map(w)
        support = tuple(int(m) for m in (w.get("support") or ()))
        support = tuple(m for m in support if m in allowed)
        cut = {"index": index, "kind": w["kind"], "length": int(w["length"]),
               "support": support, "allowed": allowed}
        if not support:
            constant_true.append(index)
        cuts.append(cut)
    return {
        "target_t": int(target_t),
        "sizes": [int(s) for s in sizes],
        "data_bits": int(sum(sizes)),
        "cuts": cuts,
        "n_cycle": sum(1 for c in cuts if c["kind"] == "cycle"),
        "n_path": sum(1 for c in cuts if c["kind"] == "path"),
        "constant_true_witnesses": constant_true,
        "infeasible_by_witness": bool(constant_true),
    }


def cut_triggered(cut, choice) -> bool:
    """P_W(a) == 1: every support machine selects a label allowed by the witness."""
    return all(choice[m] in cut["allowed"][m] for m in cut["support"])


def ref_violations(predicates, choice) -> int:
    return sum(1 for cut in predicates["cuts"] if cut_triggered(cut, choice))


def ref_feasible(predicates, choice) -> bool:
    return ref_violations(predicates, choice) == 0


def all_choices(sizes):
    return list(itertools.product(*(range(int(k)) for k in sizes)))


def _flat(sizes, machine, label):
    return sum(sizes[:machine]) + label


def slack_bits_of(cut):
    """Binary slack field size: enough values to represent 0..|S|-1."""
    size = len(cut["support"])
    return max(0, math.ceil(math.log2(size))) if size > 1 else 0


# ---------------------------------------------------------------------------
# Sparse quadratic form shared by encodings 2 and 3
# ---------------------------------------------------------------------------

class SparseQuadratic:
    """Sparse upper-triangular quadratic form: x^T Q x + constant.

    The construction mirrors ``medium_qjsp.build_qubo`` exactly (one squared
    residual per term, same slack ordering) so the dense view can be compared
    with the production matrix entry by entry.
    """

    def __init__(self, n_vars, label=""):
        self.n_vars = int(n_vars)
        self.label = label
        self.linear = [0.0] * self.n_vars
        self.quadratic = defaultdict(float)
        self.constant = 0.0

    def add_square(self, coeffs, d, weight=1.0):
        """Add weight * (sum_i coeffs[i] x_i + d)^2; dtype-stable and sparse."""
        if weight == 0.0:
            return
        indices = sorted(coeffs)
        for i in indices:
            ci = weight * coeffs[i]
            self.linear[i] += weight * (coeffs[i] * coeffs[i] + 2.0 * d * coeffs[i])
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                self.quadratic[(i, j)] += 2.0 * coeffs[i] * coeffs[j] * weight
        self.constant += weight * d * d

    def add_linear(self, index, value):
        self.linear[index] += value

    def add_quadratic(self, i, j, value):
        if i == j:
            self.linear[i] += value
            return
        self.quadratic[(min(i, j), max(i, j))] += value

    def energy(self, bits) -> float:
        bits = np.asarray(bits, dtype=float)
        if bits.shape != (self.n_vars,):
            raise ValueError("bit vector shape does not match the variable count")
        value = float(bits @ np.asarray(self.linear)) + self.constant
        for (i, j), coeff in self.quadratic.items():
            value += coeff * bits[i] * bits[j]
        return value

    def dense(self) -> np.ndarray:
        q = np.zeros((self.n_vars, self.n_vars))
        for i, value in enumerate(self.linear):
            q[i, i] += value
        for (i, j), value in self.quadratic.items():
            q[i, j] += value
        return q

    def metrics(self) -> dict:
        values = [v for v in self.linear if v != 0.0]
        values += [v for v in self.quadratic.values() if v != 0.0]
        magnitudes = [abs(v) for v in values] or [0.0]
        return {
            "n_vars": self.n_vars,
            "nnz_linear": int(sum(1 for v in self.linear if v != 0.0)),
            "nnz_quadratic": int(sum(1 for v in self.quadratic.values() if v != 0.0)),
            "nnz_total": int(len(values)),
            "coef_abs_min": float(min(magnitudes)),
            "coef_abs_max": float(max(magnitudes)),
            "coef_abs_ratio": float(max(magnitudes) / min(magnitudes)),
            "constant": float(self.constant),
        }


# ---------------------------------------------------------------------------
# Encoding 2: squared-residual QUBO (sparse)
# ---------------------------------------------------------------------------

def build_squared_residual_qubo(predicates, one_hot_weight=ONE_HOT_WEIGHT,
                                cut_weight=CUT_WEIGHT, max_vars=4096):
    """Sparse squared-residual QUBO over one-hot bits plus per-cut slack bits.

    ``E = A * sum_m (sum_a x_ma - 1)^2
        + sum_W B_W * (sum_{m in S_W} z_mW + sum_b 2^b s_b - (|S_W|-1))^2``

    with one slack field per witness.  Zero energy holds exactly when x is
    one-hot legal and every predicate of the frozen set is inactive.
    """
    sizes = predicates["sizes"]
    base = int(sum(sizes))
    slack_bits = [slack_bits_of(c) for c in predicates["cuts"]]
    total = base + sum(slack_bits)
    if total > max_vars:
        raise ValueError(f"QUBO needs {total} variables, cap is {max_vars}")
    form = SparseQuadratic(total, label="qubo")
    for m, size in enumerate(sizes):
        form.add_square({_flat(sizes, m, a): 1.0 for a in range(size)}, -1.0,
                        weight=one_hot_weight)
    offset = 0
    for cut, bits in zip(predicates["cuts"], slack_bits):
        coeffs = {}
        for m in cut["support"]:
            for a in cut["allowed"][m]:
                index = _flat(sizes, m, a)
                coeffs[index] = coeffs.get(index, 0.0) + 1.0
        for b in range(bits):
            coeffs[base + offset + b] = float(1 << b)
        offset += bits
        form.add_square(coeffs, -(len(cut["support"]) - 1.0), weight=cut_weight)
    meta = {"base_vars": base, "slack_bits": slack_bits, "total_vars": total,
            "aux_vars": total - base, "one_hot_weight": one_hot_weight,
            "cut_weight": cut_weight}
    return form, meta


def min_energy_over_aux_qubo(predicates, choice, one_hot_weight=ONE_HOT_WEIGHT,
                             cut_weight=CUT_WEIGHT) -> float:
    """Exact minimum over the slack fields for one combination.

    The slack fields are independent per witness, so the minimum is the sum of
    per-witness minima; this is exact without enumerating auxiliaries.
    """
    sizes = predicates["sizes"]
    value = 0.0
    for m, size in enumerate(sizes):
        active = 1 if 0 <= choice[m] < size else 0
        value += one_hot_weight * (active - 1) ** 2
    for cut in predicates["cuts"]:
        support = cut["support"]
        bits = slack_bits_of(cut)
        if not support:
            value += cut_weight * 1.0
            continue
        active = sum(1 for m in support if choice[m] in cut["allowed"][m])
        target = len(support) - 1
        best = None
        for s in range(1 << bits):
            residual = active + s - target
            candidate = residual * residual
            if best is None or candidate < best:
                best = candidate
        value += cut_weight * best
    return float(value)


def qubo_choice_feasible(predicates, choice, one_hot_weight=ONE_HOT_WEIGHT,
                         cut_weight=CUT_WEIGHT, tol=ZERO_TOLERANCE) -> bool:
    return min_energy_over_aux_qubo(predicates, choice, one_hot_weight,
                                    cut_weight) <= tol


# ---------------------------------------------------------------------------
# Encoding 3: exact quadratization with AND auxiliary variables
# ---------------------------------------------------------------------------

def _and_gadget_penalty(p, q, y, weight=GADGET_WEIGHT) -> float:
    """Exact binary AND penalty: >= 0 everywhere, == 0 exactly on y = p AND q."""
    return weight * (p * q - 2.0 * p * y - 2.0 * q * y + 3.0 * y)


def build_and_quadratization(predicates, one_hot_weight=ONE_HOT_WEIGHT,
                             gadget_weight=GADGET_WEIGHT,
                             product_weight=PRODUCT_WEIGHT, max_vars=4096):
    """Chain-of-AND quadratization of the k-ary witness products.

    For a witness with support (m1..mk), P_W = prod z_{mi} is built as
    u_1 = z_{m1} (a linear form, no extra variable) and u_i = AND(u_{i-1}, z_{mi})
    for i = 2..k, using the exact penalty above plus product_weight * u_k.  The
    penalty is nonnegative and vanishes exactly on the correct conjunction, so
    minimising over the auxiliaries returns zero exactly when the witness is
    inactive: the eliminant is the original constraint, not a relaxation.
    """
    sizes = predicates["sizes"]
    base = int(sum(sizes))
    plan, aux_total = [], 0
    for cut in predicates["cuts"]:
        chain = max(0, len(cut["support"]) - 1)
        plan.append({"support": cut["support"], "chain_vars": chain,
                     "first_aux": base + aux_total, "kind": cut["kind"]})
        aux_total += chain
    total = base + aux_total
    if total > max_vars:
        raise ValueError(f"AND quadratization needs {total} variables, cap is {max_vars}")
    form = SparseQuadratic(total, label="and")
    for m, size in enumerate(sizes):
        form.add_square({_flat(sizes, m, a): 1.0 for a in range(size)}, -1.0,
                        weight=one_hot_weight)
    for cut, entry in zip(predicates["cuts"], plan):
        z_terms = [{_flat(sizes, m, a): 1.0 for a in cut["allowed"][m]}
                   for m in cut["support"]]
        if not z_terms:
            form.constant += product_weight
            continue
        if len(z_terms) == 1:
            for index, coeff in z_terms[0].items():
                form.add_linear(index, product_weight * coeff)
            continue
        prev = z_terms[0]
        last_aux = None
        for step in range(1, len(z_terms)):
            cur = z_terms[step]
            y_index = entry["first_aux"] + step - 1
            for i, ci in prev.items():
                for j, cj in cur.items():
                    form.add_quadratic(i, j, gadget_weight * ci * cj)
            for i, ci in prev.items():
                form.add_quadratic(i, y_index, -2.0 * gadget_weight * ci)
            for j, cj in cur.items():
                form.add_quadratic(j, y_index, -2.0 * gadget_weight * cj)
            form.add_linear(y_index, 3.0 * gadget_weight)
            prev = {y_index: 1.0}
            last_aux = y_index
        form.add_linear(last_aux, product_weight)
    meta = {"base_vars": base, "aux_vars": aux_total, "total_vars": total,
            "plan": plan, "gadget_weight": gadget_weight,
            "product_weight": product_weight, "one_hot_weight": one_hot_weight}
    return form, meta


def min_energy_over_aux_and(predicates, choice, one_hot_weight=ONE_HOT_WEIGHT,
                            gadget_weight=GADGET_WEIGHT,
                            product_weight=PRODUCT_WEIGHT) -> float:
    """Exact minimum over the AND auxiliaries for one combination (chain DP)."""
    sizes = predicates["sizes"]
    value = 0.0
    for m, size in enumerate(sizes):
        active = 1 if 0 <= choice[m] < size else 0
        value += one_hot_weight * (active - 1) ** 2
    for cut in predicates["cuts"]:
        support = cut["support"]
        z = [1 if choice[m] in cut["allowed"][m] else 0 for m in support]
        if not z:
            value += product_weight
            continue
        if len(z) == 1:
            value += product_weight * z[0]
            continue
        dp = {z[0]: 0.0}
        for step in range(1, len(z)):
            nxt = {}
            for u_prev, cost in dp.items():
                for u_cur in (0, 1):
                    total = cost + _and_gadget_penalty(u_prev, z[step], u_cur,
                                                       gadget_weight)
                    if u_cur not in nxt or total < nxt[u_cur]:
                        nxt[u_cur] = total
            dp = nxt
        value += min(cost + product_weight * state for state, cost in dp.items())
    return float(value)


def brute_force_min_energy(form, n_base, bits_base, cap=20) -> float:
    """Brute-force minimum over every auxiliary assignment (small instances)."""
    n_aux = form.n_vars - n_base
    if n_aux > cap:
        raise ValueError(f"auxiliary enumeration is capped at {cap} bits")
    best = math.inf
    for mask in range(1 << n_aux):
        bits = list(bits_base) + [(mask >> b) & 1 for b in range(n_aux)]
        value = form.energy(bits)
        if value < best:
            best = value
    return float(best)


def choice_bits(sizes, choice):
    bits = [0] * int(sum(sizes))
    for m, label in enumerate(choice):
        bits[_flat(sizes, m, label)] = 1
    return bits


# ---------------------------------------------------------------------------
# Encoding 1: linear-constraint master problem
# ---------------------------------------------------------------------------

def linear_master_feasible(predicates, time_limit=60.0):
    """Exact feasibility of the one-hot + linear-cut master (scipy HiGHS).

    Returns ``("feasible" | "limit_with_incumbent" | "infeasible" | "unknown",
    choice or None)``; only ``infeasible`` is a proof.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix, vstack
    sizes = predicates["sizes"]
    n = int(sum(sizes))
    rows, lower, upper = [], [], []
    for m, size in enumerate(sizes):
        row = lil_matrix((1, n))
        for a in range(size):
            row[0, _flat(sizes, m, a)] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    for cut in predicates["cuts"]:
        row = lil_matrix((1, n))
        for m in cut["support"]:
            for a in cut["allowed"][m]:
                row[0, _flat(sizes, m, a)] = 1.0
        rows.append(row)
        lower.append(-np.inf)
        upper.append(len(cut["support"]) - 1.0)
    result = milp(np.zeros(n),
                  constraints=LinearConstraint(vstack(rows).tocsr(), lower, upper),
                  integrality=np.ones(n), bounds=Bounds(0, 1),
                  options={"time_limit": time_limit})
    if result.status == 2:
        return "infeasible", None
    if result.status not in (0, 1) or result.x is None:
        return "unknown", None
    values = np.rint(np.asarray(result.x)).astype(int)
    choice = []
    for m, size in enumerate(sizes):
        active = [a for a in range(size) if values[_flat(sizes, m, a)]]
        if len(active) != 1:
            return "unknown", None
        choice.append(active[0])
    if not ref_feasible(predicates, choice):
        return "unknown", None
    return ("optimal" if result.status == 0 else "limit_with_incumbent"), tuple(choice)


# ---------------------------------------------------------------------------
# Encoding 4: native conditional phase
# ---------------------------------------------------------------------------

def candidate_pool(pool):
    return circuits.CandidatePool(
        tuple(tuple(tuple(int(v) for v in order) for order in machine)
              for machine in pool))


def build_native_phase(pool, predicate_witnesses, target_t, phase_angle=math.pi,
                       penalty=1.0):
    """Native conditional phase circuit for the frozen predicate set.

    Reuses the production builder: membership flags are XORed one-hot bits
    (exact on the legal subspace), the predicate is the multi-controlled AND of
    the flags, the phase is applied and the work register is uncomputed inside
    the same circuit, so a legal input returns to the same basis state.
    """
    pool_obj = candidate_pool(pool)
    specs = [circuits.derive_witness(pool_obj, w) for w in predicate_witnesses]
    circuit = circuits.build_phase_circuit(pool_obj, specs, phase_angle=phase_angle,
                                           fixed_t=int(target_t), penalty=penalty)
    return pool_obj, circuit


def native_phase_diagonal(pool_obj, circuit, choices, max_qubits=18):
    """Ideal-statevector check of the phase diagonal and the work uncompute.

    For every requested legal one-hot input the circuit must return exactly the
    same computational basis state (work register back to |0>) with amplitude
    equal to the reference phase; the reported error is the largest absolute
    deviation, so a 1e-10 claim is measured rather than asserted.
    """
    from qiskit.quantum_info import Statevector
    n = circuit.num_qubits
    if n > max_qubits:
        return {"checked": 0,
                "skipped": f"{n} qubits exceeds the {max_qubits}-qubit simulation budget"}
    dimension = 1 << n
    worst, dirty, rows = 0.0, 0, []
    for choice, expected_phase in choices:
        index = 0
        for m, label in enumerate(choice):
            index |= 1 << pool_obj.offset(m, label)
        state = Statevector.from_int(index, dimension).evolve(circuit)
        amplitude = complex(state.data[index])
        error = abs(amplitude - expected_phase)
        worst = max(worst, error)
        residual = 1.0 - abs(amplitude) ** 2
        dirty += int(residual > 1e-12)
        rows.append({"choice": list(choice),
                     "amplitude": [amplitude.real, amplitude.imag],
                     "expected": [expected_phase.real, expected_phase.imag],
                     "abs_error": error, "residual_probability": residual})
    return {"checked": len(choices), "max_abs_error": worst,
            "dirty_work_states": dirty, "rows": rows, "qubits": n}


def transpiled_resources(circuit, max_qubits=22, optimization_level=1):
    """CX count and depth after transpilation to a hardware-agnostic basis."""
    from qiskit import transpile
    if circuit.num_qubits > max_qubits:
        return {"skipped": f"{circuit.num_qubits} qubits exceeds the transpile budget"}
    try:
        compiled = transpile(circuit, basis_gates=["cx", "u", "rz", "sx", "x"],
                             optimization_level=optimization_level, seed_transpiler=7)
    except Exception as exc:  # pragma: no cover - reported honestly if it fails
        return {"skipped": f"transpile failed: {type(exc).__name__}: {exc}"}
    counts = compiled.count_ops()
    return {"cx": int(counts.get("cx", 0)), "depth": int(compiled.depth()),
            "size": int(compiled.size()),
            "count_ops": {str(k): int(v) for k, v in counts.items()}}


# ---------------------------------------------------------------------------
# Exact CP-SAT minimisation (sizes beyond enumeration)
# ---------------------------------------------------------------------------

def cpsat_min_energy(predicates, encoding="and", time_limit=60.0, num_workers=8,
                     max_vars=600, max_cuts=None):
    """Exact minimum of one encoding's energy with CP-SAT.

    Returns the solver status verbatim; ``UNKNOWN`` is never read as a proof of
    anything.  Used where auxiliary enumeration is out of budget.
    """
    from ortools.sat.python import cp_model
    sizes = predicates["sizes"]
    cuts = predicates["cuts"] if max_cuts is None else predicates["cuts"][:max_cuts]
    base = int(sum(sizes))
    if encoding == "qubo":
        aux_per_cut = [slack_bits_of(c) for c in cuts]
    elif encoding == "and":
        aux_per_cut = [max(0, len(c["support"]) - 1) for c in cuts]
    elif encoding == "linear":
        aux_per_cut = [0] * len(cuts)
    else:
        raise ValueError(f"unknown encoding {encoding!r}")
    total = base + sum(aux_per_cut)
    if total > max_vars:
        return {"status": "skipped", "objective": None,
                "reason": f"{total} variables exceeds cap {max_vars}"}
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(base)]
    aux = [model.NewBoolVar(f"a{i}") for i in range(total - base)]
    objective = []
    for m, size in enumerate(sizes):
        terms = sum(x[_flat(sizes, m, a)] for a in range(size))
        one_hot = model.NewIntVar(-1, max(1, size - 1), f"oh{m}")
        model.Add(terms - 1 == one_hot)
        square = model.NewIntVar(0, max(1, (size - 1) ** 2), f"oh2_{m}")
        model.AddMultiplicationEquality(square, [one_hot, one_hot])
        objective.append(square)
    offset = 0
    for cut, aux_count in zip(cuts, aux_per_cut):
        support = cut["support"]
        z = []
        for m in support:
            labels = sorted(cut["allowed"][m])
            if len(labels) == sizes[m]:
                z.append(1)
                continue
            if len(labels) == 1:
                z.append(x[_flat(sizes, m, labels[0])])
                continue
            indicator = model.NewIntVar(0, 1, f"z{offset}_{m}")
            model.Add(indicator == sum(x[_flat(sizes, m, a)] for a in labels))
            z.append(indicator)
        if encoding == "linear":
            if z:
                model.Add(sum(z) <= len(support) - 1)
            continue
        if encoding == "qubo":
            field = list(z) + [aux[offset + b] * (1 << b) for b in range(aux_count)]
            if not field:
                objective.append(1)
                offset += aux_count
                continue
            low = -(len(support) - 1) - (1 << aux_count)
            high = (1 << aux_count) + len(support)
            residual = model.NewIntVar(low, high, f"res{offset}")
            model.Add(sum(field) - (len(support) - 1) == residual)
            square = model.NewIntVar(0, max(abs(low), abs(high)) ** 2, f"res2_{offset}")
            model.AddMultiplicationEquality(square, [residual, residual])
            objective.append(square)
            offset += aux_count
            continue
        # encoding == "and": exact chain, products only between variables
        if not z:
            objective.append(1)
            continue
        if len(z) == 1:
            objective.append(z[0] if not isinstance(z[0], int) else z[0])
            continue
        prev = z[0]
        for step in range(1, len(z)):
            y = aux[offset + step - 1]
            prod_pq = model.NewIntVar(0, 1, f"pq{offset}_{step}")
            if isinstance(prev, int):
                model.Add(prod_pq == prev * z[step])
            else:
                model.AddMultiplicationEquality(prod_pq, [prev, z[step]])
            prod_py = model.NewIntVar(0, 1, f"py{offset}_{step}")
            if isinstance(prev, int):
                model.Add(prod_py == prev * y)
            else:
                model.AddMultiplicationEquality(prod_py, [prev, y])
            prod_qy = model.NewIntVar(0, 1, f"qy{offset}_{step}")
            if isinstance(z[step], int):
                model.Add(prod_qy == z[step] * y)
            else:
                model.AddMultiplicationEquality(prod_qy, [z[step], y])
            penalty = model.NewIntVar(0, 4, f"pen{offset}_{step}")
            model.Add(penalty == prod_pq + (-2) * prod_py + (-2) * prod_qy + 3 * y)
            objective.append(penalty)
            prev = y
        objective.append(aux[offset + len(z) - 2])
        offset += len(z) - 1
    model.Minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(num_workers)
    started = time.perf_counter()
    status = solver.Solve(model)
    name = solver.StatusName(status)
    value = float(solver.ObjectiveValue()) if name in ("OPTIMAL", "FEASIBLE") else None
    return {"status": name, "objective": value, "variables": total,
            "cuts_encoded": len(cuts), "wall_seconds": time.perf_counter() - started}


# ---------------------------------------------------------------------------
# Penalty-weight sufficiency once an objective term is added
# ---------------------------------------------------------------------------

def penalty_weight_sweep(predicates, makespans, encoding="and",
                         scale_grid=None, tol=1e-9):
    """Where does an added linear objective break the feasibility-only ground state?

    With ``E_lambda(x) = penalty(x) + lambda * O(x)`` the ground state stays
    feasible while ``lambda * O* < min_{infeasible} penalty``; the sweep reports
    the measured flip point next to that closed-form bound, so the weight a new
    objective needs is re-derived here instead of inherited from the
    pure-feasibility encoding.

    A graph-infeasible combination that satisfies every known predicate has
    penalty zero, so no finite weight can exclude it; those combinations are
    counted separately (``infeasible_zero_penalty``) and the closed-form bound is
    only meaningful when the count is zero.
    """
    sizes = predicates["sizes"]
    grid = scale_grid or [1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
    minima = []
    for choice, makespan in zip(all_choices(sizes), makespans):
        if encoding == "qubo":
            penalty = min_energy_over_aux_qubo(predicates, choice)
        else:
            penalty = min_energy_over_aux_and(predicates, choice)
        minima.append((choice, makespan, penalty))
    infeasible = [(c, p) for c, m, p in minima if m is None]
    infeasible_penalties = [p for _, p in infeasible]
    infeasible_zero_penalty = sum(1 for p in infeasible_penalties if p <= tol)
    p_min = min(infeasible_penalties) if infeasible_penalties else None
    feasible_objectives = [m for _, m, _ in minima if m is not None]
    o_best = min(feasible_objectives) if feasible_objectives else None
    strict_bound = (p_min / o_best) if (p_min and o_best) else None
    rows = []
    for scale in grid:
        best_key, best_row = None, None
        for choice, makespan, penalty in minima:
            total = penalty + (scale * makespan if makespan is not None else 0.0)
            key = (round(total, 12), 0 if makespan is not None else 1)
            if best_key is None or key < best_key:
                best_key = key
                best_row = {"choice": list(choice), "makespan": makespan,
                            "penalty": penalty, "total": total}
        rows.append({"scale": scale, "argmin": best_row,
                     "feasible_argmin": best_row["makespan"] is not None})
    flips = [row["scale"] for row in rows if not row["feasible_argmin"]]
    return {"encoding": encoding, "p_min_infeasible": p_min, "o_best": o_best,
            "n_infeasible_graph": len(infeasible),
            "infeasible_zero_penalty": infeasible_zero_penalty,
            "closed_form_bound": strict_bound,
            "first_infeasible_scale": min(flips) if flips else None, "rows": rows}


# ---------------------------------------------------------------------------
# Per-window battery
# ---------------------------------------------------------------------------

def encoding_resources(predicates, pool, target_t, *, phase_states=None,
                       run_phase=True, max_qubits=18, transpile_max_qubits=22):
    """Resource table for the four encodings of one frozen predicate set."""
    qubo_form, qubo_meta = build_squared_residual_qubo(predicates)
    and_form, and_meta = build_and_quadratization(predicates)
    table = {
        "linear": {"data_vars": predicates["data_bits"], "aux_vars": 0,
                   "constraints": len(predicates["sizes"]) + len(predicates["cuts"]),
                   "nnz_quadratic": 0, "total_vars": predicates["data_bits"]},
        "qubo": {"data_vars": predicates["data_bits"], "metrics": qubo_form.metrics(),
                 **{k: v for k, v in qubo_meta.items() if k != "slack_bits"},
                 "slack_bits": qubo_meta["slack_bits"]},
        "and": {"data_vars": predicates["data_bits"], "metrics": and_form.metrics(),
                "aux_vars": and_meta["aux_vars"], "total_vars": and_meta["total_vars"],
                "chain_lengths": [entry["chain_vars"] for entry in and_meta["plan"]]},
    }
    if run_phase:
        pool_obj, circuit = build_native_phase(pool, predicates["witnesses"], target_t)
        resources = circuits.circuit_resources(circuit)
        table["phase"] = {
            "data_qubits": pool_obj.data_qubits,
            "aux_qubits": int(resources["metadata"].get("work_qubits", 0)),
            "qubits": int(resources["num_qubits"]),
            "logical_size": int(resources["size"]),
            "logical_depth": int(resources["depth"]),
            "count_ops": resources["count_ops"],
            "work_register_reused": bool(resources["metadata"].get("work_register_reused")),
            "witnesses_applied": int(resources["metadata"].get("witnesses_applied", 0)),
            "witnesses_skipped_constant_false": int(
                resources["metadata"].get("witnesses_skipped_constant_false", 0)),
        }
        table["phase"]["transpiled"] = transpiled_resources(
            circuit, max_qubits=transpile_max_qubits)
        if phase_states is not None:
            table["phase"]["ideal_check"] = native_phase_diagonal(
                pool_obj, circuit, phase_states, max_qubits=max_qubits)
    return table, qubo_form, and_form


def verify_zero_set(predicates, *, one_hot_weight=ONE_HOT_WEIGHT,
                    cut_weight=CUT_WEIGHT, gadget_weight=GADGET_WEIGHT,
                    product_weight=PRODUCT_WEIGHT, reference=None,
                    aux_validation_choices=2, aux_cap=16, tol=ZERO_TOLERANCE):
    """Per-combination equivalence between the reference predicates and encodings.

    Every combination is checked with the exact minimum over auxiliaries; the
    first few combinations additionally brute-force *every* auxiliary assignment
    so the closed-form and DP minimisations are validated, not assumed.
    """
    sizes = predicates["sizes"]
    q_form, _ = build_squared_residual_qubo(predicates, one_hot_weight, cut_weight)
    a_form, _ = build_and_quadratization(predicates, one_hot_weight, gadget_weight,
                                         product_weight)
    q_aux = q_form.n_vars - predicates["data_bits"]
    a_aux = a_form.n_vars - predicates["data_bits"]
    checks = {"qubo_zero_mismatch": 0, "and_zero_mismatch": 0,
              "aux_minimisation_mismatch": 0}
    aux_enum_checked = 0
    for position, choice in enumerate(all_choices(sizes)):
        feasible = (reference(predicates, choice) if reference is not None
                    else ref_feasible(predicates, choice))
        q_min = min_energy_over_aux_qubo(predicates, choice, one_hot_weight, cut_weight)
        a_min = min_energy_over_aux_and(predicates, choice, one_hot_weight,
                                        gadget_weight, product_weight)
        if (q_min <= tol) != feasible:
            checks["qubo_zero_mismatch"] += 1
        if (a_min <= tol) != feasible:
            checks["and_zero_mismatch"] += 1
        if position < aux_validation_choices:
            bits = choice_bits(sizes, choice)
            if q_aux <= aux_cap:
                if abs(brute_force_min_energy(q_form, predicates["data_bits"], bits,
                                              cap=aux_cap) - q_min) > 1e-9:
                    checks["aux_minimisation_mismatch"] += 1
                aux_enum_checked += 1
            if a_aux <= aux_cap:
                if abs(brute_force_min_energy(a_form, predicates["data_bits"], bits,
                                              cap=aux_cap) - a_min) > 1e-9:
                    checks["aux_minimisation_mismatch"] += 1
                aux_enum_checked += 1
    checks["qubo_aux_vars"] = q_aux
    checks["and_aux_vars"] = a_aux
    return checks, aux_enum_checked


def window_battery(inst, pool, witnesses, u0, target_t, *, run_phase=True,
                   max_qubits=18, phase_state_cap=32, transpile_max_qubits=22):
    """Everything T05 asks for on one frozen window."""
    predicates = compile_predicates([len(p) for p in pool], witnesses, target_t)
    predicates["witnesses"] = [w for w in witnesses
                               if w["kind"] == "cycle" or int(w["length"]) > target_t]
    sizes = predicates["sizes"]
    checks, aux_enum = verify_zero_set(predicates)

    # dense cross-check against the production encoder on the same cut set
    q_form, _ = build_squared_residual_qubo(predicates)
    prod_cuts = [{"support": tuple(c["support"]),
                  "allowed": {m: tuple(c["allowed"][m]) for m in c["support"]}}
                 for c in predicates["cuts"]]
    try:
        q_prod, meta_prod = mq.build_qubo(pool, prod_cuts)
        same_shape = q_prod.shape == q_form.dense().shape
        max_delta = float(np.max(np.abs(q_form.dense() - q_prod))) if same_shape else None
        constant_delta = abs(float(meta_prod["constant"]) - q_form.constant)
    except ValueError as exc:
        same_shape, max_delta, constant_delta = None, None, None
        checks["production_encoder_error"] = str(exc)
    checks["production_matrix_max_delta"] = max_delta
    checks["production_constant_delta"] = constant_delta
    checks["production_encoder_same_dense"] = bool(
        same_shape and max_delta is not None and max_delta < 1e-12
        and constant_delta < 1e-12)

    # independent truth for the reference predicate layer: a triggered cycle
    # witness must not be reported feasible by the graph checker, and the count
    # of graph-infeasible combinations that satisfy every known cut measures how
    # incomplete the frozen witness set is (that gap belongs to T01/T04, not to
    # the encodings: the four encodings are equivalent to the predicate set).
    cycle_unsound = 0
    graph_infeasible = 0
    graph_infeasible_satisfying = 0
    for choice in all_choices(sizes):
        feasible_graph = check_choice(inst, pool, choice)["feasible"]
        if feasible_graph and any(
                cut_triggered(c, choice) for c in predicates["cuts"]
                if c["kind"] == "cycle"):
            cycle_unsound += 1
        if not feasible_graph:
            graph_infeasible += 1
            if ref_feasible(predicates, choice):
                graph_infeasible_satisfying += 1
    checks["cycle_predicate_unsound"] = cycle_unsound
    checks["graph_infeasible_combinations"] = graph_infeasible
    checks["graph_infeasible_but_predicates_satisfied"] = graph_infeasible_satisfying

    status, _choice = linear_master_feasible(predicates, time_limit=60.0)
    any_feasible = any(ref_feasible(predicates, c) for c in all_choices(sizes))
    if status == "infeasible":
        checks["linear_master_vs_enumeration"] = int(any_feasible)
    elif status in ("optimal", "limit_with_incumbent"):
        checks["linear_master_vs_enumeration"] = int(not any_feasible)
    else:
        checks["linear_master_vs_enumeration"] = None
    checks["linear_master_status"] = status

    phase_states = None
    if run_phase:
        legal = all_choices(sizes)
        if len(legal) > phase_state_cap:
            legal = legal[:phase_state_cap]
        phase_states = [(choice, (-1.0) ** ref_violations(predicates, choice))
                        for choice in legal]
    table, _, _ = encoding_resources(predicates, pool, target_t,
                                     phase_states=phase_states, run_phase=run_phase,
                                     max_qubits=max_qubits,
                                     transpile_max_qubits=transpile_max_qubits)
    ideal = table.get("phase", {}).get("ideal_check")
    if ideal is not None and ideal.get("checked"):
        checks["native_phase_max_error"] = ideal["max_abs_error"]
        checks["native_phase_dirty_work_states"] = ideal["dirty_work_states"]
        checks["native_phase_error_over_tolerance"] = int(
            ideal["max_abs_error"] > PHASE_TOLERANCE)
    return predicates, checks, aux_enum, table


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_d0(seeds, pool_k, *, run_phase=True, max_qubits=18, transpile_max_qubits=22,
           phase_state_cap=32, sweep_every=5, seed_offset=1000):
    import data_identity
    from window_diagnostics import build_window
    results = []
    for jobs, machines in data_identity.D0_SIZES:
        for seed in seeds:
            inst = data_identity.build_d0(jobs, machines, seed)
            pool, incumbent, u0 = build_window(inst, pool_k, seed + seed_offset)
            witnesses, wstats = collect_witnesses(inst, pool, incumbent,
                                                  n_probe=16, seed=seed)
            target_t = int(u0) - 1
            predicates, checks, aux_enum, table = window_battery(
                inst, pool, witnesses, u0, target_t, run_phase=run_phase,
                max_qubits=max_qubits, phase_state_cap=phase_state_cap,
                transpile_max_qubits=transpile_max_qubits)
            makespans = [check_choice(inst, pool, c)["makespan"]
                         for c in all_choices(predicates["sizes"])]
            sweep = (penalty_weight_sweep(predicates, makespans, encoding="and")
                     if seed % sweep_every == 0 else None)
            results.append({
                "scale": f"{jobs}x{machines}", "seed": seed,
                "instance_sha256": data_identity.instance_sha256(inst),
                "c_star": data_identity.enumerate_exact_optimum(inst)["exact_optimum"],
                "u0": int(u0), "target_t": target_t,
                "pool_sizes": predicates["sizes"],
                "witness_stats": wstats,
                "predicates": {k: v for k, v in predicates.items() if k != "witnesses"},
                "checks": checks, "aux_enumeration_checked": aux_enum,
                "resources": table, "penalty_sweep": sweep,
            })
            print(f"  D0 {jobs}x{machines} seed {seed}: cuts={len(predicates['cuts'])} "
                  f"qubo_aux={checks['qubo_aux_vars']} and_aux={checks['and_aux_vars']} "
                  f"mismatch={checks['qubo_zero_mismatch']}/{checks['and_zero_mismatch']} "
                  f"prod_delta={checks['production_matrix_max_delta']} "
                  f"phase_err={checks.get('native_phase_max_error')}", flush=True)
    return results


def load_d1_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    inst = mq.Instance(report["instance_data"]["durations"],
                       report["instance_data"]["routes_zero_based"],
                       name=str(report.get("instance", Path(path).stem)))
    pool = tuple(tuple(tuple(int(v) for v in order) for order in machine)
                 for machine in report["pool_data"])
    mq.validate_pool(inst, pool)
    witnesses = []
    for w in report["witness_data"]:
        proj = mq.witness_support(
            mq.Witness(w["kind"], tuple(tuple(r) for r in w["relations"]),
                       int(w["length"]), tuple(w.get("nodes", ()))), pool)
        if proj is None:
            continue
        support, allowed = proj
        witnesses.append({"kind": w["kind"],
                          "relations": [tuple(int(x) for x in r) for r in w["relations"]],
                          "length": int(w["length"]),
                          "support": [int(m) for m in support],
                          "allowed": {int(m): [int(a) for a in allowed[m]] for m in allowed}})
    return inst, pool, witnesses, report


def run_d1(report_paths, *, samples=2000, cpsat_time=60.0, cpsat_max_cuts=12, seed=7):
    """D1 pools: randomised equivalence plus exact CP-SAT cross-checks."""
    rng = np.random.default_rng(seed)
    rows = []
    for path in report_paths:
        inst, pool, witnesses, report = load_d1_report(path)
        u0 = int(report["best_makespan"])
        incumbent = tuple(int(a) for a in report["best_choice"])
        predicates = compile_predicates([len(p) for p in pool], witnesses, u0 - 1)
        predicates["witnesses"] = [w for w in witnesses
                                   if w["kind"] == "cycle" or int(w["length"]) > u0 - 1]
        sizes = predicates["sizes"]
        checks = {"qubo_sample_mismatch": 0, "and_sample_mismatch": 0}
        sample = [incumbent]
        for _ in range(samples):
            sample.append(tuple(int(rng.integers(k)) for k in sizes))
        for choice in sample:
            feasible = ref_feasible(predicates, choice)
            if (min_energy_over_aux_qubo(predicates, choice) <= ZERO_TOLERANCE) != feasible:
                checks["qubo_sample_mismatch"] += 1
            if (min_energy_over_aux_and(predicates, choice) <= ZERO_TOLERANCE) != feasible:
                checks["and_sample_mismatch"] += 1
        status, _ = linear_master_feasible(predicates, time_limit=min(60.0, cpsat_time))
        qubo_sat = cpsat_min_energy(predicates, "qubo", time_limit=cpsat_time,
                                    max_cuts=cpsat_max_cuts, max_vars=800)
        and_sat = cpsat_min_energy(predicates, "and", time_limit=cpsat_time,
                                   max_cuts=cpsat_max_cuts, max_vars=800)
        zero_agreement = None
        if (qubo_sat.get("objective") is not None and and_sat.get("objective") is not None
                and status in ("optimal", "limit_with_incumbent", "infeasible")):
            zero_agreement = ((qubo_sat["objective"] <= 1e-9)
                              == (and_sat["objective"] <= 1e-9))
            if status == "infeasible":
                zero_agreement = zero_agreement and qubo_sat["objective"] > 1e-9
        checks["cpsat_zero_agreement"] = zero_agreement
        qubo_form, qubo_meta = build_squared_residual_qubo(predicates, max_vars=100000)
        and_form, and_meta = build_and_quadratization(predicates, max_vars=100000)
        rows.append({
            "source": Path(path).name, "instance": report.get("instance"), "u0": u0,
            "target_t": u0 - 1, "pool_sizes": sizes,
            "predicates_in_set": len(predicates["cuts"]),
            "constant_true_witnesses": predicates["constant_true_witnesses"],
            "sampled": len(sample), "checks": checks,
            "linear_master": {"status": status, "cuts": len(predicates["cuts"])},
            "cpsat": {"qubo": qubo_sat, "and": and_sat,
                      "cuts_used": min(cpsat_max_cuts, len(predicates["cuts"]))},
            "resources": {
                "linear": {"data_vars": predicates["data_bits"],
                           "constraints": len(sizes) + len(predicates["cuts"])},
                "qubo": {"data_vars": predicates["data_bits"],
                         "aux_vars": qubo_meta["aux_vars"],
                         "total_vars": qubo_meta["total_vars"],
                         "nnz_quadratic": qubo_form.metrics()["nnz_quadratic"]},
                "and": {"data_vars": predicates["data_bits"],
                        "aux_vars": and_meta["aux_vars"],
                        "total_vars": and_meta["total_vars"],
                        "nnz_quadratic": and_form.metrics()["nnz_quadratic"]},
            },
        })
        print(f"  D1 {Path(path).name}: cuts={len(predicates['cuts'])} "
              f"mismatch={checks['qubo_sample_mismatch']}/{checks['and_sample_mismatch']} "
              f"linear={status} cpsat_q={qubo_sat['status']}:{qubo_sat['objective']} "
              f"cpsat_a={and_sat['status']}:{and_sat['objective']}", flush=True)
    return rows


def clean_checks(checks) -> bool:
    """A window is clean when no counter contradicts the equivalence claims."""
    if (checks["qubo_zero_mismatch"] or checks["and_zero_mismatch"]
            or checks["aux_minimisation_mismatch"]
            or checks["cycle_predicate_unsound"]
            or checks.get("native_phase_error_over_tolerance")
            or checks.get("native_phase_dirty_work_states")
            or checks.get("linear_master_vs_enumeration")
            or not checks.get("production_encoder_same_dense")):
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="T05 encoding equivalence battery")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--pool-k", type=int, default=3)
    ap.add_argument("--d1", action="store_true")
    ap.add_argument("--d1-samples", type=int, default=2000)
    ap.add_argument("--cpsat-time", type=float, default=60.0)
    ap.add_argument("--cpsat-max-cuts", type=int, default=12)
    ap.add_argument("--no-phase", action="store_true")
    ap.add_argument("--skip-d0", action="store_true",
                    help="reuse the d0 block of an existing --out file and only add d1")
    ap.add_argument("--out", default=str(GATES_DIR / "results_encoding_equivalence_20261003"
                                         / "encoding_equivalence.json"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.skip_d0:
        if not out.exists():
            raise SystemExit(f"--skip-d0 needs an existing result file at {out}")
        result = json.loads(out.read_text(encoding="utf-8"))
    else:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
        result = {
            "schema_version": SCHEMA_VERSION,
            "work_package": "T05 (Issue #19, PR #18 docs TODO)",
            "weights": {"one_hot": ONE_HOT_WEIGHT, "cut": CUT_WEIGHT,
                        "gadget": GADGET_WEIGHT, "product": PRODUCT_WEIGHT},
            "phase_tolerance": PHASE_TOLERANCE,
            "d0": run_d0(seeds, args.pool_k, run_phase=not args.no_phase),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                  default=_json_default), encoding="utf-8")
        print(f"wrote d0 block to {out}", flush=True)
    if args.d1:
        result["d1"] = run_d1(
            sorted(str(p) for p in (ROOT / "code" / "candidate_quantum_medium"
                                    / "results" / "corrected_20261001").glob("*_milp.json")),
            samples=args.d1_samples, cpsat_time=args.cpsat_time,
            cpsat_max_cuts=args.cpsat_max_cuts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=_json_default),
                   encoding="utf-8")
    clean = sum(1 for row in result["d0"] if clean_checks(row["checks"]))
    print(f"wrote {out}")
    print(f"D0 windows clean: {clean}/{len(result['d0'])}")
    if "d1" in result:
        d1_clean = sum(1 for row in result["d1"]
                       if not row["checks"]["qubo_sample_mismatch"]
                       and not row["checks"]["and_sample_mismatch"])
        print(f"D1 pools without sampling mismatch: {d1_clean}/{len(result['d1'])}")
    return 0


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj).__name__)


if __name__ == "__main__":
    sys.exit(main())
