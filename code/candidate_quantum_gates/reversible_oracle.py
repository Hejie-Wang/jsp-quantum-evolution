#!/usr/bin/env python3
"""T11: reversible micro-oracle for the D0 predicate, with a counted cost model.

Implements the conditional query model of the T11 work package of
docs/量子计算能否改善JSP问题的求解+子问题拆解TODO.md:

    O_T: |a>|b>|0> -> |a>|b xor 1[G(a) acyclic and C(a) <= T]>|0>

Scope of this implementation: D0 instances whose machines all carry exactly two
operations (the 2x2 family).  For those pools each machine has exactly one
binary direction bit (which of its two operations runs first), so the whole
predicate can be evaluated *structurally*:

  * the direction bit of machine m is computed from the one-hot data bits by
    XORing the candidates whose order puts the reference operation first
    (exact on the legal one-hot subspace, like the production gate layer);
  * the predicate is then a truth table over the k direction bits only
    (k = number of machines), evaluated with the independent graph checker
    ``window_diagnostics.check_choice`` - no cost table over the 2^(data bits)
    combination space is ever built, and no exact answer file is read;
  * the circuit is a compute / copy / uncompute construction (multi-controlled
    X per satisfying direction pattern into one term ancilla, one CX into the
    answer, then the mirrored uncompute), so every work qubit returns to |0>.

Everything the oracle costs is counted: data qubits, direction ancillas, term
ancillas, multi-controlled X gates, the mirrored uncompute, and the transpiled
CX count and depth.  The break-even analysis is deliberately conditional: it
reports the per-gate time below which T_Q < T_C for the measured classical
sampling cost and the window's success probability p, and it reports "no
break-even" when p = 0 (infeasible target), instead of fabricating a number.

No sampler, backend or hardware is used; the statevector checks are ideal
simulations of the built circuit.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GATES_DIR = Path(__file__).resolve().parent
MEDIUM_DIR = ROOT / "code" / "candidate_quantum_medium"
for _p in (str(GATES_DIR), str(MEDIUM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import medium_qjsp as mq  # noqa: E402
from window_diagnostics import build_window, check_choice, generate_d0  # noqa: E402

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Reversible wire circuit with an explicit mirror for uncomputation
# ---------------------------------------------------------------------------

class WireCircuit:
    """Straight-line reversible circuit: allocated wires, X/CX/MCX, mirroring.

    ``mark()`` records the current end of the gate list; ``uncompute(mark)``
    replays that block in reverse, which is what returns every work wire to its
    initial value.  Gates are stored symbolically and only materialised into a
    qiskit circuit by :meth:`to_qiskit`.
    """

    def __init__(self):
        self.wires = 0
        self.gates = []
        self.labels = {}

    def alloc(self, count=1, label=""):
        first = self.wires
        for offset in range(count):
            self.labels[first + offset] = f"{label}{offset}" if count > 1 else label
        self.wires += count
        return first if count == 1 else list(range(first, first + count))

    def mark(self):
        return len(self.gates)

    def x(self, wire):
        self.gates.append(("x", (), wire))

    def cx(self, control, target):
        self.gates.append(("cx", (control,), target))

    def mcx(self, controls, target):
        controls = tuple(int(c) for c in controls)
        if not controls:
            self.x(target)
        elif len(controls) == 1:
            self.cx(controls[0], target)
        else:
            self.gates.append(("mcx", controls, target))

    def xor_into(self, sources, target):
        for source in sources:
            self.cx(source, target)

    def controlled_pattern_xor(self, controls, pattern, target):
        """XOR the target when every control matches ``pattern`` (0/1 per bit)."""
        zeros = [c for c, bit in zip(controls, pattern) if bit == 0]
        for wire in zeros:
            self.x(wire)
        self.mcx(controls, target)
        for wire in zeros:
            self.x(wire)

    def uncompute(self, mark, end=None):
        """Replay ``gates[mark:end]`` in reverse (``end`` defaults to the tail)."""
        block = self.gates[mark:len(self.gates) if end is None else end]
        for name, controls, target in reversed(block):
            if name == "x":
                self.x(target)
            elif name == "cx":
                self.cx(controls[0], target)
            else:
                self.mcx(controls, target)

    def counts(self) -> dict:
        ops = {"x": 0, "cx": 0, "mcx": 0}
        for name, _controls, _target in self.gates:
            ops[name] += 1
        return {"wires": self.wires, "gates": len(self.gates), **ops}

    def to_qiskit(self, name="oracle"):
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(self.wires, name=name)
        for gate, controls, target in self.gates:
            if gate == "x":
                circuit.x(target)
            elif gate == "cx":
                circuit.cx(controls[0], target)
            else:
                circuit.mcx(list(controls), target)
        return circuit


# ---------------------------------------------------------------------------
# Structural predicate: direction bits -> (acyclic, makespan)
# ---------------------------------------------------------------------------

def binary_direction_machines(pool) -> list:
    """Machines with exactly two operations: one binary order bit each."""
    return [m for m, machine in enumerate(pool) if len(machine[0]) == 2]


def direction_labels(pool, machine) -> tuple:
    """(label_forward, forward_pair): the candidate whose order runs u before v."""
    first = pool[machine][0]
    pair = (first[0], first[1])
    forward = [a for a, order in enumerate(pool[machine])
               if order.index(pair[0]) < order.index(pair[1])]
    if len(forward) != 1:
        raise ValueError("one binary direction bar per machine requires exactly "
                         "one candidate with the reference order")
    return forward[0], pair


def structural_table(inst, pool, target_t):
    """Truth table of (acyclic, makespan <= target) over the direction bits.

    The table has 2^k entries (k machines with two operations), each evaluated
    with the independent graph checker on the combination the direction pattern
    selects; the 2^(data bits) combination space is never enumerated here.
    """
    machines = binary_direction_machines(pool)
    k = len(machines)
    if k == 0:
        raise ValueError("oracle needs at least one two-operation machine")
    if len(machines) != len(pool):
        raise ValueError("every machine must be binary for this oracle scope")
    entries = []
    for pattern in itertools.product((0, 1), repeat=k):
        choice = [0] * len(pool)
        for slot, machine in enumerate(machines):
            forward, _pair = direction_labels(pool, machine)
            other = [a for a in range(len(pool[machine])) if a != forward]
            if len(other) != 1:
                raise ValueError("binary machine must have exactly two candidates")
            choice[machine] = forward if pattern[slot] else other[0]
        result = check_choice(inst, pool, tuple(choice))
        entries.append({"pattern": list(pattern), "choice": choice,
                        "acyclic": bool(result["feasible"]),
                        "makespan": result["makespan"],
                        "satisfies": bool(result["feasible"]
                                          and result["makespan"] is not None
                                          and result["makespan"] <= target_t)})
    return {"k": k, "machines": machines, "entries": entries}


def classical_predicate(entry) -> int:
    return int(entry["satisfies"])


# ---------------------------------------------------------------------------
# Oracle construction
# ---------------------------------------------------------------------------

def build_oracle(inst, pool, target_t, name="O_T"):
    """Reversible oracle circuit plus the structural table it encodes.

    Layout: ``data`` = one-hot candidate bits; ``direction`` = one binary order
    flag per machine; ``term`` = one shared minterm ancilla; ``answer`` = the
    predicate bit the oracle XORs.  All of ``direction``/``term`` are uncomputed.
    """
    sizes = [len(machine) for machine in pool]
    data_bits = int(sum(sizes))
    circuit = WireCircuit()
    data = circuit.alloc(data_bits, "x")
    data = list(data) if isinstance(data, list) else [data]
    table = structural_table(inst, pool, target_t)
    machines = table["machines"]
    forward = {m: direction_labels(pool, m)[0] for m in machines}
    direction = circuit.alloc(len(machines), "d")
    direction = list(direction) if isinstance(direction, list) else [direction]
    compute_mark = circuit.mark()
    for slot, machine in enumerate(machines):
        offset = sum(sizes[:machine])
        circuit.xor_into([data[offset + forward[machine]]], direction[slot])
    term = circuit.alloc(1, "t")
    term = term if isinstance(term, int) else term[0]
    answer = circuit.alloc(1, "b")
    answer = answer if isinstance(answer, int) else answer[0]
    satisfying = [entry for entry in table["entries"] if entry["satisfies"]]
    for entry in satisfying:
        circuit.controlled_pattern_xor(direction, entry["pattern"], term)
    copy_mark = circuit.mark()
    circuit.cx(term, answer)
    circuit.uncompute(compute_mark, end=copy_mark)
    return {
        "circuit": circuit,
        "table": table,
        "layout": {"data_bits": data_bits, "direction_bits": len(machines),
                   "term_bits": 1, "answer_bits": 1,
                   "total_wires": circuit.wires,
                   "data_offset": 0,
                   "answer_wire": answer},
        "satisfying_patterns": len(satisfying),
    }


def verify_oracle(inst, pool, target_t, max_qubits=16):
    """Ideal-statevector check on every legal one-hot input.

    The answer wire must equal the classical predicate and every work wire must
    return to |0>; the reported counters are measured, not asserted.
    """
    from qiskit.quantum_info import Statevector
    built = build_oracle(inst, pool, target_t)
    circuit = built["circuit"]
    layout = built["layout"]
    sizes = [len(machine) for machine in pool]
    qc = circuit.to_qiskit()
    n = qc.num_qubits
    if n > max_qubits:
        return {"checked": 0, "skipped": f"{n} qubits exceeds the budget"}
    dimension = 1 << n
    mismatches, dirty, worst = 0, 0, 0.0
    rows = []
    for choice in itertools.product(*(range(k) for k in sizes)):
        index = 0
        for machine, label in enumerate(choice):
            index |= 1 << (sum(sizes[:machine]) + label)
        state = Statevector.from_int(index, dimension).evolve(qc)
        result = check_choice(inst, pool, choice)
        expected = int(result["feasible"] and result["makespan"] is not None
                       and result["makespan"] <= target_t)
        p1 = float(abs(state.data[index | (1 << layout["answer_wire"])]) ** 2)
        p0 = float(abs(state.data[index]) ** 2)
        got = 1 if p1 > p0 else 0
        if got != expected:
            mismatches += 1
        residual = 1.0 - p0 - p1
        dirty += int(abs(residual) > 1e-12)
        worst = max(worst, abs(p1 - expected), abs(p0 - (1 - expected)))
        rows.append({"choice": list(choice), "expected_answer": expected,
                     "got_answer": got, "p_answer_0": p0, "p_answer_1": p1,
                     "work_residual": residual, "makespan": result["makespan"]})
    return {"checked": len(rows), "answer_mismatches": mismatches,
            "dirty_work_states": dirty, "max_probability_error": worst,
            "qubits": n, "rows": rows}


def oracle_resources(inst, pool, target_t, optimization_level=1):
    """Counted resources: logical gates, transpiled CX/depth, qubit breakdown."""
    from qiskit import transpile
    built = build_oracle(inst, pool, target_t)
    circuit = built["circuit"]
    qc = circuit.to_qiskit()
    logical = circuit.counts()
    result = {
        "layout": built["layout"],
        "satisfying_patterns": built["satisfying_patterns"],
        "logical": logical,
        "logical_depth": qc.depth(),
    }
    try:
        compiled = transpile(qc, basis_gates=["cx", "u", "rz", "sx", "x"],
                             optimization_level=optimization_level, seed_transpiler=7)
        counts = compiled.count_ops()
        result["transpiled"] = {"cx": int(counts.get("cx", 0)),
                                "depth": int(compiled.depth()),
                                "size": int(compiled.size()),
                                "count_ops": {str(k): int(v) for k, v in counts.items()}}
    except Exception as exc:  # pragma: no cover - reported honestly
        result["transpiled"] = {"skipped": f"{type(exc).__name__}: {exc}"}
    return result


# ---------------------------------------------------------------------------
# Break-even analysis (conditional resource claim)
# ---------------------------------------------------------------------------

def classical_sample_cost(inst, pool, repeats=2000, seed=7):
    """Mean wall time of one classical uniform sample plus graph evaluation."""
    rng = np.random.default_rng(seed)
    sizes = [len(machine) for machine in pool]
    started = time.perf_counter()
    for _ in range(repeats):
        choice = tuple(int(rng.integers(k)) for k in sizes)
        check_choice(inst, pool, choice)
    return (time.perf_counter() - started) / repeats


def success_probability(inst, pool, target_t, preparation="legal_uniform"):
    """p = probability that the chosen preparation yields a satisfying state."""
    sizes = [len(machine) for machine in pool]
    legal = list(itertools.product(*(range(k) for k in sizes)))
    hits = 0
    for choice in legal:
        result = check_choice(inst, pool, choice)
        if result["feasible"] and result["makespan"] is not None \
                and result["makespan"] <= target_t:
            hits += 1
    if preparation == "legal_uniform":
        return hits / len(legal)
    if preparation == "bitstring_uniform":
        return hits / (1 << int(sum(sizes)))
    raise ValueError(f"unknown preparation {preparation!r}")


def breakeven_analysis(inst, pool, target_t, *, repeats=2000, seed=7,
                       gate_times_seconds=(1e-9, 1e-8, 1e-7, 1e-6)):
    """Per-gate time below which T_Q < T_C, with the classical cost measured.

    ``T_Q ~ (c_prep + c_oracle + c_uncompute)/sqrt(p) + T_fixed`` and
    ``T_C ~ c_classical/p``; with ``g`` counted gates and per-gate time ``tau``
    the break-even is ``tau* = c_classical / (g * sqrt(p))`` (``T_fixed = 0``).
    ``p = 0`` yields no break-even at all and is reported as such.
    """
    resources = oracle_resources(inst, pool, target_t)
    gates = resources["logical"]["gates"]
    classical = classical_sample_cost(inst, pool, repeats=repeats, seed=seed)
    p = success_probability(inst, pool, target_t)
    rows = []
    for tau in gate_times_seconds:
        if p <= 0:
            rows.append({"gate_seconds": tau, "quantum_seconds": None,
                         "classical_seconds": None, "quantum_wins": None,
                         "max_fixed_overhead_seconds": None})
            continue
        quantum = gates * tau / math.sqrt(p)
        classical_cost = classical / p
        rows.append({"gate_seconds": tau, "quantum_seconds": quantum,
                     "classical_seconds": classical_cost,
                     "quantum_wins": bool(quantum < classical_cost),
                     "max_fixed_overhead_seconds": max(0.0, classical_cost - quantum)})
    bound = (classical / (gates * math.sqrt(p))) if p > 0 else None
    return {"gates": gates, "classical_seconds_per_sample": classical,
            "success_probability": p, "breakeven_gate_seconds": bound,
            "rows": rows, "resources": resources}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_d0(seeds, *, repeats=2000):
    import data_identity
    rows = []
    for seed in seeds:
        inst = data_identity.build_d0(2, 2, seed)
        pool, incumbent, u0 = build_window(inst, 2, seed + 1000)
        for target_t in sorted({u0 - 1, u0}):
            if target_t < 0:
                continue
            verification = verify_oracle(inst, pool, target_t)
            analysis = breakeven_analysis(inst, pool, target_t, repeats=repeats)
            rows.append({
                "seed": seed, "target_t": int(target_t), "u0": int(u0),
                "instance_sha256": data_identity.instance_sha256(inst),
                "pool_sizes": [len(machine) for machine in pool],
                "pool_sha256": mq.content_hash(pool),
                "verification": verification,
                "breakeven": {k: v for k, v in analysis.items() if k != "resources"},
                "resources": analysis["resources"],
            })
            print(f"  2x2 seed {seed} T={target_t}: checked={verification['checked']} "
                  f"mismatch={verification['answer_mismatches']} "
                  f"dirty={verification['dirty_work_states']} "
                  f"gates={analysis['gates']} p={analysis['success_probability']:.3f} "
                  f"tau*={analysis['breakeven_gate_seconds']}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="T11 reversible micro-oracle battery")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--out", default=str(GATES_DIR / "results_reversible_oracle_20261003"
                                         / "reversible_oracle.json"))
    args = ap.parse_args(argv)
    lo, hi = args.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    result = {
        "schema_version": SCHEMA_VERSION,
        "work_package": "T11 (Issue #19, PR #18 docs TODO)",
        "scope": "D0 2x2 family: every machine carries exactly two operations, so the "
                 "predicate is a table over one binary direction bit per machine",
        "d0": run_d0(seeds, repeats=args.repeats),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=_json_default),
                   encoding="utf-8")
    clean = sum(1 for row in result["d0"]
                if row["verification"]["answer_mismatches"] == 0
                and row["verification"]["dirty_work_states"] == 0)
    print(f"wrote {out}")
    print(f"oracle windows clean: {clean}/{len(result['d0'])}")
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
