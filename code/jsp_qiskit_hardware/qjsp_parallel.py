#!/usr/bin/env python3
"""Swap-network parallelization of the JSP permutation mixer (theory only, no QPU).

Idea (Kivlichan et al. PRL 120, 110501 swap-network structure):
  swaps on different machines commute (disjoint registers) -> fully parallel;
  swaps (m,k), (m,k') on one machine commute when |k-k'| >= 2 (disjoint positions).
  Edge-coloring the path graph needs only 2 colors, so the M(J-1) sequential
  transpositions per evolution layer compress into <= 2 parallel stages
  (1 stage when J = 2). Within a stage the order is irrelevant (commuting
  involutions: prod (cos b I + i sin b S_i) = exp(i b sum S_i)).

This script verifies, without any quantum hardware:
  1. numpy-level equivalence of serial vs parallel-stage evolution (2x2/3x3/4x3);
  2. the 2x2 Qiskit circuit built from stages still matches qjsp.evolve exactly;
  3. stage/depth scaling table up to 50x20;
  4. ideal Aer sampling and FakeFez noisy prediction for the 2x2 circuit,
     compared against the real ibm_fez results stored in hw_L*.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
QJSP_DIR = HERE.parent / "JSP_量子演化可运行代码" / "jsp_quantum_evolution"
sys.path.insert(0, str(QJSP_DIR))
import qjsp  # noqa: E402


def enumerate_swaps(model):
    """qjsp stores swaps as (axis, mapping) appended in (m, k) ascending order."""
    return [(i // (model.instance.jobs - 1), i % (model.instance.jobs - 1), sw)
            for i, sw in enumerate(model.swaps)]


def schedule_swaps(model):
    """Group transpositions into parallel stages by parity edge-coloring."""
    tagged = enumerate_swaps(model)
    even = [sw for _m, k, sw in tagged if k % 2 == 0]  # k even, any machine
    odd = [sw for _m, k, sw in tagged if k % 2 == 1]
    return [stage for stage in (even, odd) if stage]


def apply_layer_staged(model, state, s, h, stages):
    psi = state * np.exp((-1j * s * h) * model.potential)
    if model.d:
        beta = (1 - s) * h / model.d
        cosine, sine = math.cos(beta), 1j * math.sin(beta)
        for stage in stages:  # within a stage all swaps commute: order-free
            for swap in stage:
                psi = cosine * psi + sine * model.swap_state(psi, swap)
    return psi


def evolve_staged(model, stages, tau, layers, s_final):
    psi = model.uniform_state()
    h = tau / layers
    for ell in range(layers):
        s = s_final * (ell + 0.5) / layers
        psi = apply_layer_staged(model, psi, s, h, stages)
    return psi


def compare_serial_vs_parallel(inst_name, tau=20.0, layers=2000, s_final=0.95):
    inst = qjsp.Instance.read(str(QJSP_DIR / "data" / inst_name))
    model = qjsp.PermutationModel(inst)
    stages = schedule_swaps(model)
    out = {"instance": inst_name, "num_swaps": model.d,
           "serial_stages": model.d, "parallel_stages": len(stages)}
    for L in ([1, 2, 5, 10] if model.dimension <= 256 else []) + [layers]:
        psi_serial, _ = qjsp.evolve(model, tau=tau, layers=L, s_final=s_final)
        psi_par = evolve_staged(model, stages, tau, L, s_final)
        fid = float(abs(np.vdot(psi_serial, psi_par)) ** 2)
        tv = 0.5 * float(np.abs(np.abs(psi_serial) ** 2 - np.abs(psi_par) ** 2).sum())
        out[f"fidelity_L{L}"] = fid
        out[f"tv_L{L}"] = tv
    return out


def stage_scaling_table():
    rows = []
    for j, m in ((2, 2), (3, 3), (4, 3), (5, 5), (8, 8), (50, 20)):
        serial = m * (j - 1)
        parallel = 1 if j == 2 else 2
        rows.append({"J": j, "M": m, "serial_swap_stages": serial,
                     "parallel_swap_stages": parallel,
                     "depth_reduction": serial / parallel})
    return rows


def build_circuit_staged(model, stages, tau, layers, s_final):
    """2x2 circuit: cost phase + one RX(-2*beta) per swap, grouped by stage."""
    from qiskit import QuantumCircuit
    if model.shape != (2, 2):
        raise ValueError("circuit builder supports 2x2 only")
    potential = model.potential
    h = tau / layers
    qc = QuantumCircuit(2, name="qjsp_2x2_parallel")
    qc.h(0); qc.h(1)
    for ell in range(layers):
        s = s_final * (ell + 0.5) / layers

        def phase(r0, r1):
            return -s * h * float(potential[2 * r0 + r1])
        p00, p10, p01, p11 = phase(0, 0), phase(1, 0), phase(0, 1), phase(1, 1)
        b = (p00 - p10 + p01 - p11) / 4
        c = (p00 + p10 - p01 - p11) / 4
        e = (p00 - p10 - p01 + p11) / 4
        qc.rz(-2 * b, 0)
        qc.rz(-2 * c, 1)
        qc.rzz(-2 * e, 0, 1)
        beta = (1 - s) * h / model.d
        for stage in stages:  # 2x2: a single stage -> both RX in parallel
            for axis, _mapping in stage:
                qc.rx(-2 * beta, axis)
            qc.barrier()
    return qc


def sample_counts(qc, backend, shots, seed=7):
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    meas = qc.copy()
    meas.measure_all()
    isa = generate_preset_pass_manager(backend=backend, optimization_level=1).run(meas)
    job = AerSampler.from_backend(backend).run([isa], shots=shots)
    return job.result()[0].data.meas.get_counts(), isa


def main():
    report = {"method": "parity edge-colored swap-network parallelization", }

    # 1. numpy-level serial vs parallel
    report["numpy_checks"] = [
        compare_serial_vs_parallel(n) for n in ("demo_2x2.json", "demo_3x3.json", "demo_4x3.json")]

    # 2. stage scaling
    report["stage_scaling"] = stage_scaling_table()

    # 3./4. 2x2 circuits: ideal + FakeFez noisy prediction vs stored real results
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from qiskit.quantum_info import Statevector

    inst = qjsp.Instance.read(str(QJSP_DIR / "data" / "demo_2x2.json"))
    model = qjsp.PermutationModel(inst)
    stages = schedule_swaps(model)
    aer = AerSimulator()
    fake = FakeFez()
    runs = []
    for L in (1, 2, 5, 10):
        qc = build_circuit_staged(model, stages, 20.0, L, 0.95)
        sv = np.asarray(Statevector.from_instruction(qc).data)
        probs = np.empty(4)
        for r0 in (0, 1):
            for r1 in (0, 1):
                probs[2 * r0 + r1] = abs(sv[r0 + 2 * r1]) ** 2

        def tv(counts):
            f = np.zeros(4)
            tot = sum(counts.values())
            for key, n in counts.items():
                f[2 * int(key[-1]) + int(key[-2])] = n / tot
            return 0.5 * float(np.abs(f - probs).sum()), f

        counts_ideal, _ = sample_counts(qc, aer, 1024)
        counts_fake, isa_fake = sample_counts(qc, fake, 1024)
        tv_ideal, _ = tv(counts_ideal)
        tv_fake, freq_fake = tv(counts_fake)
        row = {"layers": L, "tv_ideal_aer": tv_ideal, "tv_fakefez": tv_fake,
               "fakefez_freq_numpy_order": [float(x) for x in freq_fake],
               "ideal_probs_numpy_order": [float(x) for x in probs],
               "fakefez_depth": isa_fake.depth()}
        hw_file = HERE / f"hw_L{L}.json"
        if hw_file.exists():
            hw = json.loads(hw_file.read_text(encoding="utf-8"))
            row["tv_real_ibm_fez"] = hw["hardware"]["tv_distance_vs_ideal"]
            row["real_job_id"] = hw["job_id"]
        runs.append(row)
        print(f"layers={L}: TV ideal-Aer={tv_ideal:.4f}  FakeFez={tv_fake:.4f}  "
              f"real={row.get('tv_real_ibm_fez')}", flush=True)
    report["runs_2x2"] = runs

    out = HERE / "parallel_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["numpy_checks"], ensure_ascii=False, indent=2))
    print(json.dumps(report["stage_scaling"], ensure_ascii=False, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
