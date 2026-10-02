# Offline candidate quantum gates

This package is the gate-level continuation of the Kimi candidate-pool design.
It is intentionally local-only: use the conda `qskit` environment, Qiskit
`quantum_info`, Aer, and `transpile` with a generic basis/coupling map. There
are no provider, account, cloud-backend, or hardware-job imports.

It implements P0-P3 of `docs/candidate_quantum_improvements_20261002.md`:

* **P0 constraint/phase semantics** (`circuits.py`): witness projection is the
  per-machine intersection (`allowed_labels` / `derive_witness`, matching
  `medium_qjsp.witness_support`); cycle AND path witnesses both carry the
  positive `penalty` (`H = T + penalty * (cycles + over-T paths)`); demo
  witnesses come from real graph evaluation, never hand-written evidence;
  constant-false witnesses emit nothing, constant-true ones only a global
  phase; initial-state preparation is separate (`prepare_legal_basis`,
  `prepare_uniform_legal`).
* **P1 resources** (`circuits.py`): one shared work register for all
  witnesses (`machines + 3` flag/predicate/compare/active qubits plus
  comparator work), and joint transitions use the exact CX/X/multi-controlled
  -Rx construction with zero ancillas instead of a dense `UnitaryGate`
  (dense matrices remain as test references only). 44 witnesses on the
  15x20 derived pool now compile to 183 logical qubits; the old layout held
  only 2 witnesses in 206 qubits.
* **P2 closed loop** (`search_loop.py`): multilayer fixed-T circuit
  `legal initial state -> [witness phase -> XY -> joint] x p -> measurement`
  plus the outer search loop (propose 2-4 machine joint actions from the
  current witnesses, sample, deduplicate, run the full graph evaluation,
  accept only independently verified better schedules, add witnesses from
  failures). Layer, parameter-evaluation, shots and wall-time budgets are
  recorded. Sampling failure never proves infeasibility.
* **P3 diagnostics** (`pool_optimum_milp.py`, `search_comparison.py`): the
  exact pool-inner optimum MILP (one-hot x + continuous start times + C,
  machine edges relaxed by `U*(1-x_ma)`, minimize C) and a multi-seed
  comparison of uniform / XY / XY+joint / classical search with the exact
  MILP reference timed separately. Results live in `results_p3_20261002/`.

## Run

```powershell
conda run --no-capture-output -n qskit python -m unittest discover `
  -s code/candidate_quantum_gates -p "test_*.py" -v

conda run --no-capture-output -n qskit python code/candidate_quantum_gates/run_offline.py `
  --medium-result code/candidate_quantum_medium/results/corrected_20261001/15x20_derived_seed11_kaiwu.json

conda run --no-capture-output -n qskit python code/candidate_quantum_gates/search_loop.py --demo

conda run --no-capture-output -n qskit python code/candidate_quantum_gates/pool_optimum_milp.py `
  --report code/candidate_quantum_medium/results/corrected_20261001/15x15_seed7_milp.json `
  --report code/candidate_quantum_medium/results/corrected_20261001/20x15_seed7_milp.json `
  --report code/candidate_quantum_medium/results/corrected_20261001/15x20_derived_seed11_milp.json `
  --output code/candidate_quantum_gates/results_p3_20261002/pool_milp_evidence.json

conda run --no-capture-output -n qskit python code/candidate_quantum_gates/search_comparison.py `
  --cases demo_3x3 5x5 5x5 --seeds 7 11 13 `
  --output code/candidate_quantum_gates/results_p3_20261002/search_comparison.json
```

The tests use tiny circuits only. The 15x20 compile creates a resource-only
183-qubit circuit and never allocates a statevector. Earlier gate-level
results are kept in `results_final_20261001.json` and
`docs/candidate_quantum_gates_results_20261001.md`.

## Dynamic candidates and accelerated evaluation

`adaptive_search.py` refreshes whole-machine candidates around critical
blocks, uses tabu memory/restarts, and periodically recombines candidates on
up to six active machines. Frozen machines still participate in the full DAG
evaluation and witness projection. `classical` is the default; `uniform` and
`quantum` add controlled ablations of joint proposals. Quantum mode currently
uses an **exact classical simulator**, never a QPU job.

`compact_simulator.py` simulates the existing phase/complete-graph XY/joint
ansatz in the legal candidate basis. It omits illegal one-hot states and clean
ancillas, matches Qiskit amplitudes in regression tests, and caps the state
count at 65,536. It builds witness predicates, not a schedule-cost lookup
table. `search_loop.py` uses this backend by default; select
`simulation_backend="qiskit"` in the Python API for the gate reference.

```bash
python -m pip install -r code/candidate_quantum_gates/requirements-speed.txt
python code/candidate_quantum_gates/adaptive_search.py task_data/tai50_20_01.txt --seconds 10 --mode classical --output results/ta61.json
python code/candidate_quantum_gates/adaptive_search.py task_data/tai50_20_01.txt --seconds 10 --mode quantum --output results/ta61_quantum.json
python code/candidate_quantum_gates/benchmark_adaptive.py --seconds 10 --output results/adaptive_comparison
```

CPU evaluation uses Numba. CUDA evaluation is implemented in
`evaluate_batch.cu`, loaded by CuPy, with chunked transfers and identical cycle
semantics. On an NVIDIA machine install the CuPy wheel matching its CUDA
runtime (for example `cupy-cuda12x`), then use `--backend cuda`. Explicit CUDA
requests fail if unavailable; `--backend auto` records any CPU fallback.
The included host execution of the CUDA kernel body checks its indexing and
logic; it does **not** replace the optional on-device parity test or establish
GPU speedup. Transfers, allocation, and synchronization count toward timing.

Each result includes independently verified starts/orders, instance hash,
training/evaluation counts, proposal contributions and setup/search/end-to-end
times. `--seconds` bounds search after setup, with checks between iterations;
an in-progress batch can overrun the deadline. The benchmark charges the same
240-schedule initialization to each paired mode. Published bounds are only
reporting metadata and never passed to search. See
`benchmark_bounds.json` for identity checks and sources, and
[the measured report](../../docs/candidate_adaptive_results_20261002.md)
for results and limitations.
