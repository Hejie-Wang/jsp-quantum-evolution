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
