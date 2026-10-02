# Offline candidate quantum gates

This package is the gate-level continuation of the Kimi candidate-pool design.
It is intentionally local-only: use the conda `qskit` environment, Qiskit
`quantum_info`, Aer, and `transpile` with a generic basis/coupling map. There
are no provider, account, cloud-backend, or hardware-job imports.

```powershell
conda run --no-capture-output -n qskit python -m unittest discover `
  -s code/candidate_quantum_gates -p test_circuits.py -v
```

The tests use tiny circuits only. The 15x20 runner creates a resource-only
160-data-qubit circuit and never allocates a statevector.
