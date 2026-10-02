# Medium JSP Candidate Master

This package continues Kimi's [candidate design](../../docs/quantum_candidate_design.md)
and [3x3 demo](../candidate_quantum_demo/README.md). It runs a fixed-T feasibility
master over machine-order candidate pools, with graph cycle/path separation.
The two backends are SciPy/HiGHS MILP and **classical** Kaiwu simulated annealing.
See [the approved repair scope](../../docs/superpowers/specs/2026-10-01-candidate-medium-correctness-design.md).

The medium solver does not enumerate the Cartesian product or construct a
Hamiltonian/statevector. Kaiwu uses a dense QUBO matrix with a 2048-variable
allocation cap. This iteration leaves gate compilation, quantum mixers and
QPU execution to the later tasks in Kimi's design.

## Run With Conda

Run these commands from the repository root. The existing `Kaiwu` conda
environment supplies Python 3.10.20, NumPy 2.2.6 and SciPy 1.15.3. Kaiwu's
distribution metadata reports 1.4.1; the imported module reports `__version__`
1.0.7. We checked its actual `solve` signature and record both versions in
[the environment audit](results/corrected_20261001/environment_audit.json).
The `qskit` environment supplies Python 3.11 and Qiskit 2.5.2; this experiment
does not need Qiskit gates.

```powershell
conda run --no-capture-output -n qskit python -m unittest discover -s code/candidate_quantum_demo -p test_demo.py -v
conda run --no-capture-output -n Kaiwu python -m unittest discover -s code/candidate_quantum_medium -p test_medium.py -v
conda run --no-capture-output -n Kaiwu python code/candidate_quantum_medium/benchmark_medium.py --iterations 100 --time-limit 40 --output-dir code/candidate_quantum_medium/results/corrected_20261001
```

The benchmark builds one pool per instance/seed and passes the same pool to
both backends. Defaults: K=8, 240 append-only priority-rule schedules,
seeds 7/11, at most 100 master calls, 40 seconds per cut loop. Each Kaiwu call
uses 50 iterations per temperature and retains up to 30 samples; the loop
evaluates at most four distinct legal candidate combinations per iteration.

For a single 15-job, 20-machine run:

```powershell
conda run --no-capture-output -n Kaiwu python code/candidate_quantum_medium/run_medium.py --instance task_data/ta21.txt --first-jobs 15 --master milp --candidates 8 --schedules 240 --iterations 100 --time-limit 40 --output code/candidate_quantum_medium/results/my_15x20_milp.json
```

Replace `--master milp` with `--master kaiwu --sa-iterations-per-t 50
--sa-size-limit 30` for SA. Add `--verbose` for iteration logs.

Local `ta21.txt` has **20 jobs and 20 machines**. `--first-jobs 15` keeps the
first 15 jobs and all 20 machines. The output labels this as a **derived
instance, not a published 15x20 benchmark**. Source path/hash and derivation
appear in the result file. `task_data` is ignored by Git; saved results embed
the arrays required to check the schedule without those local text files.

## Verify Saved Results

```powershell
$files = (Get-ChildItem code/candidate_quantum_medium/results/corrected_20261001/*_seed*.json).FullName
conda run --no-capture-output -n Kaiwu python code/candidate_quantum_medium/verify_result.py $files
```

The verifier checks instance/pool hashes, dimensions, the simple lower bound,
candidate selection, start times, job precedence, machine non-overlap and
selected machine orders. It does not trust the graph decoder's feasibility
flag and does not verify an optimization certificate.

## Correctness And Limits

QUBO energy is `x.T @ Q @ x + constant` for an upper-triangular `Q`.
Tests compare it with squared residuals for legal and illegal binary encodings
and with the installed SDK's Ising conversion. The decoder uses the auxiliary
field spin as a gauge reference. It rejects illegal one-hot readings rather
than repairing them. Legality fractions count distinct binary samples,
including slack bits; they are not frequencies of raw annealer output.

Only HiGHS status 2 supplies an infeasibility proof. Time-limit incumbents must
pass integrality and cut checks. An empty-support active witness forbids the
whole pool. Paths activate only when their length exceeds T. The solver keeps
the verified source incumbent on time, iteration, resource or annealer failure.

`proof_certificate` and `stop_reason` have separate meanings. A pool bound is
not a global JSP bound. SA failure supplies no infeasibility proof. A gap
against `max(job sum, machine sum)` is an upper bound on relative suboptimality,
not a gap against a known optimum.

The loop budget excludes pool generation. Reports include both timings.
HiGHS receives the remaining loop budget. The installed Kaiwu API has no initial-state or
per-call timeout argument; one call may exceed the loop budget, and reports
record the overrun. Its second `solve` argument is `negtail_flip`, not a warm
start. No GPU/QPU timing or quantum-advantage claim applies to these runs.

Results and negative findings appear in
[the run report](../../docs/candidate_medium_results_20261001.md).
