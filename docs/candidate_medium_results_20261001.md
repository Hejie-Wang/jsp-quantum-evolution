# Candidate-Master Run Report: 2026-10-01

## Outcome

We continued Kimi's candidate-encoded JSP design, repaired the local medium
implementation, and completed 12 runs: three instance shapes, seeds 7/11,
and MILP/Kaiwu backends. Both MILP runs on the derived 15x20 instance improved
makespan from 1685 to 1675. Kaiwu retained the source incumbent in all six
runs. The 15x15 and 20x15 runs did not improve their baselines.

We reloaded all 12 result files and checked their best schedules using saved
instance arrays, candidate pools, start times and selected orders. All passed.
None of these medium runs proved pool or global optimality. Each reached its
100-iteration budget before its 40-second loop budget.

The medium experiments use **classical** MILP and simulated annealing. This
iteration implements the fixed-T candidate-master and separation workflow,
not the later quantum gate/mixer stages. The 3x3 demo remains an ideal
classical simulation of the mathematical quantum design.

## Provenance And Environments

Remote: `https://github.com/Hejie-Wang/jsp-quantum-evolution.git`, branch `main`.
The sync found Kimi's `4354841` candidate design/demo commit already current.
The approved repair spec was committed as `65a6dea`. Implementation and
experiment files remain local; we did not push changes. We preserved the
pre-existing untracked medium package and historical result snapshots.

All Python commands used conda interpreters:

| Environment | Python | Packages | Use |
|---|---|---|---|
| `Kaiwu` | 3.10.20 | NumPy 2.2.6, SciPy 1.15.3 | Medium tests, MILP, SA, verification |
| `qskit` | 3.11.16 | NumPy 2.4.6, SciPy 1.17.1, Qiskit 2.5.2 | Original demo tests/reproduction; classical compatibility tests |

Kaiwu distribution metadata reports **1.4.1**, but the imported package's
`__version__` reports **1.0.7**. We inspected the installed interface:
`solve(self, ising_matrix=None, negtail_flip=True, sort_solutions=False)`.
We used its keyword arguments and checked conversion algebra against that
installed SDK. See [the environment audit](../code/candidate_quantum_medium/results/corrected_20261001/environment_audit.json).
We did not change conda packages to resolve this packaging inconsistency.
The original benchmark snapshots preserve metadata version 1.4.1; new runner
outputs also record runtime version and interface signature.

## Correctness Repairs

We retained machine-order pools, fixed-T feasibility and graph separation.
We repaired these behaviors:

- Upper-triangular QUBO cross terms now include the factor of two. Reported
  energy includes the constant, so it matches squared residuals on illegal
  one-hot encodings as well as legal encodings.
- Only HiGHS status 2 supplies an infeasibility proof. Time-limit incumbents
  must satisfy integrality, bounds, one-hot constraints and current cuts.
- Always-active witnesses survive projection with empty support. The master
  activates path witnesses only for length greater than T.
- The loop starts from a verified source incumbent. It rejects illegal
  one-hot readings and retains the incumbent on budget/resource/SA failure.
- Kaiwu spin decoding uses the auxiliary field spin as a gauge reference.
  The second `solve` argument is a tail-flip flag, not a warm-start state.
- Reports separate `proof_certificate` from `stop_reason` and save inputs,
  hashes, candidate pools, selected orders, start times, settings, timing,
  sample legality and QUBO allocation counts.

The 2048-variable cap precedes dense QUBO allocation. HiGHS receives the
remaining loop time. The installed Kaiwu interface lacks an interruptible
per-call limit; reports record overruns. These 12 runs had zero loop-budget
overrun. Pool generation takes extra time and appears separately in results.

## Experiment Settings

K=8 candidates per machine, 240 append-only priority-rule dispatch schedules,
seeds 7 and 11. The builder retains the best source schedule's machine orders
as each machine's candidate zero. We built each pool once and supplied that
same pool to both backends; paired pool hashes match for all six pairs.

Each loop allowed 100 iterations and 40 seconds, excluding pool generation.
MILP used a candidate-rank tie-break objective. Kaiwu used initial temperature
100, alpha 0.98, cutoff 0.01, 50 iterations per temperature, size limit 30,
and up to four distinct legal candidate combinations evaluated per iteration.
Both backends used the same cuts and independent graph/schedule checks.

The local inputs are `tai15_15_01_test.txt`, `tai20_15_01_test.txt`, and
`ta21.txt`. **Local ta21 is 20x20.** The 15x20 test keeps its first 15 jobs
and all 20 machines; it is a **derived instance, not a published benchmark**.
The ta21 source SHA256 is
`4316eb2cb5900589da736b4d76e39524b3723d9a6bfffa00a1292b4b8a476dea`.
Results save the derivation and arrays. Local `task_data` remains Git-ignored.

## Measured Results

The times below measure the cut loop in seconds, including master and graph
evaluation. See [summary.csv](../code/candidate_quantum_medium/results/corrected_20261001/summary.csv)
for pool-plus-loop time, pool hashes and result paths, and
[summary.json](../code/candidate_quantum_medium/results/corrected_20261001/summary.json)
for the same rows in structured form.

| Shape | Seed | Simple LB | Initial | MILP Best | MILP Seconds | Kaiwu Best | Kaiwu Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15x15 | 7 | 977 | 1462 | 1462 | 30.39 | 1462 | 37.08 |
| 15x15 | 11 | 977 | 1462 | 1462 | 21.42 | 1462 | 37.29 |
| 20x15 | 7 | 1139 | 1865 | 1865 | 25.21 | 1865 | 29.06 |
| 20x15 | 11 | 1139 | 1865 | 1865 | 24.32 | 1865 | 32.57 |
| 15x20 derived | 7 | 1217 | 1685 | 1675 | 19.98 | 1685 | 37.44 |
| 15x20 derived | 11 | 1217 | 1685 | 1675 | 18.56 | 1685 | 38.19 |

The simple LB is `max(job sum, machine sum)`. It does not give a known optimum.
The 10-unit 15x20 improvement is about 0.59% of the source makespan. The two
seeds on one instance do not establish performance across the instance family.
These timings include neither gate compilation nor quantum-device execution.

### Kaiwu Readings

Each SA run returned 3000 binary configurations after per-call deduplication,
including slack bits. The fractions below count those configurations, not
raw annealer frequencies. Configurations may recur across calls. "Cut-valid"
means legal one-hot and satisfied current projected cuts; it does not imply
an acyclic graph or a schedule meeting T. The peak variable count includes
candidate bits and slack bits, before the added Ising field spin.

| Shape | Seed | Illegal One-Hot | Cut-Valid Readings | Peak QUBO Variables | Graph Evaluations Including Initial |
|---|---:|---:|---:|---:|---:|
| 15x15 | 7 | 87.53% | 290 | 213 | 83 |
| 15x15 | 11 | 86.83% | 371 | 213 | 105 |
| 20x15 | 7 | 92.97% | 199 | 200 | 68 |
| 20x15 | 11 | 92.87% | 205 | 199 | 73 |
| 15x20 derived | 7 | 88.87% | 238 | 255 | 98 |
| 15x20 derived | 11 | 93.57% | 132 | 244 | 77 |

Kaiwu found some zero-residual master readings, but no better schedule in
these runs. Illegal one-hot readings account for 86.83%-93.57% of configurations
under these settings. Pool quality, cut coverage and sampler legality need
separate experiments; these results do not identify which change would
improve schedule quality. They provide no evidence of quantum advantage.

## Demo And Tests

The original six demo tests pass. Reproduction preserves pool optimum 11 and
ground energy 11. With tau=20 and 80 midpoint steps, the ordinary driver gives
ground probability 0.0190864; the joint driver gives 0.0119777. The joint driver
performed worse under this setting. We preserved that negative result in
[the demo reproduction](../code/candidate_quantum_demo/results/codex_20261001_reproduction.json).

All **22 medium tests** pass in the Kaiwu conda environment. They cover the
3x3 pool optimum 11 and its MILP pool proof, binary/slack energy enumeration,
SDK conversion and gauge invariance, statuses/incumbents, path thresholds,
unavoidable cuts, invalid pools, rejection of illegal readings, saved-result
corruption, and incumbent retention on exits. Tiny enumeration lives in
tests/demo only. It does not form part of the medium solver.

We checked the CLI's derived-instance and zero-budget behavior. We then
reloaded the 12 benchmark JSONs in a separate process and validated job
precedence, machine non-overlap, selected orders, makespan and hashes. This
checks feasible schedule evidence; it does not certify an optimality claim.

## Reproduce

See [the package README](../code/candidate_quantum_medium/README.md) for conda
commands to run tests, the paired benchmark, a single 15x20 run and saved-file
verification. Re-running into `corrected_20261001` overwrites that experiment
directory's matching result files; use a new `--output-dir` to keep a new run.

The completed result is a correctness-checked candidate-master baseline at
medium scale. The next design phase needs separate gate/mixer implementation
and tests before a quantum-method comparison. Candidate coverage and classical
SA quality remain unresolved; none of the medium outcomes proves optimality.
