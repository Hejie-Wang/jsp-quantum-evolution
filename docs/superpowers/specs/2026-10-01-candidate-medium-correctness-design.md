# Candidate-Master Correctness and Medium-Scale Verification

## Scope

Continue the Kimi design in `docs/quantum_candidate_design.md` using the existing,
uncommitted `code/candidate_quantum_medium` implementation. Keep machine-candidate
encoding, fixed-T feasibility, graph separation, and independently checked
schedules. Do not enumerate the candidate Cartesian product in production.

This work repairs the classical interfaces and compares scipy/HiGHS MILP with
Kaiwu SDK simulated annealing. It does not implement Qiskit projector gates,
joint quantum mixers, CUDA, QPU submissions, or claims of quantum advantage.
All Python commands use existing conda environments: `Kaiwu` and `qskit`.

## Confirmed Defects

- QUBO affine-square expansion omits a factor of two in upper-triangular cross
  terms and omits the constant from reported energy. Illegal one-hot states can
  share the minimum. Existing tests check legal assignments only.
- MILP time-limit status is misclassified as infeasibility, permitting false
  pool-optimality certificates.
- Unavoidable witnesses with empty control support are discarded.
- Kaiwu samples are silently repaired with argmax. The known feasible source
  schedule is not retained as the initial incumbent.
- Runtime limits are checked only between master calls, and saved results lack
  sufficient schedule and input evidence for independent reproduction.

## Design

### QUBO and Kaiwu

Use an explicitly upper-triangular QUBO matrix with doubled cross coefficients.
Return the constant offset so `x.T @ Q @ x + offset` equals the sum of squared
residuals for every binary configuration, including illegal encodings. Preserve
the constant penalty for an always-active cut with empty support.

Convert through the installed Kaiwu SDK. Decode spin samples relative to the
auxiliary field spin introduced by QUBO-to-Ising conversion, remove duplicate
binary samples, and sort by offset-inclusive energy. Count illegal one-hot
samples and do not silently repair them or count them as successful readings.
Independently decode legal candidates even when existing cuts are violated;
such results may supply new valid witnesses, but are not master successes.

### MILP and Witnesses

Only solver status 2 proves infeasibility. A time-limit result is unknown, with
any returned incumbent accepted only after checking integrality, one-hot
constraints, and all current cuts. Solver errors must not produce certificates.
Pass the remaining outer-loop time to the MILP master.

Retain raw cycle/path witnesses. At fixed T, activate cycles and only paths
longer than T. Drop never-active witnesses, not always-active witnesses. An
empty-support active witness makes the master infeasible. All path conditions
remain intersections of machine precedence relations.

### Incumbent, Limits, and Results

Validate pools before solving: nonempty machines, unique candidates, exact
machine-operation permutations. Try the supplied initial choice (the builder's
all-zero choice) before sampling and retain a verified incumbent on budget
exhaustion or annealing failure. Each improvement is independently validated.

Record stop reason separately from proof status. SA failure is never an
infeasibility proof. Pool bounds are never global JSP bounds. Persist the best
start times, machine orders, instance arrays, input hashes, pool identity,
settings, master/evaluation timing, sample legality, and QUBO resource counts.
Place a variable cap before dense QUBO allocation. Kaiwu lacks an interruptible
per-call time limit in this interface; record any single-call budget overrun.

## Verification

1. Keep the six existing Kimi mathematical demo tests passing.
2. Exhaustively test all binary encodings and small slack assignments against
   direct squared residuals on tiny fixtures, including one-hot violations.
3. Mock MILP timeout, proven infeasibility, errors, and valid/invalid incumbents.
4. Test unavoidable witnesses, path threshold boundaries, invalid pools, budget
   exit with incumbent retention, SDK gauge decoding, and no silent repair.
5. Recheck the fixture pool optimum 11 with MILP; SA is a sampler without proof.
6. Compare identical generated pools (K=8, 240 schedules, seeds 7 and 11 where
   practical) on 15x15, 20x15, and a clearly labeled 15x20 derived instance.
   The derived instance selects the first 15 jobs of local `ta21.txt`, which
   actually contains 20 jobs and 20 machines. It is not a published benchmark.
7. Independently revalidate every saved best schedule from saved instance data,
   rather than relying on the solver's feasibility flag. Report negative results
   and heuristic-only outcomes without claiming optimality or quantum speedup.

## Delivery

Preserve all pre-existing uncommitted files and historical result snapshots.
Make scoped changes to the medium package, add regression tests and a runnable
benchmark entry point, and document commands and measured results. Keep original
demo results unchanged; new reproduction files use distinct filenames.
