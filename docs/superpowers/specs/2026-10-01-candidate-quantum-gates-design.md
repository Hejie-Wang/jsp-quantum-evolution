# Offline Candidate Quantum Gates

Date: 2026-10-01

## Goal and Scope

Continue Kimi's machine-candidate design in `docs/quantum_candidate_design.md`
and its implementation handoff. Convert witness phases and candidate mixers
into actual Qiskit circuits, validate them locally, and measure compilation
resources for a derived 15-job, 20-machine problem. Preserve the existing
mathematical demo, medium solver, user results, and historical negative results.

This phase does not promise quantum advantage, original-JSP global optimality,
or a quantum solution of the medium instance. Small gate correctness and
medium offline compilation are separate deliverables.

## Hard Offline Boundary

- Run all Python through the existing conda `qskit` environment.
- Use local `qiskit.quantum_info`, Qiskit Aer, and offline `transpile` only.
- Do not import IBM Runtime/providers, load accounts or credentials, discover
  cloud backends, submit jobs to hardware, or contact quantum services.
- Do not install packages or change environments unless separately authorized.
- Compile against explicitly defined generic basis gates and, optionally, an
  offline coupling map. Such a target is not a calibrated hardware backend.
- Persist `execution_mode=local_simulator` or `offline_resource_compile` and
  `hardware_jobs_submitted=0`. Tests audit prohibited provider/account calls;
  the CLI must expose no hardware option.

## Architecture and Data Contracts

Add an independent package under `code/candidate_quantum_gates/`, using the
existing classical instance, pool-validation, graph-decoding, witness, and
schedule-verification interfaces where possible. Keep imports free of jobs.

The pool identity includes the instance and ordered candidate contents, a
content hash, and version. Witnesses retain raw nodes and machine precedence
relations; allowed-label masks are derived and bound to the pool hash.
Recompute masks after any pool update. Reject malformed pools, stale masks,
unsupported witness kinds, and out-of-range labels rather than repairing them.

Joint transitions retain support, left/right labels, weight, witness provenance,
and pool identity. Requests include fixed-T or register-T mode, phase scale,
product-formula order, steps, shots, seed, and limits. Results record raw counts,
illegal one-hot counts, verified schedules, witness feedback, timings, circuit
resources, and whether constraints/transitions were omitted.

## Witness Phases

Assign candidate qubits machine-major, candidate-minor. Within each machine
exactly one bit is set. Assign binary T bits least-significant-first. Qiskit
counts are decoded explicitly from classical-bit indices, not display order.

Initialize a legal candidate basis state or a per-machine single-excitation
superposition. For each witness, intersect all relations on the same machine
before computing allowed-label flags. On the legal one-hot subspace, XORing
allowed candidate bits into a zero flag computes membership reversibly.

Remove full-set conditions and skip empty-set witnesses. An empty support is
a constant true predicate, not a reason to discard the witness. Apply
`exp(-i * angle * predicate)` and reverse every flag/comparator computation.
Cycle predicates are conjunctions of machine flags. Register-T paths add the
condition `T < length`, using a reversible integer comparator; fixed-T paths
are included exactly when `length > T`. Thresholds outside the represented
range are handled explicitly. The register-T objective adds weighted bit
phases for T, with penalty `Lambda > sum(processing_times)`.

Never synthesize phases from an enumerated cost table or a global diagonal
matrix. Complete tiny-demo witness coverage is a test oracle only. Production
witnesses come from classical decoding of sampled or saved choices.

## Mixers and Evolution

Use connected per-machine XY exchanges as the baseline, preserving excitation
count. A ring or complete graph is an explicit request setting; the first
comparison uses the same graph and weights in both variants. Register-T uses
single-bit X mixing. Phase gates do not create one-hot leakage.

Propose joint transitions from 2--4 nontrivially constrained machines of a
current witness, without querying optimal labels. Each term connects two
legal support configurations and acts as identity outside its support.
Compile its two-level rotation using reversible basis permutations and a
controlled rotation on support qubits only, not a dense full-space unitary.
Undo the basis permutations. Each endpoint must differ on all support
machines; retain ordinary XY terms to preserve connectivity.

Record a fixed ordering and first-order product formula. Use midpoint schedules
with a joint catalyst proportional to `4*s*(1-s)`, zero at both endpoints.
Normalize with conservative operator-norm bounds derived from term weights,
not by enumerating energies. Include all raw scales in results, apply the same
normalization policy across variants, and test step refinement on tiny spaces.
Do not claim that finite-step reordered noncommuting gates are equivalent.

## Small Sampling and Feedback

Use Kimi's fixed 3x3 fixture first. Compare ordinary and witness-guided joint
circuits at predefined parameters and multiple seeds, retaining regressions.
Use fixed-T for the initial feedback loop to reduce simulator size; test the
register-T phase separately. Keep the verified incumbent throughout.

Each round compiles current witnesses, samples locally, rejects illegal
encodings without argmax repair, classically decodes unique legal choices,
verifies feasible schedules, and adds valid cycle/over-target path witnesses.
Never use exhaustive optimum labels to drive feedback. Tiny exhaustive
validation and exact classical-master diagnostics are separately timed.

Stop on round/evaluation/time limits and return the verified incumbent.
Sampling failure does not certify infeasibility or increase a lower bound.
Report sampling shots and unique graph evaluations separately; comparisons
use both equal-shot/evaluation budgets and measured end-to-end time, without
claiming a matched-time advantage from a single run.

## Medium Offline Resources

Reuse saved corrected 15x20 data and its validated K=8 pool. This is a derived
case from ta21, not a published 15x20 benchmark. Candidate data require 160
qubits; report clock and peak reusable ancillas in addition to this number.

Build one-layer ordinary/joint circuits from available valid raw witnesses.
First use at most 32 witnesses and 8 joint terms; store selection order, full
counts, included identifiers, and omitted counts. These limits make the layer
an incomplete witness model and must be labeled as such. Report logical
operations before lowering and basis-gate counts/depth after offline lowering,
including uncomputation. Routed and unrouted resources are distinct; do not
present unrouted depth as hardware depth. No medium statevector or full
candidate-combination enumeration is permitted.

Simulation checks must reject more than 20 total qubits before allocating a
statevector. Default simulator memory budget is 512 MiB. Compilation limits
include 2,000,000 lowered instructions and 60 seconds per resource case, with
explicit aborted/partial results rather than guessed counts. Start with a
smaller witness subset if synthesis exceeds these limits. Any resource claim
identifies the actual witness subset and actual lowering completed.

## Verification and Acceptance

1. Test allowed-mask compilation, pool-version mismatch, input validation,
   empty/full supports, and unambiguous bit ordering.
2. Compare individual gate phases against analytical witness predicates on
   every tiny legal basis state and random legal superpositions. Include
   `T=L-1`, `T=L`, `T=L+1`, objective phases, and ancilla return to zero.
3. Check XY/joint unitarity, one-hot preservation, transition endpoints,
   identity outside support, and baseline connectivity. Compare gate action
   to independently built small reference operators, up to global phase.
4. Check product-formula refinement and equal-scale accounting. Retain worse
   joint outcomes; success is correct execution, not mandatory improvement.
5. Run feedback from incomplete witnesses and independently verify every
   reported improved schedule. Exercise illegal counts and budget exits.
6. Test offline boundaries and simulation size rejection; run existing demo
   and medium tests as regressions in their respective conda environments.
7. Save reproducible local experiment/resource JSON, environment versions,
   parameters, pool/witness hashes, and a concise report distinguishing
   implemented tests, simulation outcomes, and medium compilation limits.

## Risks and Deferred Work

High-control phases and support rotations can dominate depth despite reusable
ancillas. Gate-level simulation is much more expensive than the old legal-label
demo; the fixed-T loop and explicit size caps contain this cost. Candidate pools
and partial witnesses can limit solution quality independently of dynamics.
No noise-model benchmark, QPU access, CUDA integration, error correction,
50x20 execution, or tuning-based quantum-advantage claim is included.
