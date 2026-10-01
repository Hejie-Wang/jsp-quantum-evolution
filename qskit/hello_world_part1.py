"""Hello World Part 1: 2-qubit Bell state on a real IBM QPU."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_estimator import Estimator

# Step 1: Map the problem
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
observables = [SparsePauliOp(label) for label in observables_labels]

# Step 2: Optimize for the target backend
service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True)
print(f"backend: {backend.name} ({backend.num_qubits} qubits)", flush=True)

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)

# Step 3: Execute with the Estimator primitive
estimator = Estimator(mode=backend)
estimator.options.resilience_level = 1
estimator.options.default_shots = 5000

mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]

job = estimator.run([(isa_circuit, mapped_observables)])
print(f"Job ID: {job.job_id()}", flush=True)

job_result = job.result()
pub_result = job_result[0]

# Step 4: Analyze
values = pub_result.data.evs
errors = pub_result.data.stds

print("\nExpectation values:")
for label, v, e in zip(observables_labels, values, errors):
    print(f"  {label}: {v:+.4f} +/- {e:.4f}")

plt.figure(figsize=(7, 4))
plt.errorbar(observables_labels, values, yerr=errors, fmt="-o", capsize=4)
plt.xlabel("Observables")
plt.ylabel("Values")
plt.title(f"Hello World: Bell state on {backend.name} (job {job.job_id()})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hello_world_part1.png", dpi=150)
print("\nPlot saved to hello_world_part1.png")
