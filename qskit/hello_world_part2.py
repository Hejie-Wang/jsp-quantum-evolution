"""Hello World Part 2: 100-qubit GHZ state on a real IBM QPU."""
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.options_models import EstimatorOptions
from qiskit_ibm_runtime.executor_estimator import Estimator


def get_qc_for_n_qubit_GHZ_state(n: int) -> QuantumCircuit:
    """Create a QuantumCircuit that prepares the n-qubit GHZ state."""
    if isinstance(n, int) and n >= 2:
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
    else:
        raise Exception("n is not a valid input")
    return qc


# Step 1: Map the problem
n = 100
qc = get_qc_for_n_qubit_GHZ_state(n)

# ZZII...II, ZIZI...II, ... , ZIII...IZ
operator_strings = [
    "Z" + "I" * i + "Z" + "I" * (n - 2 - i) for i in range(n - 1)
]
operators = [SparsePauliOp(operator) for operator in operator_strings]

# Step 2: Optimize for the target backend
service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True, min_num_qubits=100)
print(f"backend: {backend.name} ({backend.num_qubits} qubits)", flush=True)

pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
isa_circuit = pm.run(qc)
isa_operators_list = [op.apply_layout(isa_circuit.layout) for op in operators]

# Step 3: Execute with error suppression (dynamical decoupling)
options = EstimatorOptions()
options.resilience_level = 1
options.dynamical_decoupling.enable = True
options.dynamical_decoupling.sequence_type = "XY4"

estimator = Estimator(backend, options=options)

job = estimator.run([(isa_circuit, isa_operators_list)])
job_id = job.job_id()
print(f"Job ID: {job_id}", flush=True)

result = job.result()[0]

# Step 4: Post-process
data = list(range(1, len(operators) + 1))
values = result.data.evs
values = [v / values[0] for v in values]  # normalize

print("\n<Z0 Zi> / <Z0 Z1>:")
for d, v in zip(data, values):
    print(f"  i={d}: {v:.4f}")

plt.figure(figsize=(8, 4.5))
plt.plot(data, values, marker="o", markersize=3, label="100-qubit GHZ state")
plt.xlabel("Distance between qubits $i$")
plt.ylabel(r"$\langle Z_i Z_0 \rangle / \langle Z_1 Z_0 \rangle $")
plt.title(f"100-qubit GHZ on {backend.name} (job {job_id})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hello_world_part2.png", dpi=150)
print("\nPlot saved to hello_world_part2.png")
