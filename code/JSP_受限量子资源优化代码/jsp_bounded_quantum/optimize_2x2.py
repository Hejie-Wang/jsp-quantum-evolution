"""Reproduce barrier-free original evolution and export tuned shallow request."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.quantum_info import Statevector
from bounded_qjsp import Budget, hadamard_transform, run_request
from qjsp import write_json

def main():
    out=Path('results');out.mkdir(exist_ok=True)
    costs=np.array([31,18,32,31]);coeff=hadamard_transform(costs)/4
    rows=[]
    for layers in [1,2,5,10]:
        qc=QuantumCircuit(2);qc.h(range(2));h=20/layers
        for ell in range(layers):
            s=.95*(ell+.5)/layers
            qc.rz(2*s*h*coeff[1],0);qc.rz(2*s*h*coeff[2],1)
            qc.rzz(2*s*h*coeff[3],0,1)
            qc.rx(-(1-s)*h,0);qc.rx(-(1-s)*h,1)
        compiled=transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=7)
        a=Statevector.from_instruction(qc).probabilities();b=Statevector.from_instruction(compiled).probabilities()
        rows.append(dict(layers=layers,ideal_p_optimal=float(a[1]),compiled_cx=int(compiled.count_ops().get('cx',0)),depth=compiled.depth(),probability_max_error=float(np.max(abs(a-b)))))
        with (out/f'original_2x2_L{layers}_optimized.qpy').open('wb') as f:qpy.dump(compiled,f)
    write_json(out/'barrier_free_reproduction.json',rows)
    tuned=json.loads((out/'tuned_2x2_ideal_only.json').read_text())[1]
    request=dict(instance=json.loads(Path('data/demo_2x2.json').read_text()),orders_zero_based=[[1,2],[0,3]],moves=[[0,0],[1,0]],costs=costs.tolist(),angles=tuned['angles'],budgets=asdict(Budget(bits=2,layers=2,train_evaluations=256)))
    write_json(out/'tuned_2x2_request.json',request)
    run_request(request,out/'tuned_2x2_compiled')
    from hardware_once import decode_hardware
    assert decode_hardware(request,{'01':1024})['makespan_after']==18
    assert decode_hardware(request,{'10':1024})['makespan_after']==31
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
