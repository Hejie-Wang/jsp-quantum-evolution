#!/usr/bin/env python3
"""Compare bounded quantum simulation with explicit classical local baselines."""
from pathlib import Path
import json
import platform
import numpy as np
import qiskit
import scipy
from qiskit import transpile
from qiskit.quantum_info import Statevector

from bounded_qjsp import (Budget, build_circuit, check_compiled_budget, fit_angles,
                          phase_model, probabilities, run_request, simulate_search)
from qjsp import Instance, demo_instance, export_schedule, write_json


def main():
    root=Path(__file__).resolve().parent
    output=root/'results'; output.mkdir(exist_ok=True)
    budget=Budget(bits=4,layers=2,max_rounds=20,max_cost_evaluations=320,shots=1024,train_evaluations=96)
    rows=[]
    instances=[demo_instance(),demo_instance(5,5),demo_instance(10,5)]
    instances += [Instance.read(path) for path in sorted((root/'data').glob('tai50_20_*.txt'))]
    for inst in instances:
        for mode in ('ideal_quantum','uniform','exact_local'):
            report,request=simulate_search(inst,budget,seed=7,mode=mode)
            write_json(output/f'{inst.name}_{mode}.json',report)
            export_schedule(inst,report['schedule'],output/f'{inst.name}_{mode}.csv')
            row={key:report[key] for key in ('instance','mode','initial_classical_makespan',
                 'final_makespan','seconds','local_table_cost_evaluations','ideal_parameter_objective_evaluations')}
            row['maximum_local_bits']=max((r['bits'] for r in report['trace']),default=0)
            if request is not None:
                request_path=output/f'{inst.name}_request.json';write_json(request_path,request)
                compiled,_=run_request(request,output/f'{inst.name}_compiled')
                row['compiled_last_block']=compiled['resources']
            rows.append(row)
            print(inst.name,mode,report['initial_classical_makespan'],'->',report['final_makespan'],
                  f"{report['seconds']:.3f}s",flush=True)
    # Same four exact JSP costs as the user's upload, reordered into Qiskit bits.
    costs=np.array([31,18,32,31])
    phase=phase_model(costs)
    tuned=[]
    for layers in (1,2):
        angles,training=fit_angles(phase,layers=layers,evaluations=256,seed=7)
        circuit=build_circuit(phase,angles)
        compiled=transpile(circuit,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=7)
        p=Statevector.from_instruction(compiled).probabilities()
        tuned.append({'layers':layers,'angles':angles.tolist(),'ideal_p_optimal':float(p[1]),
                      'ideal_probabilities_qiskit_order':p.tolist(),
                      'mean_cost':float(p@costs),'training':training,
                      'abstract_all_to_all_compilation':check_compiled_budget(compiled,Budget()),
                      'hardware_run':False})
    write_json(output/'tuned_2x2_ideal_only.json',tuned)
    write_json(output/'benchmark.json',{'environment':{'python':platform.python_version(),
                 'numpy':np.__version__,'scipy':scipy.__version__,'qiskit':qiskit.__version__},
                 'budget':budget.__dict__,'runs':rows,
                 'hardware_jobs_submitted':0,
                 'notes':['single fixed-seed run per case; not a speedup benchmark',
                          'quantum and uniform engines sample the explicitly enumerated local table',
                          'exact_local already knows the best local option, and is an essential baseline',
                          'compiled counts target a synthetic line, NOT calibrated ibm_fez']})


if __name__=='__main__':main()
