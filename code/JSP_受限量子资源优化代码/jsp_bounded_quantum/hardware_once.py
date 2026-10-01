#!/usr/bin/env python3
"""Compile one bounded request; --submit explicitly runs one IBM hardware job.

No account creation, no embedded credentials, no automatic backend/plan choice.
Backend-connected and Runtime execution paths require user-side verification;
the delivered project was tested offline and spent zero QPU usage.
"""
import argparse
import json
import math
from pathlib import Path
import platform

import numpy as np
import qiskit

from bounded_qjsp import apply_moves, phase_model, probabilities, request_circuit, run_request
from qjsp import evaluate_orders, export_schedule, validate_schedule, write_json


def decode_hardware(request, counts):
    inst, phase, _, _ = request_circuit(request)
    shots = sum(counts.values())
    if shots <= 0 or any(int(v) != v or v < 0 for v in counts.values()):
        raise ValueError("invalid measurement counts")
    observed = np.zeros(len(request['costs']), dtype=float)
    for key, count in counts.items():
        key = key.replace(' ', '')
        if len(key) != phase['bits'] or any(ch not in '01' for ch in key):
            raise ValueError("unexpected measurement bitstring")
        observed[int(key, 2)] += count/shots
    before = evaluate_orders(inst, request['orders_zero_based'])
    orders = request['orders_zero_based']; best = before
    for x in np.flatnonzero(observed):
        candidate = apply_moves(orders=request['orders_zero_based'], moves=request['moves'], bit_index=int(x))
        decoded = evaluate_orders(inst, candidate)
        if decoded['feasible'] and decoded['makespan'] < best['makespan']:
            orders, best = candidate, decoded
    validate_schedule(inst, best['starts'])
    ideal = probabilities(phase['approximation'], request['angles'])
    costs = np.asarray(request['costs'])
    return {"shots": shots, "counts": counts, "orders_zero_based": orders,
            "incumbent_before": before['makespan'], "makespan_after": best['makespan'],
            "schedule": best, "tv_distance_to_ideal": float(.5*np.abs(observed-ideal).sum()),
            "local_minimum_probability_diagnostic": float(observed[costs == costs.min()].sum()),
            "global_optimum_claim": False}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request',required=True)
    ap.add_argument('--backend',required=True)
    ap.add_argument('--ibm-instance',required=True)
    ap.add_argument('--submit',action='store_true')
    ap.add_argument('--max-job-seconds',type=int,default=10)
    ap.add_argument('--output-dir',default='hardware_result')
    args=ap.parse_args()
    if args.max_job_seconds < 1: ap.error('max-job-seconds must be positive')
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.executor_sampler import Sampler
    import qiskit_ibm_runtime
    request=json.loads(Path(args.request).read_text())
    service=QiskitRuntimeService(instance=args.ibm_instance)
    backend=service.backend(args.backend)
    report,compiled=run_request(request,args.output_dir,backend)
    output=Path(args.output_dir)
    report['environment']={'python':platform.python_version(),'qiskit':qiskit.__version__,
                           'qiskit_ibm_runtime':qiskit_ibm_runtime.__version__}
    report['physical_mapping']=[{'physical_qubit':compiled.find_bit(i.qubits[0]).index,
                                'classical_bit':compiled.find_bit(i.clbits[0]).index}
                               for i in compiled.data if i.operation.name=='measure']
    props=backend.properties()
    if props is not None:
        (output/'calibration.json').write_text(json.dumps(props.to_dict(),default=str,indent=2)+'\n')
    target=backend.target
    used={compiled.find_bit(q).index for i in compiled.data if i.operation.name not in ('delay','barrier') for q in i.qubits}
    resets=target.get('reset',{})
    if any((q,) not in resets or resets[(q,)] is None or resets[(q,)].duration is None for q in used):
        raise ValueError('missing reset duration: cannot produce a complete preflight time estimate')
    init_s=max(resets[(q,)].duration for q in used)
    rep_delay=backend.default_rep_delay
    if rep_delay is None:raise ValueError('backend repetition interval unavailable')
    duration=report['resources']['scheduled_duration_seconds']
    shots=request['budgets']['shots']
    estimate=2+shots*(duration+init_s+rep_delay)
    report['estimated_single_subjob_usage_seconds']=float(estimate)
    report['estimate_is_not_a_billing_guarantee']=True
    report['server_max_execution_time_seconds']=args.max_job_seconds
    write_json(output/'preflight.json',report)
    if estimate>args.max_job_seconds:
        raise ValueError('estimated usage exceeds --max-job-seconds; no job submitted')
    print(json.dumps({'backend':backend.name,'resources':report['resources'],
                      'estimated_seconds':estimate,'will_submit':args.submit},indent=2))
    if not args.submit:return
    sampler=Sampler(mode=backend)
    sampler.options.execution.init_qubits=True
    sampler.options.execution.rep_delay=rep_delay
    sampler.options.twirling.enable_gates=False
    sampler.options.twirling.enable_measure=False
    sampler.options.dynamical_decoupling.enable=False
    sampler.options.max_execution_time=args.max_job_seconds
    # The Runtime limit is a server timeout, not an exact invoicing cap; no
    # automatic retry is performed if this single submission fails or times out.
    options=sampler.options.model_dump(mode='json')
    write_json(output/'runtime_options.json',options)
    job=sampler.run([compiled],shots=shots)
    report['job_id']=job.job_id();report['hardware_jobs_submitted']=1
    write_json(output/'submitted_job.json',report)
    result=job.result()
    decoded=decode_hardware(request,result[0].data.meas.get_counts())
    report.update(decoded)
    usage=None
    if hasattr(job,'usage'):
        try:usage=job.usage()
        except Exception as exc:report['usage_query_error']=type(exc).__name__
    report['actual_usage']=usage
    # Result metadata may include provider-specific datetime objects.
    report['result_metadata']=json.loads(json.dumps(result[0].metadata,default=str))
    report['actual_usage']=json.loads(json.dumps(usage,default=str))
    write_json(output/'hardware_result.json',report)
    inst,_,_,_=request_circuit(request)
    export_schedule(inst,decoded['schedule'],output/'schedule.csv')
    print(json.dumps({'job_id':report['job_id'],'before':decoded['incumbent_before'],
                      'after':decoded['makespan_after'],'tv_distance':decoded['tv_distance_to_ideal']},indent=2))


if __name__=='__main__':main()
