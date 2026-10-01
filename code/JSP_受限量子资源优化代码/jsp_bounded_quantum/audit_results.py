#!/usr/bin/env python3
"""Audit uploaded counts. Does not infer a specific noise cause from them."""
import argparse
import json
from pathlib import Path
import numpy as np
from qjsp import write_json


def wilson(successes,n,z=1.95996398454):
    p=successes/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half),float(center+half)]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',required=True)
    ap.add_argument('--output',default='results/upload_audit.json')
    args=ap.parse_args()
    rows=[];rng=np.random.default_rng(20260930)
    for path in sorted(Path(args.input_dir).glob('hw_L*.json')):
        report=json.loads(path.read_text())
        counts=report['hardware']['counts'];n=sum(counts.values())
        actual=np.zeros(4)
        for key,count in counts.items():
            key=key.replace(' ','')
            actual[2*int(key[-1])+int(key[-2])]=count/n
        ideal=np.array(report['ideal_probabilities_numpy_index_order'])
        tv=float(.5*np.abs(actual-ideal).sum())
        null_counts=rng.multinomial(n,ideal,size=20000)/n
        null_tv=.5*np.abs(null_counts-ideal).sum(axis=1)
        optimum=int(np.argmin(report['cost_table_numpy_index_order']))
        successes=int(round(actual[optimum]*n))
        rows.append({'layers':report['layers'],'tau':report['tau'],'shots':n,
                     'ideal_p_optimal':float(ideal[optimum]),'hardware_p_optimal':successes/n,
                     'hardware_p_optimal_wilson95':wilson(successes,n),
                     'two_qubit_gates':report['transpiled']['two_qubit_gates'],
                     'reported_circuit_seconds':report['transpiled']['scheduled_duration_seconds'],
                     'tv_recomputed':tv,'tv_under_ideal_multinomial_99_percentile':float(np.quantile(null_tv,.99)),
                     'monte_carlo_tail_estimate_plus_one':float((1+np.sum(null_tv>=tv))/20001),
                     'ideal_numpy_vs_qiskit_difference':report['statevector_vs_numpy_max_prob_diff']})
    rows.sort(key=lambda r:r['layers'])
    write_json(args.output,{'rows':rows,'null_hypothesis':'independent ideal-distribution shots; reference only',
        'diagnosis':'Differences exceed shot noise under this null; archived data cannot isolate transpilation, calibration, readout or other execution errors.',
        'missing_evidence':['executed QPY','physical layout','calibration snapshot','resolved runtime options','actual usage']})
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
