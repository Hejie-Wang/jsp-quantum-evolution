#!/usr/bin/env python3
"""Reproduce same-process component timings and optional GPU parity checks."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np
from adaptive_search import gt_initial, mq
from batch_evaluator import BatchEvaluator, INVALID
from search_loop import demo_candidate_pool, covering_demo_witnesses, train_layer_params


def timed(function, repeats=3):
    samples = []
    for _ in range(repeats):
        start = perf_counter(); value = function(); samples.append(perf_counter() - start)
    return value, {'median_seconds':median(samples), 'samples_seconds':samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instance', type=Path)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inst, rng = mq.Instance.read(args.instance), np.random.default_rng(41)
    p, routes = np.array(inst.processing, np.int32), np.array(inst.machine_of, np.int32)
    base = gt_initial(p, routes, inst.jobs, inst.machines, 41)
    batch = np.repeat(base[None], args.batch_size, axis=0)
    for orders in batch:
        for _ in range(int(rng.integers(0,4))):
            m = int(rng.integers(inst.machines)); a = int(rng.integers(inst.jobs-1))
            orders[m,a], orders[m,a+1] = orders[m,a+1], orders[m,a]
    def reference():
        decoded = [mq.decode(inst, orders) for orders in batch]
        return np.array([d['makespan'] if d['feasible'] else INVALID for d in decoded])
    cpu = BatchEvaluator(inst)
    cpu.evaluate(batch)  # JIT warmup is excluded from steady-state component timing.
    expected, reference_time = timed(reference)
    values, cpu_time = timed(lambda:cpu.evaluate(batch))
    np.testing.assert_array_equal(values, expected)
    result = {'instance':inst.name, 'batch_size':len(batch), 'feasible':int(np.sum(values<INVALID)),
              'python_graph':reference_time, 'numba_batch':cpu_time, 'cpu_parity':True,
              'cuda_requested':args.cuda, 'cuda_tested':False, 'hardware_jobs_submitted':0}
    if args.cuda:
        gpu = BatchEvaluator(inst,'cuda')
        gpu.evaluate(batch)
        values, gpu_time = timed(lambda:gpu.evaluate(batch))
        np.testing.assert_array_equal(values,expected)
        result.update(cuda_tested=True, cuda_parity=True, cuda_batch=gpu_time)
    pool = demo_candidate_pool(); _, specs = covering_demo_witnesses(pool)
    reports = {}
    for backend in ('qiskit','compact'):
        train = lambda:train_layer_params(pool,specs,(),13,1,layer_counts=(1,2),
                                         max_evaluations=24,mode='xy',simulation_backend=backend)
        report, timing = timed(train)
        reports[backend] = {'timing':timing,'expected_energy':report['expected_energy'],
                            'params':report['params'],'evaluations':report['evaluations']}
    np.testing.assert_allclose(reports['qiskit']['expected_energy'],reports['compact']['expected_energy'],atol=1e-10)
    np.testing.assert_allclose(reports['qiskit']['params'],reports['compact']['params'],atol=1e-10)
    result['quantum_training'] = reports
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
