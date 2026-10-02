#!/usr/bin/env python3
"""Paired dynamic-pool ablation; BKS is reporting metadata, never solver input."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from adaptive_search import mq, solve


def fixed_pool_reference(inst, pool, initial, seconds, seed):
    report = mq.run_candidate_loop(inst, pool, master='milp', max_iterations=100000,
                                   time_limit=seconds, seed=seed, log=None)
    orders = mq.choice_orders(pool, report['best_choice'])
    decoded = mq.decode(inst, orders)
    assert report['initial_makespan'] == initial['makespan']
    assert decoded['feasible'] and mq.validate_schedule(inst, decoded['starts']) == report['best_makespan']
    return {'mode':'fixed_pool', 'seed':seed, 'initial_makespan':initial['makespan'],
            'best_makespan':report['best_makespan'], 'iterations':report['iterations'],
            'graph_evaluations':report['evaluation_count'], 'proposal_calls':0, 'proposal_improvements':0,
            'search_seconds':report['wall_seconds'], 'wall_seconds':report['wall_seconds'],
            'best_orders':orders, 'starts':decoded['starts'], 'independent_schedule_verification':True,
            'pool_sha256':mq.content_hash(pool), 'hardware_jobs_submitted':0,
            'stop_reason':report['stop_reason'], 'proof_certificate':report['proof_certificate'],
            'source':'medium_qjsp.run_candidate_loop, unchanged MILP backend'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('benchmark_bounds.json'))
    parser.add_argument('--instances', nargs='+', default=['ta01', 'ta11', 'ta21', 'ta61', 'ta62', 'ta63'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 11])
    parser.add_argument('--modes', nargs='+', choices=['fixed_pool', 'classical', 'uniform', 'quantum'],
                        default=['fixed_pool', 'classical', 'uniform', 'quantum'])
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.instances:
        meta = manifest['instances'][name]
        inst = mq.Instance.read(root / meta['path'])
        identity = mq.content_hash({'durations': inst.durations.tolist(), 'routes': inst.routes.tolist()})
        if identity != meta['instance_sha256']:
            raise ValueError(f'{name}: instance does not match the reference bound')
        for seed in args.seeds:
            tic = perf_counter()
            pool, initial = mq.build_pool(inst, 8, 240, seed)
            init_seconds = perf_counter() - tic
            for mode in args.modes:
                if mode == 'fixed_pool':
                    report = fixed_pool_reference(inst, pool, initial, args.seconds, seed)
                else:
                    report = solve(inst, seconds=args.seconds, seed=seed, mode=mode,
                                   max_iterations=args.iterations, backend=args.backend,
                                   initial_orders=initial['orders'])
                report.update({'benchmark_instance': name, 'reference_upper_bound': meta['upper_bound'],
                    'reference_lower_bound': meta['lower_bound'], 'initial_pool_build_seconds': init_seconds,
                    'end_to_end_seconds': init_seconds + report['wall_seconds'],
                    'gap_to_reference_upper_bound_pct': 100 * (report['best_makespan'] / meta['upper_bound'] - 1),
                    'improvement_from_initial_pct': 100 * (1 - report['best_makespan'] / report['initial_makespan'])})
                filename = f'{name}_seed{seed}_{mode}.json'
                (args.output / filename).write_text(json.dumps(report, indent=2) + '\n')
                row = {k: report[k] for k in ('benchmark_instance','seed','mode','initial_makespan',
                        'best_makespan','reference_upper_bound','gap_to_reference_upper_bound_pct',
                        'improvement_from_initial_pct','iterations','graph_evaluations',
                        'proposal_calls','proposal_improvements','search_seconds','end_to_end_seconds')}
                row['report'] = filename
                rows.append(row)
                (args.output / 'summary.json').write_text(json.dumps({
                    'base_commit': '10682e461f36e5c1a879858d249e3f2fd52ea252',
                    'python': platform.python_version(), 'platform': platform.platform(),
                    'numpy': np.__version__, 'backend_requested': args.backend,
                    'search_budget_seconds': args.seconds, 'max_iterations': args.iterations,
                    'hardware_jobs_submitted': 0, 'rows': rows}, indent=2) + '\n')
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
