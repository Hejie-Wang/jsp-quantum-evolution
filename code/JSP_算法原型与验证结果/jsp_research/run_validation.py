"""Re-run all mathematical/feasibility checks offline using included data."""
from pathlib import Path
import argparse
import json
from jsp_core import Instance,analyze,validate_starts,subset_lower_bound,export_schedule,small_validation

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'data')
    args=parser.parse_args()
    root=Path(__file__).parent;out=root/'results';out.mkdir(exist_ok=True)
    small=small_validation()
    (out/'small_validation.json').write_text(json.dumps(small,indent=2))
    reports=[]
    for i in range(3):
        name=f'ta{61+i}';inst=Instance.read(args.data/f'tai50_20_{i+1:02}.txt')
        lb=subset_lower_bound(inst)
        edges={tuple(sorted((int(a),int(b)))) for row in inst.machine for a,b in zip(row,row[1:])}
        r={'instance':name,'jobs':inst.n,'machines':inst.m,'lower_bound':lb,
           'job_chain_lower_bound':int(inst.p.sum(1).max()),
           'pair_variables_per_machine':inst.n*(inst.n-1)//2,
           'machine_quotient_edges':len(edges),'published_optimum':[2868,2869,2755][i]}
        ref=root/'reference'/f'{name}_published_order.json'
        if ref.exists():
            source=json.loads(ref.read_text())
            lookup={(v//inst.m,int(inst.machine.ravel()[v])):v for v in range(inst.N)}
            orders=[[lookup[j,k] for j in row] for k,row in enumerate(source['jobs_zero_based'])]
            result=analyze(inst,orders)
            assert result['feasible'] and result['makespan']==source['published_makespan']
            assert validate_starts(inst,result['starts'])==lb['value']
            export_schedule(inst,result,out/f'{name}_verified_optimal_schedule.csv')
            r.update(verified_upper_bound=result['makespan'],optimality_proved_locally=True,
                     upper_bound_source=source['source'])
        else:
            candidates=[]
            for mode in ['baseline','insertion']:
                pth=out/f'{name}_{mode}_order.txt'
                if not pth.exists():continue
                orders=[list(map(int,l.split())) for l in pth.read_text().splitlines() if l.strip()]
                result=analyze(inst,orders)
                assert result['feasible'] and validate_starts(inst,result['starts'])==result['makespan']
                run=json.loads((out/f'{name}_{mode}_run.json').read_text())
                assert run['makespan']==result['makespan']
                candidates.append((result['makespan'],mode,result))
            if candidates:
                val,mode,result=min(candidates,key=lambda x:x[0])
                export_schedule(inst,result,out/f'{name}_baseline_schedule.csv')
                r.update(verified_upper_bound=val,optimality_proved_locally=(val==lb['value']),
                         upper_bound_source=f'classical_{mode}',
                         relative_gap_to_lb=(val-lb['value'])/lb['value'])
            else:r.update(verified_upper_bound=None,optimality_proved_locally=False)
        reports.append(r)
        print(name,'LB',lb['value'],'verified UB',r['verified_upper_bound'],
              'optimality proved locally',r['optimality_proved_locally'])
    (out/'large_validation.json').write_text(json.dumps(reports,indent=2))
    print('Small exact model optimal makespan:',small['optimum'])

if __name__=='__main__':main()
