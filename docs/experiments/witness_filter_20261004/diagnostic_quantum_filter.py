"""Read-only D2 audit. Exact classical simulation, NOT QPU timing or speedup.
Run from repository root: python docs/experiments/witness_filter_20261004/diagnostic_quantum_filter.py
Frozen pools and source from acbc48c; rerun against that source version.
Truth tables are for evaluation only; neither training nor marking uses C(a).
"""
import ast
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'code/candidate_quantum_gates'), str(ROOT/'code/candidate_quantum_medium')]
import circuits
import search_loop as sl
import medium_qjsp as mq
from compact_simulator import CompactSimulator
from witness_surrogate import collect_witnesses
from window_diagnostics import check_choice

# Load the original proposer class unchanged without importing the unrelated
# OR-Tools benchmark module. Only its compact simulator and _train are used.
source = ROOT/'code/candidate_quantum_gates/final_benchmark.py'
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'QuantumProposer')
scope = dict(np=np, circuits=circuits, sl=sl, time=time,
             GAMMAS=(.05,.12,.25), BETAS=(.1,.25,.45))
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
QuantumProposer = scope['QuantumProposer']

def best_of(prob, energy, good, batch=64):
    """Exact probability: min-energy of b iid draws, first draw breaks ties."""
    answer = 0.
    for e in np.unique(energy):
        level = energy == e
        mass = float(prob[level].sum())
        if mass:
            ge = min(1.,float(prob[energy >= e].sum()))
            gt = min(1.,float(prob[energy > e].sum()))
            answer += float(prob[level & good].sum())/mass*(ge**batch-gt**batch)
    return float(answer)

def classical_product(pool,specs,T):
    """10 surrogate-only trials; no full candidate table required to train."""
    best_m=[np.ones(k)/k for k in pool.sizes]
    def energy(marginals):
        return sum(np.prod([marginals[m][list(a)].sum() for m,a in enumerate(s.allowed)])
                   for s in specs if s.kind=='cycle' or s.length>T)
    best_e=energy(best_m)
    for angles in itertools.product((.1,.25,.45),repeat=2):
        marginals=[]
        for k in pool.sizes:
            v=np.ones(k,dtype=complex)/np.sqrt(k)
            for b in angles:
                for i in range(k):
                    for j in range(i+1,k):
                        a,c=v[i],v[j]
                        v[i]=np.cos(b)*a-1j*np.sin(b)*c
                        v[j]=np.cos(b)*c-1j*np.sin(b)*a
            marginals.append(np.abs(v)**2)
        e=energy(marginals)
        if e < best_e-1e-12:
            best_e,best_m=e,marginals
    return best_m,float(best_e)

def main():
    start=time.perf_counter()
    data=json.loads((ROOT/'code/candidate_quantum_gates/results_final_benchmark_20261003/d2_confirmation.json').read_text())
    rows=[]
    for wi,w in enumerate(data['windows']):
        inst=mq.Instance.read(ROOT/'task_data'/f"{w['instance']}.txt")
        pool=w['production_hint']
        pobj=circuits.CandidatePool(tuple(tuple(tuple(v for v in order) for order in machine) for machine in pool))
        inc=(0,)*len(pool)
        raw,stats=collect_witnesses(inst,pool,inc,n_probe=16,seed=0)
        medium=[mq.Witness(x['kind'],tuple(tuple(r) for r in x['relations']),int(x['length']),()) for x in raw]
        specs=sl.graph_witness_specs(pobj,medium)
        labels=np.array(np.unravel_index(np.arange(np.prod(pobj.sizes)),pobj.sizes)).T
        costs=[]
        for a in labels:
            c=check_choice(inst,pool,a)
            costs.append(c['makespan'] if c['feasible'] else np.inf)
        costs=np.array(costs)
        assert costs[0]==w['u0'],(wi,costs[0],w['u0'])
        assert min(costs)==w['pool_optimum'],(wi,min(costs),w['pool_optimum'])
        n=len(costs)
        good=costs <= w['target']
        for tag,T in [('original_target_minus_1',w['target']-1),('corrected_target',w['target']),('operational_u0_minus_1',w['u0']-1)]:
            operational=tag=='operational_u0_minus_1'
            truth=(costs <= T) if operational else good
            sim=CompactSimulator(pobj,specs,(),T)
            proposer=QuantumProposer(inst,pool,specs,(),T,sampling='compact',seed=0)
            params=proposer._train()
            q=sim.probabilities(params,'xy')
            u=np.ones(n)/n
            # Remove phase while retaining identical mixer angles. This state
            # factorises across machines and is classically easy to sample.
            q0=sim.probabilities([(0.,b,j) for g,b,j in params],'xy')
            qprod=np.ones(n)
            for m,k in enumerate(pobj.sizes):
                marginal=np.bincount(labels[:,m],weights=q,minlength=k)
                qprod*=marginal[labels[:,m]]
            cm,ce=classical_product(pobj,specs,T)
            qc=np.ones(n)
            for m,marginal in enumerate(cm):
                qc*=marginal[labels[:,m]]
            assert np.isclose(qc@sim.energy,ce)
            wider=sim.train(mode='xy',budget=12,seed=0,layers=2)
            qw=sim.probabilities(wider['params'],'xy')
            if qw@sim.energy > u@sim.energy:
                qw=u.copy()  # pre-declared surrogate-only uniform rollback
            zero=sim.energy==0
            alpha=float(zero.mean())
            alpha_product=float(qc[zero].sum())
            pc=float(qc[truth].sum())
            # Exact one-step amplitude amplification around product-state
            # preparation; only valid as this formula when G is a subset of Z.
            product_aa1=pc*(3-4*alpha_product)**2 if not np.any(truth & ~zero) else None
            assert np.isclose(q.sum(),1)
            if tag!='original_target_minus_1':
                assert not np.any(truth & ~zero), 'a valid witness excluded a target solution'
            # A single Grover iteration using ONLY witness-zero predicate.
            state=np.ones(n,dtype=complex)/np.sqrt(n)
            state[zero]*=-1
            state=2*state.mean()-state
            grover=np.abs(state)**2
            assert np.isclose(grover.sum(),1)
            rows.append(dict(window=wi,instance=w['instance'],seed=w['seed'],iteration=w['trajectory_iteration'],strategy=w['strategy'],stratum=w['rho_stratum'],
                variant=tag,N=n,T=T,u0=w['u0'],goal_count=int(truth.sum()),zero_count=int(zero.sum()),alpha=alpha,
                goal_excluded=int((truth & ~zero).sum()),uniform1=float(u[truth].sum()),quantum1=float(q[truth].sum()),
                uniform64_filter=best_of(u,sim.energy,truth),quantum64_filter=best_of(q,sim.energy,truth),
                no_phase64_filter=best_of(q0,sim.energy,truth),product_marginals64_filter=best_of(qprod,sim.energy,truth),
                classical_product10_64_filter=best_of(qc,sim.energy,truth),classical_product10_1=float(qc[truth].sum()),
                classical_product_zero_mass=alpha_product,product_aa1=product_aa1,
                wider12_64_filter=best_of(qw,sim.energy,truth),wider12_1=float(qw[truth].sum()),
                quantum_zero_mass=float(q[zero].sum()),conditional_uniform_zero=float(truth[zero].mean()) if zero.any() else 0.,
                grover1=float(grover[truth].sum()),grover1_zero_mass=float(grover[zero].sum()),
                uniform64_all_verified=float(1-(1-truth.mean())**64),
                original_expected_energy=float(q@sim.energy),uniform_expected_energy=float(u@sim.energy),params=params))
        print(f'window {wi+1}/{len(data["windows"])} checked',flush=True)
    out=dict(commit='acbc48cf5005fe53127d15460b9bdbd0e769c40b',scope='exact classical diagnostic, frozen 32 selected D2 pools; no hardware or wall-clock advantage claim',seconds=time.perf_counter()-start,rows=rows)
    (OUTPUT_DIR/'diagnostic_quantum_filter_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    for tag in sorted(set(r['variant'] for r in rows)):
        rs=[r for r in rows if r['variant']==tag and r['goal_count']>0 and r['stratum']!='zero']
        print(tag,'positive windows',len(rs))
        for key in ['uniform1','quantum1','uniform64_filter','quantum64_filter','alpha','quantum_zero_mass','conditional_uniform_zero','grover1','uniform64_all_verified']:
            print(key,float(np.mean([r[key] for r in rs])))
        print('quantum64 wins',sum(r['quantum64_filter']>r['uniform64_filter']+1e-12 for r in rs))
        print('excluded goals',sum(r['goal_excluded'] for r in rs))

if __name__=='__main__': main()
