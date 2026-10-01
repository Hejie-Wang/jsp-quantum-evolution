"""Auditable JSP graph model and small-instance cut/QUBO verification.

No quantum hardware is used. The exact cut master deliberately enumerates
machine permutations and is only intended for small correctness experiments.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque, defaultdict
from itertools import combinations, permutations, product
from pathlib import Path
import csv
import json
from math import gcd
from functools import reduce
import numpy as np


@dataclass
class Instance:
    p: np.ndarray
    machine: np.ndarray

    def __post_init__(self):
        self.p = np.asarray(self.p, dtype=int)
        self.machine = np.asarray(self.machine, dtype=int)
        assert self.p.shape == self.machine.shape and self.p.ndim == 2
        self.n, self.m = self.p.shape
        assert np.all(self.p > 0)
        assert all(sorted(row) == list(range(self.m)) for row in self.machine)
        self.N = self.n * self.m
        self.d = self.p.ravel().tolist()
        self.groups = [[v for v in range(self.N)
                        if self.machine.ravel()[v] == k] for k in range(self.m)]
        self.job_arcs = [(j*self.m+k, j*self.m+k+1)
                         for j in range(self.n) for k in range(self.m-1)]

    @classmethod
    def read(cls, path):
        a = np.loadtxt(path, dtype=int, ndmin=2)
        if len(a) % 2:
            raise ValueError('Expected times followed by equally many machine rows')
        n = len(a)//2
        return cls(a[:n], a[n:]-1)


def analyze(inst, orders):
    """Topological longest paths, or one directed cycle; all arithmetic integer."""
    assert len(orders) == inst.m
    for actual, expected in zip(orders, inst.groups):
        assert sorted(actual) == expected
    machine_arcs = [(u,v) for row in orders for u,v in zip(row,row[1:])]
    adj = [[] for _ in range(inst.N)]
    deg = [0]*inst.N
    for u,v in inst.job_arcs + machine_arcs:
        adj[u].append(v)
        deg[v] += 1
    q = deque(v for v in range(inst.N) if deg[v] == 0)
    start, parent, topo = [0]*inst.N, [-1]*inst.N, []
    while q:
        u=q.popleft(); topo.append(u)
        for v in adj[u]:
            end=start[u]+inst.d[u]
            if end>start[v]: start[v],parent[v]=end,u
            deg[v]-=1
            if deg[v]==0: q.append(v)
    if len(topo)<inst.N:
        # An iterative DFS avoids recursion-depth assumptions for large inputs.
        color=[0]*inst.N; pred=[-1]*inst.N
        for root in range(inst.N):
            if color[root]: continue
            color[root]=1; stack=[(root,iter(adj[root]))]
            while stack:
                u,it=stack[-1]
                v=next(it,None)
                if v is None: color[u]=2;stack.pop();continue
                if color[v]==0:
                    color[v]=1;pred[v]=u;stack.append((v,iter(adj[v])))
                elif color[v]==1:
                    path=[u]
                    while path[-1]!=v: path.append(pred[path[-1]])
                    path.reverse();path.append(v)
                    return {'feasible':False,'cycle_edges':list(zip(path,path[1:]))}
        raise AssertionError('Cycle expected')
    last=max(range(inst.N),key=lambda v:start[v]+inst.d[v])
    path=[last]
    while parent[path[-1]]!=-1: path.append(parent[path[-1]])
    path.reverse()
    return {'feasible':True,'makespan':start[last]+inst.d[last],
            'starts':start,'path':path}


def validate_starts(inst, starts):
    """Independent direct non-overlap and job precedence check."""
    assert len(starts)==inst.N
    assert all(isinstance(s,(int,np.integer)) and s>=0 for s in starts)
    for u,v in inst.job_arcs: assert starts[u]+inst.d[u]<=starts[v]
    for group in inst.groups:
        row=sorted(group,key=lambda v:starts[v])
        for u,v in zip(row,row[1:]): assert starts[u]+inst.d[u]<=starts[v]
    return max(starts[v]+inst.d[v] for v in range(inst.N))


def subset_lower_bound(inst):
    """max(min head + sum durations + min tail) over machine subsets.

    Scanning sets {v: head[v]>=r, tail[v]>=q} attains the maximum over
    arbitrary nonempty subsets: their closure only adds positive durations.
    """
    head=inst.p.cumsum(1)-inst.p
    tail=inst.p.sum(1)[:,None]-inst.p.cumsum(1)
    best={'value':int(inst.p.sum(1).max()),'type':'job_chain'}
    for k,g in enumerate(inst.groups):
        pp=np.array([inst.d[v] for v in g]);rr=head.ravel()[g];qq=tail.ravel()[g]
        for r in np.unique(rr):
            for q in np.unique(qq):
                ix=(rr>=r)&(qq>=q)
                if not ix.any(): continue
                value=int(r+q+pp[ix].sum())
                if value>best['value']:
                    best={'value':value,'type':'machine_subset','machine_1_based':k+1,
                          'head':int(r),'tail':int(q),'processing_sum':int(pp[ix].sum()),
                          'operation_ids_zero_based':[g[i] for i in np.flatnonzero(ix)]}
    return best


def orientations(orders):
    return frozenset((u,v) for row in orders for u,v in combinations(row,2))


def separator(inst, result, horizon):
    if not result['feasible']:
        edges=result['cycle_edges'];kind='cycle'
    elif result['makespan']>horizon:
        path=result['path'];edges=list(zip(path,path[1:]));kind='path'
    else: return None
    fixed=set(inst.job_arcs)
    # Every returned pair means the literal "u precedes v on its machine".
    return kind, frozenset(e for e in edges if e not in fixed)


def all_orders(inst):
    return product(*(permutations(g) for g in inst.groups))


def exact_cut_decision(inst,horizon):
    """Finite lazy-cut master. Exponential; not a large-instance solver."""
    cuts=[];kinds=defaultdict(int);rounds=0
    while True:
        candidate=None
        for order in all_orders(inst):
            chosen=orientations(order)
            if all(not cut<=chosen for cut in cuts):
                candidate=order;break
        if candidate is None:
            return {'status':'INFEASIBLE','rounds':rounds,'cuts':dict(kinds)},cuts
        rounds+=1
        result=analyze(inst,candidate)
        violation=separator(inst,result,horizon)
        if violation is None:
            return {'status':'FEASIBLE','makespan':result['makespan'],
                    'rounds':rounds,'cuts':dict(kinds)},cuts
        kind,cut=violation
        if not cut:
            return {'status':'INFEASIBLE','rounds':rounds,'cuts':dict(kinds)},cuts
        assert cut not in cuts
        cuts.append(cut);kinds[kind]+=1


def machine_qubo(group, arc_prices):
    """Lagrangian linear-order QUBO in upper-triangular dictionary convention.

    E(x)=constant+sum(Q[i,j]*x[i]*x[j]), i<=j, with x_uv=1 iff u<v in order.
    A=sum(abs(linear coefficients))+1 forces transitivity at any global min.
    """
    pairs=list(combinations(sorted(group),2));index={p:i for i,p in enumerate(pairs)}
    linear=np.zeros(len(pairs));constant=0.0
    for (u,v),price in arc_prices.items():
        if u not in group or v not in group: continue
        if u<v: linear[index[u,v]]+=price
        else: constant+=price;linear[index[v,u]]-=price
    A=float(np.abs(linear).sum()+1)
    Q=defaultdict(float)
    for i,c in enumerate(linear):Q[i,i]+=float(c)
    for u,v,w in combinations(sorted(group),3):
        a,b,c=index[u,v],index[v,w],index[u,w]
        Q[min(a,b),max(a,b)]+=A
        Q[c,c]+=A
        Q[min(a,c),max(a,c)]-=A
        Q[min(b,c),max(b,c)]-=A
    return {'pairs':pairs,'constant':constant,'penalty':A,'Q':dict(Q)}


def qubo_energy(model,bits):
    return model['constant']+sum(c*bits[i]*bits[j] for (i,j),c in model['Q'].items())


def certify_cut_infeasibility(inst,cuts):
    """Small exact-pricing demonstration; LP suggests weights, integers certify.

    Only the final INTEGER sum of exact local minima is a certificate. LP status
    or a rounded floating point objective is never used as an infeasibility proof.
    The normalized dual contains every local permutation (small instances only).
    """
    from scipy.optimize import linprog
    K=len(cuts)
    rows=[]
    for k,g in enumerate(inst.groups):
        for order in permutations(g):
            arcs=frozenset(combinations(order,2))
            rows.append([-len(c & arcs) for c in cuts]+[int(z==k) for z in range(inst.m)])
    rows.append([1]*K+[0]*inst.m)
    objective=[len(c)-1 for c in cuts]+[-1]*inst.m
    result=linprog(objective,A_ub=rows,b_ub=[0]*(len(rows)-1)+[1],
                   bounds=[(0,None)]*K+[(None,None)]*inst.m,method='highs')
    if not result.success:return {'certified':False,'lp_status':result.message}
    weights=[max(0,round(float(x)*1000000)) for x in result.x[:K]]
    divisor=reduce(gcd,weights,0)
    if divisor:weights=[w//divisor for w in weights]
    prices=defaultdict(int)
    for weight,cut in zip(weights,cuts):
        for e in cut:prices[e]+=weight
    minima=[min(sum(prices[u,v] for u,v in combinations(row,2))
                for row in permutations(g)) for g in inst.groups]
    value=sum(w*(1-len(c)) for w,c in zip(weights,cuts))+sum(minima)
    return {'certified':bool(value>0),'integer_dual_value':value,
            'integer_multipliers':weights,'exact_machine_minima':minima}


def small_validation():
    inst=Instance([[3,2,2],[2,1,4],[4,3,1]],[[0,1,2],[1,2,0],[2,0,1]])
    schedules=[(o,analyze(inst,o)) for o in all_orders(inst)]
    optimum=min(r['makespan'] for _,r in schedules if r['feasible'])
    trace=[];cut_checks=0;qubo_checks=0;cycle_cut_checks=0
    for _,r in schedules:
        if not r['feasible']:
            _,cut=separator(inst,r,optimum)
            for order,other in schedules:
                if other['feasible']:
                    assert not cut<=orientations(order);cycle_cut_checks+=1
    for T in range(int(inst.p.sum(1).max())-1,optimum+2):
        status,cuts=exact_cut_decision(inst,T)
        assert (status['status']=='FEASIBLE')==(T>=optimum)
        for cut in cuts:
            for o,r in schedules:
                if r['feasible'] and r['makespan']<=T:
                    assert not cut<=orientations(o);cut_checks+=1
        # Integer multipliers yield integer certificates, without rounding error.
        prices=defaultdict(int)
        for i,c in enumerate(cuts):
            for edge in c:prices[edge]+=1+i%3
        minima=[]
        for group in inst.groups:
            model=machine_qubo(group,prices)
            qmin=min(qubo_energy(model,b) for b in product((0,1),repeat=len(model['pairs'])))
            pmin=min(sum(prices[u,v] for u,v in combinations(row,2)) for row in permutations(group))
            assert qmin==pmin;minima.append(qmin);qubo_checks+=1
        dual=sum((1+i%3)*(1-len(c)) for i,c in enumerate(cuts))+sum(minima)
        if T>=optimum:assert dual<=0
        certificate=certify_cut_infeasibility(inst,cuts)
        if certificate['certified']:assert T<optimum
        trace.append({'T':T,**status,'lagrangian_value_at_test_multipliers':dual,
                      'coordinated_integer_certificate':certificate})
    # Independent machine sequences can form a cross-machine cycle.
    two=Instance([[1,1],[1,1]],[[0,1],[1,0]])
    assert not analyze(two,[[3,0],[1,2]])['feasible']
    return {'instance':{'p':inst.p.tolist(),'machine_zero_based':inst.machine.tolist()},
            'enumerated_machine_order_combinations':len(schedules),'optimum':optimum,
            'globally_feasible_combinations':sum(r['feasible'] for _,r in schedules),
            'cut_validity_checks':cut_checks,'qubo_permutation_minimum_checks':qubo_checks,
            'cycle_cut_validity_checks':cycle_cut_checks,
            'cross_machine_cycle_counterexample_verified':True,'horizon_results':trace}


def export_schedule(inst, result, path):
    assert result['feasible']
    assert validate_starts(inst,result['starts'])==result['makespan']
    with Path(path).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['job','operation','machine','start','finish','duration'])
        for v,s in enumerate(result['starts']):
            j,k=divmod(v,inst.m)
            writer.writerow([j+1,k+1,int(inst.machine[j,k])+1,s,s+inst.d[v],inst.d[v]])


if __name__=='__main__':
    out=Path(__file__).parent/'results';out.mkdir(exist_ok=True)
    report=small_validation()
    (out/'small_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
