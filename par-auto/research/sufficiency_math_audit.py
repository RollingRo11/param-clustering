"""Executable arithmetic checks for sparse-subset proofs; no model training.

Standard-library-only counterexamples and exhaustive finite cases. These checks
support the written derivations; numerical checks do not replace their proofs.
"""
import itertools
import json
import math
from pathlib import Path
import random


def lse(xs):
    m=max(xs);return m+math.log(sum(math.exp(x-m) for x in xs))


def kl(z,w):
    a=lse(z);b=lse(w)
    return sum(math.exp(x-a)*((x-a)-(y-b)) for x,y in zip(z,w))


def norm(x):return math.sqrt(sum(t*t for t in x))
def mv(a,x):return [sum(u*v for u,v in zip(row,x)) for row in a]
def mm(a,b):return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
def add(a,b):return [x+y for x,y in zip(a,b)]
def sub(a,b):return [x-y for x,y in zip(a,b)]
def frob(a):return math.sqrt(sum(x*x for row in a for x in row))
def matsum(ms):return [[sum(m[i][j] for m in ms) for j in range(len(ms[0][0]))] for i in range(len(ms[0]))]


def main():
    result={};atoms=[1.,10.,-10.];target=sum(atoms)
    singleton=[kl([target,0],[target-v,0]) for v in atoms]
    rank=sorted(range(3),key=lambda i:singleton[i],reverse=True)
    best=min(itertools.combinations(range(3),1),key=lambda ids:kl([target,0],[sum(atoms[i] for i in ids),0]))
    result['cancellation']=dict(atoms=atoms,individual_kl=singleton,ranked_one=rank[:1],optimal_one=list(best),
                               ranked_kl=kl([target,0],[atoms[rank[0]],0]),optimal_kl=kl([target,0],[atoms[best[0]],0]))
    assert result['cancellation']['optimal_kl']==0 and result['cancellation']['ranked_kl']>1
    # All intact and individual-ablation observations agree, joint deletion differs.
    plain=lambda m:1.
    interacting=lambda m:1.+10*(1-m[0])*(1-m[1])
    observed=[(1,1),(0,1),(1,0)]
    assert all(plain(m)==interacting(m) for m in observed)
    result['identical_singleton_observations']=dict(observed=[list(m) for m in observed],joint=[plain((0,0)),interacting((0,0))])
    # Budget sufficiency is non-monotone: restoring a +10 component destroys an exact subset.
    result['nonmonotone']=dict(exact_subset_kl=kl([1.,0.],[1.,0.]),superset_kl=kl([1.,0.],[11.,0.]))
    # Positive identical atoms: sparsity does not follow from positivity alone.
    N=1000;eps=.01
    needed=next(k for k in range(N+1) if kl([1.,0.],[k/N,0.])<=eps)
    result['positive_atoms_no_free_sparsity']=dict(N=N,tolerance=eps,minimum_unweighted_kept=needed,
                                                reweighted_atoms_needed=1)
    # Exact all-subset operator amplification example from the report.
    e=.01;M=1000.
    h=[1.,0.];changed=[1.,e];a=[[1.,0.],[0.,M]]
    output=mv(a,changed)
    result['upstream_error_amplification']=dict(local_first_error=e,local_second_error=0.,output_error=norm(sub(output,h)),
                                               selected_operator_norm=M,incorrect_original_operator_bound=e)
    assert result['upstream_error_amplification']['output_error']==10
    # Test both global KL inequalities over a wide logit and perturbation range.
    rng=random.Random(915);max_ratio_l2=max_ratio_range=0.
    for _ in range(10000):
        n=rng.randrange(2,20);z=[rng.uniform(-8,8) for _ in range(n)];d=[rng.uniform(-8,8) for _ in range(n)]
        measured=kl(z,add(z,d));center=sum(d)/n
        bound_l2=.25*sum((x-center)**2 for x in d)
        bound_range=(max(d)-min(d))**2/8
        assert measured<=bound_l2+1e-12 and measured<=bound_range+1e-12
        max_ratio_l2=max(max_ratio_l2,measured/bound_l2)
        max_ratio_range=max(max_ratio_range,measured/bound_range)
    result['kl_bounds']=dict(cases=10000,max_ratio_l2=max_ratio_l2,max_ratio_range=max_ratio_range)
    # Exhaustive binary subsets of two linear maps separated by tanh (L<=1).
    checks=0;max_ratio=0.
    for _ in range(50):
        pieces=[]
        for layer in range(2):
            pieces.append([[[rng.uniform(-2,2) for _ in range(2)] for _ in range(2)] for _ in range(3)])
        full=[matsum(p) for p in pieces];x=[rng.uniform(-1,1) for _ in range(2)]
        pre=mv(full[0],x);hidden=[math.tanh(v) for v in pre];answer=mv(full[1],hidden)
        for bits in itertools.product([0,1],repeat=6):
            sparse=[]
            for layer in range(2):
                ms=[pieces[layer][i] for i in range(3) if bits[layer*3+i]]
                sparse.append(matsum(ms) if ms else [[0.,0.],[0.,0.]])
            first_error=norm(sub(mv(sparse[0],x),pre))
            second_defect=norm(sub(mv(sparse[1],hidden),answer))
            bound=second_defect+frob(sparse[1])*first_error
            actual=norm(sub(mv(sparse[1],[math.tanh(v) for v in mv(sparse[0],x)]),answer))
            assert actual<=bound+1e-12
            if bound>1e-12:max_ratio=max(max_ratio,actual/bound)
            checks+=1
    result['finite_error_recursion']=dict(exhaustive_cases=checks,max_actual_bound_ratio=max_ratio)
    # Tanh-GELU derivative global bound 7/4 from the written analytic inequality.
    max_slope=0.
    for i in range(-100000,100001):
        x=i/10000.;a=math.sqrt(2/math.pi);b=.044715;t=a*(x+b*x*x*x);th=math.tanh(t)
        derivative=.5*(1+th)+.5*x*a*(1+3*b*x*x)*(1-th*th)
        max_slope=max(max_slope,abs(derivative));assert abs(derivative)<=1.75
    result['gelu_derivative_check']=dict(grid_points=200001,observed_max=max_slope,proved_upper=1.75)
    # Uniform exactly-d removed-set moment: shows why random deletion can prefer spreading.
    z=[[.6,1.],[.8,-.5],[-.4,-.5],[0.,2.]];C=len(z);d=2;q=d/C
    measured=sum(norm([sum(z[j][i] for j in inds) for i in range(2)])**2 for inds in itertools.combinations(range(C),d))/math.comb(C,d)
    total=[sum(v[i] for v in z) for i in range(2)];A=sum(norm(v)**2 for v in z);B=norm(total)**2
    formula=q*q*B+q*(1-q)/(C-1)*(C*A-B)
    assert abs(measured-formula)<1e-12
    result['uniform_deletion_moment']=dict(exact=measured,formula=formula)
    # Finite RMS-normalization bounds, including balls containing the origin.
    max_ratio=0.
    for _ in range(5000):
        n=rng.randrange(2,20);eps=10**rng.uniform(-6,-1)
        h=[rng.uniform(-2,2) for _ in range(n)];e=[rng.uniform(-2,2) for _ in range(n)]
        gamma=[rng.uniform(-2,2) for _ in range(n)];r=norm(e)
        def rms(x):
            scale=math.sqrt(sum(a*a for a in x)/n+eps)
            return [g*a/scale for g,a in zip(gamma,x)]
        bound=max(abs(g) for g in gamma)*r/math.sqrt(max(norm(h)-r,0)**2/n+eps)
        actual=norm(sub(rms(add(h,e)),rms(h)))
        assert actual<=bound+1e-12
        max_ratio=max(max_ratio,actual/bound)
    result['finite_rms_bound']=dict(cases=5000,max_actual_bound_ratio=max_ratio)
    # Full Q/K/V changes and causal row-softmax, not fixed attention routing.
    def transpose(a):return list(map(list,zip(*a)))
    def matrix_add(a,b):return [add(x,y) for x,y in zip(a,b)]
    def matrix_sub(a,b):return [sub(x,y) for x,y in zip(a,b)]
    def causal_attention(q,k):
        logits=mm(q,transpose(k));a=[]
        for t,row in enumerate(logits):
            allowed=[v/math.sqrt(2) for v in row[:t+1]];partition=lse(allowed)
            a.append([math.exp(v-partition) for v in allowed]+[0.]*(len(row)-t-1))
        return a
    max_ratio=0.
    for _ in range(1000):
        q,k,v,dq,dk,dv=[[[rng.uniform(-1,1) for _ in range(2)] for _ in range(3)] for _ in range(6)]
        rq,rk,rv=map(frob,[dq,dk,dv]);a=causal_attention(q,k)
        ra=.5*(frob(q)*rk+frob(k)*rq+rq*rk)/math.sqrt(2)
        bound=frob(a)*rv+frob(v)*ra+ra*rv
        ap=causal_attention(matrix_add(q,dq),matrix_add(k,dk))
        actual=frob(matrix_sub(mm(ap,matrix_add(v,dv)),mm(a,v)))
        assert actual<=bound+1e-12
        max_ratio=max(max_ratio,actual/bound)
    result['finite_causal_attention_bound']=dict(cases=1000,max_actual_bound_ratio=max_ratio)
    # Tail inequality and optimal continuous allocation formula.
    max_tail_ratio=max_allocation_ratio=0.
    for _ in range(2000):
        p=rng.uniform(.2,.8);s=(1-p)/p
        b=sorted([10**rng.uniform(-3,1) for _ in range(30)],reverse=True)
        k=rng.randrange(1,30);Bp=sum(x**p for x in b)**(1/p)
        bound=p/(1-p)*Bp*k**(-s);actual=sum(b[k:])
        assert actual<=bound+1e-10
        max_tail_ratio=max(max_tail_ratio,actual/bound)
        weights=[10**rng.uniform(-2,2) for _ in range(5)];K=50.
        mass=sum(a**p for a in weights);allocation=[K*a**p/mass for a in weights]
        exact=sum(a*k**(-s) for a,k in zip(weights,allocation))
        formula=mass**(s+1)/K**s
        assert abs(exact-formula)<=max(1e-10,formula*1e-12)
        arbitrary=[rng.uniform(.1,2) for _ in weights];scale=K/sum(arbitrary)
        candidate=sum(a*(scale*k)**(-s) for a,k in zip(weights,arbitrary))
        assert formula<=candidate+1e-10
        max_allocation_ratio=max(max_allocation_ratio,formula/candidate)
    result['tail_and_allocation']=dict(cases=2000,max_tail_bound_ratio=max_tail_ratio,
                                      max_optimal_arbitrary_allocation_ratio=max_allocation_ratio)
    path=Path(__file__).with_name('sufficiency_math_audit.json');path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
