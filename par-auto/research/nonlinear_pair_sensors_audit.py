"""Small arithmetic audits of proposed sensors, not a model experiment.

Standard library only. Derivations live in nonlinear_pair_sensors.md.
"""
import json
import math
from pathlib import Path
import random


def dot(a, b): return sum(x*y for x, y in zip(a, b))
def add(a, b): return [x+y for x, y in zip(a, b)]
def sub(a, b): return [x-y for x, y in zip(a, b)]
def scale(a, c): return [x*c for x in a]
def total(xs): return [sum(v) for v in zip(*xs)]
def softmax(x):
    w = [math.exp(v-max(x)) for v in x]
    return scale(w, 1/sum(w))
def rms(x): return scale(x, 1/math.sqrt(dot(x,x)/len(x)+1e-6))
def gelu(x): return .5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+.044715*x**3)))


def main():
    rng = random.Random(20260915)
    vec = lambda n: [rng.uniform(-2,2) for _ in range(n)]
    errors = dict(pair_chain_rule=0., pair_finite_difference=0., key_finite=0.,
                  simultaneous_qk_deletion=0., residual_norm_correction=0.)
    for _ in range(1000):
        qs, ks = [vec(4) for _ in range(3)], [[vec(4) for _ in range(3)] for _ in range(5)]
        q, keys = total(qs), [total(k) for k in ks]
        logits = [dot(q,k)/2 for k in keys]
        p, b = softmax(logits), vec(5)
        gs = [pr*(br-dot(p,b)) for pr,br in zip(p,b)]
        q_gradient = total([scale(k,g/2) for k,g in zip(keys,gs)])
        pair_sum = sum(gs[s]*dot(qs[0],ks[s][j])/2 for s in range(5) for j in range(3))
        errors['pair_chain_rule'] = max(errors['pair_chain_rule'], abs(pair_sum-dot(qs[0],q_gradient)))
        eps = 1e-5
        f = lambda t: dot(softmax([dot(add(q,scale(qs[0],t)),k)/2 for k in keys]),b)
        fd = (f(eps)-f(-eps))/(2*eps)
        errors['pair_finite_difference'] = max(errors['pair_finite_difference'],abs(fd-pair_sum))
        # Remove one key component at one source position; change a single logit.
        s = rng.randrange(5); d = -dot(q,ks[s][0])/2
        patched = logits.copy(); patched[s] += d
        observed = dot(sub(softmax(patched),p),b)
        eta = p[s]*math.expm1(d)/(1+p[s]*math.expm1(d))
        expected = eta*(b[s]-dot(p,b))
        errors['key_finite'] = max(errors['key_finite'],abs(observed-expected))
        # Removing query and key requires adding back the double-counted pair.
        observed = dot(sub(q,qs[0]),sub(keys[s],ks[s][0]))/2-logits[s]
        expected = (-dot(qs[0],keys[s])-dot(q,ks[s][0])+dot(qs[0],ks[s][0]))/2
        errors['simultaneous_qk_deletion'] = max(errors['simultaneous_qk_deletion'],abs(observed-expected))
        # A residual node has both a bypass and a normalized consumer.
        r,z,bypass,consumer = vec(4),vec(4),vec(4),vec(4)
        inv = 1/math.sqrt(dot(r,r)/4+1e-6)
        jz = sub(scale(z,inv),scale(r,inv**3*dot(r,z)/4))
        # J is symmetric here (unit norm weights).
        jb = sub(scale(consumer,inv),scale(r,inv**3*dot(r,consumer)/4))
        before = add(bypass,jb)
        dn = sub(rms(sub(r,z)),rms(r))
        corrected = -dot(before,z)+dot(consumer,add(dn,jz))
        expected = -dot(bypass,z)+dot(consumer,dn)
        errors['residual_norm_correction'] = max(errors['residual_norm_correction'],abs(corrected-expected))
    assert errors['pair_finite_difference'] < 1e-7
    assert max(v for k,v in errors.items() if k!='pair_finite_difference') < 1e-10
    # A saturated softmax can make a large finite effect look negligible.
    p = softmax([12.,0.]); patched=softmax([0.,0.])
    linear = -12*p[0]*(1-p[0]); finite=patched[0]-p[0]
    assert abs(finite)>1000*abs(linear)
    # Exact first GELU is insufficient for downstream curvature.
    downstream_predicted = 0.  # derivative of h*h at h=GELU(0)=0
    downstream_actual = gelu(1.)**2
    assert downstream_actual>.7
    # Many nonzero pairs need not imply many active original components.
    # Nor does a large pairwise absolute sum imply any net effect.
    cancelling_pairs=[1.,-1.]
    assert sum(cancelling_pairs)==0 and sum(map(abs,cancelling_pairs))==2
    result=dict(cases=1000,max_errors=errors,
                saturated_softmax=dict(linear_effect=linear,finite_effect=finite),
                downstream_curvature=dict(prediction=downstream_predicted,actual=downstream_actual),
                note='Arithmetic checks only; no 67M model results for these new sensors.')
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
