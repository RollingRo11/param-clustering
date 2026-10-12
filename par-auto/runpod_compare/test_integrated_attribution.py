"""Compare factor attribution with explicit independent component-gate gradients."""
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from integrated_attribution import capture_scaled, signed_scores, integrated_scores


def test_matches_explicit_gates_and_completeness():
    torch.manual_seed(7)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__(); self.wte=nn.Embedding(4,3); self.linear=nn.Linear(3,4,bias=False)
        def forward(self,x): return (self.linear(self.wte(x)),)
    model=Toy().cuda().double().eval().requires_grad_(False)
    p=SimpleNamespace(read=torch.eye(3,device='cuda',dtype=torch.float64),write=model.linear.weight.detach())
    rows=torch.tensor([[0,1,2,3],[1,3,2,0]],device='cuda')
    before={n:v.clone() for n,v in model.state_dict().items()}
    cache,loss=capture_scaled(model,rows,1.,('linear',),None)
    assert abs(loss-float(F.cross_entropy(model(rows[:,:-1])[0].flatten(0,1),rows[:,1:].flatten(),reduction='sum'))) < 1e-12
    nodes,weights=np.polynomial.legendre.leggauss(5); gold=torch.zeros(2,3,3,device='cuda',dtype=torch.float64)
    for alpha,weight in zip((nodes+1)/2,weights/2):
        gates=torch.full((2,3,3),float(alpha),device='cuda',dtype=torch.float64,requires_grad=True)
        h=model.wte(rows[:,:-1]); logits=((h @ p.read)*gates) @ p.write.T
        objective=F.cross_entropy(logits.flatten(0,1),rows[:,1:].flatten(),reduction='sum')
        expected=torch.autograd.grad(objective,gates)[0]
        cache,_=capture_scaled(model,rows,float(alpha),('linear',),None)
        torch.testing.assert_close(signed_scores(cache,[p],None)[0],expected,rtol=1e-10,atol=1e-12)
        gold+=float(weight)*expected
    actual,audit=integrated_scores(model,rows,[p],5,('linear',),None)
    torch.testing.assert_close(actual[0],gold.abs(),rtol=1e-10,atol=1e-12)
    full=F.cross_entropy(model(rows[:,:-1])[0].flatten(0,1),rows[:,1:].flatten(),reduction='sum')
    zero=F.cross_entropy(torch.zeros(6,4,device='cuda',dtype=torch.float64),rows[:,1:].flatten(),reduction='sum')
    assert abs(audit['signed_attribution_sum']-float(full-zero)) < 1e-7
    for name,value in model.state_dict().items(): torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert not model.linear._forward_hooks and not model.wte._forward_hooks


if __name__=='__main__':
    test_matches_explicit_gates_and_completeness()
    from train_pair import method, core
    import parauto_structured as structured
    torch.manual_seed(123)
    entry=dict(kind='scalar_ig',structured=True,h=torch.randn(8,5,3,device='cuda',dtype=torch.bfloat16),
               events={'g':torch.randn(8,5,4,device='cuda',dtype=torch.bfloat16)},
               shared={'quadrature_weights':[.2]*5})
    read=torch.randn(3,6,device='cuda',requires_grad=True)
    write=torch.randn(4,6,device='cuda',requires_grad=True)
    cfg=method.TrainConfig(precision='bf16')
    actual=method.training_scores(entry,core.FreePieces(read,write),cfg)
    actual_grad=torch.autograd.grad((actual+1e-6).sqrt().sum(),(read,write))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        expected=structured.integrated_scalar_scores(entry,read,write)
    expected_grad=torch.autograd.grad((expected+1e-6).sqrt().sum(),(read,write))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for a,b in zip(actual_grad,expected_grad): torch.testing.assert_close(a,b,rtol=0,atol=0)
    print('INTEGRATED_ATTRIBUTION_TEST_PASSED',flush=True)
