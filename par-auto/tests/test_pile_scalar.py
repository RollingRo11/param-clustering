import torch
import torch.nn.functional as F
import parauto_pile as pile
import parauto_pile_scalar as scalar
import parauto_pile_last as last
from test_pile import tiny_model


def test_full_scalar_backward_matches_local_early_gate_and_final_analytic_cache():
    model=tiny_model();ids=torch.tensor([[1,3,2,7]]);labels=torch.tensor([[3,2,7,1]])
    entries,_=scalar.capture(model,ids,labels,torch.float32)
    name='h.0.attn.q_proj';entry=entries[pile.MODULES.index(name)]
    read,write=torch.randn(8),torch.randn(8)
    gate=torch.tensor(0.,requires_grad=True)
    def intervene(module,args,out):
        delta=(args[0][0,1]@read)*write
        position=F.one_hot(torch.tensor(1),4).float()[None,:,None]
        return out+gate*position*delta
    hook=model.get_submodule(name).register_forward_hook(intervene)
    try:
        logits=model(ids)
        actual=torch.autograd.grad(F.cross_entropy(logits.flatten(0,1),labels.flatten(),reduction='sum'),gate)[0]
    finally:hook.remove()
    predicted=(entry['h'][1]@read)*(entry['events']['g'][1]@write)
    torch.testing.assert_close(predicted,actual,atol=1e-6,rtol=2e-5)
    r,_=last.head_factor(model,4)
    reference,_=last.capture(model,ids,labels,r,torch.float32)
    for name,ref in zip(last.MODULES,reference['scalar']):
        torch.testing.assert_close(entries[pile.MODULES.index(name)]['events']['g'],ref['events']['g'],atol=1e-6,rtol=2e-5)
