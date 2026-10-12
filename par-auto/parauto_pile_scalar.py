"""One true CE backward caches scalar sensitivities for all 24 LM matrices."""
import torch
import torch.nn.functional as F
import parauto_pile as pile


def capture(model, ids, labels, cache_dtype=torch.bfloat16):
    pre, post, hooks = {}, {}, []
    for name in pile.MODULES:
        def hook(module, inputs, output, name=name):
            pre[name], post[name] = inputs[0], output
        hooks.append(model.get_submodule(name).register_forward_hook(hook))
    try:
        with torch.enable_grad():
            out = model(ids, embed=model.wte(ids).detach().requires_grad_())
            loss = F.cross_entropy(out.flatten(0, 1), labels.flatten(), reduction='sum')
            grads = torch.autograd.grad(loss, [post[n] for n in pile.MODULES])
        entries = []
        for name, grad in zip(pile.MODULES, grads):
            h, g = pre[name].detach().flatten(0, 1).to(cache_dtype), grad.detach().flatten(0, 1).to(cache_dtype)
            if not (h.isfinite().all() and g.isfinite().all()): raise ValueError('Nonfinite scalar cache')
            entries.append(dict(structured=True, kind='scalar_loss', h=h, events={'g':g}, shared={},
                                net_rms=torch.zeros(len(h), device=h.device)))
        return entries, float(loss.detach()/labels.numel())
    finally:
        for hook in hooks: hook.remove()
