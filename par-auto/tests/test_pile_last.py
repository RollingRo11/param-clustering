import torch
import torch.nn.functional as F
import parauto_core as core
import parauto_method as method
import parauto_pile_last as last
import parauto_structured as structured
from test_pile import tiny_model


def test_exact_last_scores_and_factor_gradients_match_full_logit_jacobian():
    model = tiny_model()
    ids = torch.tensor([[1, 3, 2, 7]])
    labels = torch.tensor([[3, 2, 7, 1]])
    r, error = last.head_factor(model, ids.shape[1])
    assert error < 1e-12
    caches, _ = last.capture(model, ids, labels, r, torch.float32)
    for i, name in enumerate(last.MODULES):
        module = model.get_submodule(name)
        def forward(delta):
            def intervene(_module, inputs, output):
                mask = F.one_hot(torch.tensor(1), ids.shape[1]).float()[None, :, None]
                return output + mask * delta
            hook = module.register_forward_hook(intervene)
            try: return model(ids)
            finally: hook.remove()
        jac = torch.autograd.functional.jacobian(forward, torch.zeros(module.weight.shape[0]))
        assert jac[0, [0, 2, 3]].abs().max() == 0  # no later attention
        p = core.initialize_pieces(module.weight, 16)
        p.read.requires_grad_(); p.write.requires_grad_()
        observed = last.scores(caches['output'][i], p)[1]
        j = jac.flatten(0, 2)
        expected = (caches['output'][i]['h'][1] @ p.read).abs() * (j @ p.write).square().mean(0).sqrt()
        torch.testing.assert_close(observed, expected, atol=2e-6, rtol=3e-5)
        actual_grad = torch.autograd.grad(observed.sum(), (p.read, p.write))
        ref_grad = torch.autograd.grad(expected.sum(), (p.read, p.write))
        for a, b in zip(actual_grad, ref_grad): torch.testing.assert_close(a, b, atol=5e-6, rtol=1e-4)


def test_last_scalar_cache_matches_true_cross_entropy_backward():
    model = tiny_model(); ids = torch.tensor([[1, 3, 2, 7]]); labels = torch.tensor([[3, 2, 7, 1]])
    r, _ = last.head_factor(model, 4)
    caches, _ = last.capture(model, ids, labels, r, torch.float32)
    posts, hooks = [], []
    for name in last.MODULES:
        hooks.append(model.get_submodule(name).register_forward_hook(lambda m, a, o: posts.append(o)))
    try:
        out = model(ids, embed=model.wte(ids).detach().requires_grad_())
        grad = torch.autograd.grad(F.cross_entropy(out.flatten(0, 1), labels.flatten(), reduction='sum'), posts)
    finally:
        for hook in hooks: hook.remove()
    for entry, g in zip(caches['scalar'], grad):
        torch.testing.assert_close(entry['events']['g'], g.flatten(0, 1), atol=1e-6, rtol=2e-5)


def test_nonlinear_last_tail_matches_full_model_with_joint_masks():
    model = tiny_model(); ids = torch.tensor([[1, 3, 2, 7]]); labels = ids
    r, _ = last.head_factor(model, 4)
    _, tail = last.capture(model, ids, labels, r, torch.float32)
    pieces = [core.initialize_pieces(model.get_submodule(n).weight, 16) for n in last.MODULES]
    for keep in (True, False):
        masks = [torch.ones(4, 16) if keep else torch.rand(4, 16) > .4 for _ in pieces]
        hooks = []
        for name, p, mask in zip(last.MODULES, pieces, masks):
            def replace(m, args, out, p=p, mask=mask): return core.masked_output(args[0], p, mask[None])
            hooks.append(model.get_submodule(name).register_forward_hook(replace))
        try: reference = model(ids).flatten(0, 1)
        finally:
            for h in hooks: h.remove()
        torch.testing.assert_close(last.tail_forward(model, tail, pieces, masks), reference, atol=2e-6, rtol=1e-5)
        if keep: torch.testing.assert_close(reference, tail['logits'], atol=2e-6, rtol=1e-5)


def test_last_exact_fit_preserves_projection_and_applies_updates():
    model = tiny_model(); ids = torch.tensor([[1, 3, 2, 7]])
    r, _ = last.head_factor(model, 4)
    caches, _ = last.capture(model, ids, ids, r)
    weights = [model.get_submodule(n).weight for n in last.MODULES]
    init = [core.initialize_pieces(w, 16) for w in weights]
    cfg = method.TrainConfig(steps=3, batch=4, log_interval=1, projection_precision='fp32')
    for sensor in ('output', 'scalar'):
        pieces, rows = method.fit(caches[sensor], weights, init, cfg, log=lambda _: None)
        assert rows[0][-1]['optimizer_updates'] == 3
        assert max(core.reconstruction_error(w, p) for w, p in zip(weights, pieces)) < 1e-5
