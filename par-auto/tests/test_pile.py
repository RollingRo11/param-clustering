import copy
import torch
import parauto_core as core
import parauto_method as method
import parauto_pile as pile


def tiny_model():
    torch.manual_seed(412)
    cfg = dict(vocab_size=11, n_layer=4, n_head=2, n_embd=8, n_intermediate=12,
               rotary_base=10000., rms_norm_eps=1e-6, n_ctx=6)
    return pile.target_source.Model67(cfg, 'plain').eval().requires_grad_(False)


def test_pile_vjp_matches_local_gate_derivative_through_attention():
    model = tiny_model(); ids = torch.tensor([[1, 3, 2, 7]])
    seed = 171
    entries, logits = pile.capture_probes(model, ids, 3, torch.Generator().manual_seed(seed))
    assert tuple(entries) == pile.MODULES
    direction = torch.randint(0, 2, logits.shape, dtype=torch.int8, generator=torch.Generator().manual_seed(seed)).float()
    direction = (direction*2-1) / (logits.shape[1]*logits.shape[2])**.5
    name, token = 'h.0.attn.k_proj', 1
    module = model.get_submodule(name)
    read, write = torch.randn(8), torch.randn(8)
    gate = torch.tensor(0., requires_grad=True)
    def intervene(_module, inputs, output):
        local = (inputs[0][0, token] @ read)*write
        position = torch.nn.functional.one_hot(torch.tensor(token), ids.shape[1]).float()
        return output + gate * position[None, :, None] * local
    hook = module.register_forward_hook(intervene)
    try:
        actual = torch.autograd.grad((model(ids)*direction).sum(), gate)[0]
    finally: hook.remove()
    entry = entries[name]
    predicted = (entry['h'][token].float()@read) * (entry['events']['cot'][token, 0].float()@write)
    torch.testing.assert_close(predicted, actual, atol=5e-4, rtol=.01)
    assert all(e['h'].dtype == e['events']['cot'].dtype == torch.bfloat16 for e in entries.values())


def test_pile_full_keep_restores_all_24_matrices_and_model_forward():
    model = tiny_model(); ids = torch.tensor([[1, 3, 2, 7]])
    parts = [core.initialize_pieces(model.get_submodule(n).weight, 16) for n in pile.MODULES]
    baseline = model(ids)
    with pile.masked_model(model, parts, [torch.ones(1, 4, 16) for _ in parts]):
        reconstructed = model(ids)
    torch.testing.assert_close(reconstructed, baseline, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(model(ids), baseline, atol=0, rtol=0)
    metrics = pile.evaluate(model, parts, torch.cat([ids, torch.tensor([[1]])], dim=-1), probes=2, budgets=[])
    assert metrics['budgets'][str(24*16)]['kl'] < 1e-10
    assert metrics['tokens'] == 4


def test_joint_resume_preserves_adam_rng_and_schedules():
    model = tiny_model(); ids = torch.tensor([[1, 3, 2, 7]])
    cache, _ = pile.capture_probes(model, ids, 3, torch.Generator().manual_seed(171))
    names = pile.MODULES[:2]
    weights = [model.get_submodule(n).weight for n in names]
    entries = [cache[n] for n in names]
    initial = [core.initialize_pieces(w, 16) for w in weights]
    cfg = method.TrainConfig(steps=6, batch=3, log_interval=1, projection_precision='fp32')
    full, _ = method.fit(entries, weights, initial, cfg, log=lambda row: None)
    snapshots = []
    method.fit(entries, weights, initial, cfg, log=lambda row: None, stop_after=2,
               checkpoint=lambda state: snapshots.append(copy.deepcopy(state)))
    resumed, _ = method.fit(entries, weights, initial, cfg, log=lambda row: None, resume=snapshots[-1])
    for a, b in zip(full, resumed):
        torch.testing.assert_close(a.read, b.read, atol=0, rtol=0)
        torch.testing.assert_close(a.write, b.write, atol=0, rtol=0)
