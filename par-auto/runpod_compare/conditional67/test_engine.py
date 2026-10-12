"""Small CPU checks for derivatives, conditional proposals, and exact resume."""
import copy
from dataclasses import replace
import io
import math
import unittest

import torch
from torch import nn
import torch.nn.functional as F

import engine as e


class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, n, nn.Linear(d, d, bias=False))

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = q @ k.transpose(-1, -2)/math.sqrt(x.shape[-1])
        mask = torch.ones(x.shape[-2], x.shape[-2], device=x.device).tril().bool()
        return self.o_proj(scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c_fc, self.down_proj = nn.Linear(d, d*2, bias=False), nn.Linear(d*2, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn, self.mlp = Attention(d), MLP(d)
        self.rms_1, self.rms_2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        x = x+self.attn(self.rms_1(x))
        return x+self.mlp(self.rms_2(x))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(11, 4)
        self.h = nn.ModuleList([Block(4), Block(4)])
        self.ln_f, self.lm_head = nn.LayerNorm(4), nn.Linear(4, 11, bias=False)

    def forward(self, ids):
        x = self.wte(ids)
        for block in self.h:
            x = block(x)
        return self.lm_head(self.ln_f(x)), None


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = Toy().eval().requires_grad_(False)
        self.pieces = [e.core.initialize_pieces(self.model.get_submodule(n).weight, 6, seed=0) for n in e.names(self.model)]
        self.rows = torch.randint(11, (2, 5))
        self.cfg = e.Config(steps=4, contexts=2, microbatch=1, precision='fp32', start_budget=48,
                            end_budget=24, anneal_steps=3, final_budgets=(24,), max_moves=4, probes=3)

    def test_all_on_and_restore(self):
        ids = self.rows[:, :-1]
        before = self.model(ids)[0]
        gate = torch.ones(2, 4, 72)
        torch.testing.assert_close(e.run(self.model, ids, self.pieces, gate, self.cfg), before, atol=2e-6, rtol=2e-6)
        try:
            with e.factor_forwards(self.model, self.pieces, gate):
                raise RuntimeError('intentional')
        except RuntimeError:
            pass
        torch.testing.assert_close(self.model(ids)[0], before, rtol=0, atol=0)

    def test_gate_derivative_and_batch_units(self):
        ids = self.rows[:, :-1]
        logp, support = e.dense_reference(self.model, ids, self.pieces, self.cfg, 3, 24)
        gate = support.float().requires_grad_()
        loss = e.kl_per_sequence(e.run(self.model, ids, self.pieces, gate, self.cfg), logp)
        g, = torch.autograd.grad(loss.sum(), gate)
        delta = torch.zeros_like(gate)
        chosen = g.abs().flatten().topk(5).indices
        delta.flatten()[chosen] = g.flatten()[chosen].sign()
        eps = .002
        plus = e.kl_per_sequence(e.run(self.model, ids, self.pieces, gate.detach()+eps*delta, self.cfg), logp).sum()
        minus = e.kl_per_sequence(e.run(self.model, ids, self.pieces, gate.detach()-eps*delta, self.cfg), logp).sum()
        torch.testing.assert_close((plus-minus)/(2*eps), (g*delta).sum(), rtol=.015, atol=2e-5)
        one = gate.detach()[:1].requires_grad_()
        g1, = torch.autograd.grad(e.kl_per_sequence(e.run(self.model, ids[:1], self.pieces, one, self.cfg), logp[:1]).sum(), one)
        torch.testing.assert_close(g[:1], g1, rtol=1e-5, atol=2e-6)

    def test_fisher_covariance(self):
        prob = torch.tensor([.2, .3, .5]).expand(60000, -1)
        v = e.fisher_seed(prob, torch.Generator().manual_seed(1))
        expected = torch.diag(prob[0])-torch.outer(prob[0], prob[0])
        torch.testing.assert_close(v.T @ v/len(v), expected, rtol=.03, atol=.004)

    def test_stable_kl_difference(self):
        # A tiny change on top of a large KL must survive the subtraction.
        p = torch.tensor([[[.8, .15, .05]]], dtype=torch.float64).log()
        minus = torch.tensor([[[-70., 0., -5.]]], dtype=torch.float64)
        plus = minus+torch.tensor([[[1e-6, 0., 0.]]], dtype=torch.float64)
        observed = e.stable_kl_difference(plus, minus, p)
        # Softmax mass at logit -70 is negligible: derivative is -p_0=-.8.
        torch.testing.assert_close(observed, torch.tensor([-8e-7], dtype=torch.float64), rtol=1e-6, atol=1e-13)
        naive = e.kl_per_sequence(plus.float(), p.float())-e.kl_per_sequence(minus.float(), p.float())
        self.assertEqual(float(naive), 0.)

    def test_joint_replacement_cancels_and_reduces_count(self):
        support = torch.tensor([[[True, True, False, False]]])
        # Retained effects 1+1 are exactly replaceable by absent effect 2.
        effects = torch.tensor([[[[1.], [1.], [2.], [5.]]]])
        grad = torch.zeros(1, 1, 4)
        indices = (torch.tensor([[[0, 1]]]), torch.tensor([[[2, 3]]]))
        proposal = e.candidate_tensors(grad, support, replace(self.cfg, rate=.1), indices=indices, effects=effects)
        result = e.apply_candidates(support, proposal, 1)
        self.assertEqual(result.tolist(), [[[False, False, True, False]]])
        self.assertAlmostEqual(float(proposal['cost']), -.1, places=6)

    def test_restoration_completeness_on_complementary_pair(self):
        # D=(1-ab)^2; at a=b=0 BOTH endpoint derivatives vanish.
        points = (torch.arange(1000, dtype=torch.float64)+.5)/1000
        a = points.clone().requires_grad_()
        b = points.clone().requires_grad_()
        ga, gb = torch.autograd.grad(((1-a*b)**2).sum(), (a, b))
        self.assertAlmostEqual(float(-(ga+gb).mean()), 1., places=5)

    def test_each_selector_checks_real_error(self):
        logp, initial = e.dense_reference(self.model, self.rows[:, :-1], self.pieces, self.cfg, 13, 24)
        baseline = e.objective(e.kl_per_sequence(e.run(self.model, self.rows[:, :-1], self.pieces, initial.float(), self.cfg), logp), initial, self.cfg.rate)
        for variant in e.VARIANTS:
            support, info = e.select(self.model, self.rows[:, :-1], self.pieces, initial, logp, self.cfg, variant, 14)
            value = e.objective(e.kl_per_sequence(e.run(self.model, self.rows[:, :-1], self.pieces, support.float(), self.cfg), logp), support, self.cfg.rate)
            self.assertTrue(bool((value <= baseline+1e-6).all()), variant)
            self.assertTrue(bool((support.sum(-1) <= initial.sum(-1)).all()))

    def test_training_resume_every_arm(self):
        for variant in e.VARIANTS:
            a = e.clone_pieces(self.pieces)
            opt = e.create_optimizer(a, self.cfg)
            e.train_step(self.model, self.rows, a, opt, self.cfg, variant, 0)
            saved = copy.deepcopy(dict(pieces=a, optimizer=opt.state_dict()))
            e.train_step(self.model, self.rows, a, opt, self.cfg, variant, 1)
            b = e.clone_pieces(saved['pieces'])
            bopt = e.create_optimizer(b, self.cfg)
            bopt.load_state_dict(saved['optimizer'])
            e.train_step(self.model, self.rows, b, bopt, self.cfg, variant, 1)
            for x, y, n in zip(a, b, e.names(self.model), strict=True):
                torch.testing.assert_close(x.read, y.read, atol=0, rtol=0)
                torch.testing.assert_close(x.write, y.write, atol=0, rtol=0)
                self.assertLess(e.core.reconstruction_error(self.model.get_submodule(n).weight, x), 1e-5)
            self.assertTrue(any(not torch.equal(x.read, y.read) for x, y in zip(a, self.pieces)))


if __name__ == '__main__':
    unittest.main()
