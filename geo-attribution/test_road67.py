import unittest

import torch


# Parfact implementations live in the sibling parfact directory.
import sys as _parfact_sys
from pathlib import Path as _ParfactPath
for _parfact_dir in (_ParfactPath(__file__).resolve().parent / "parfact",
                     _ParfactPath(__file__).resolve().parent.parent / "parfact"):
    if _parfact_dir.is_dir():
        _parfact_sys.path.insert(0, str(_parfact_dir))
        break

import parfact67 as legacy
import parfact67_streaming as stream
import road67


class Road67Tests(unittest.TestCase):
    def test_butterfly_roundtrip(self):
        layer = road67.Butterfly(8, 5, 3)
        with torch.no_grad():
            layer.angles.normal_(0, .3)
        value = torch.randn(2, 7, 8)
        recovered = layer(layer(value), transpose=True)
        self.assertLess(float((value - recovered).abs().max()), 2e-6)

    def test_routed_linear_all_on_is_exact(self):
        weight = torch.randn(12, 8)
        left, singular, right = torch.linalg.svd(weight, full_matrices=False)
        layer = road67.RoutedLinear(left, singular, right, 0, 16, 5, 7)
        with torch.no_grad():
            layer.rotation.angles.normal_(0, .2)
        value = torch.randn(2, 4, 8)
        layer.gates = torch.ones(2, 4, 16)
        actual = layer(value)
        expected = value @ weight.T
        self.assertLess(float((actual - expected).norm() / expected.norm()),
                        2e-6)

    def test_tiny_system_trains(self):
        cfg = {"vocab_size": 17, "n_layer": 4, "n_head": 2, "n_embd": 8,
               "n_intermediate": 16, "rotary_base": 10000.0,
               "rms_norm_eps": 1e-6, "n_ctx": 16}
        model = legacy.S67.Model67(cfg, "plain").eval()
        model.lm_head.weight.data.copy_(model.wte.weight.data)
        basis = stream.exact_svd_basis(model, "cpu")
        system = road67.RoadSystem(model, basis, 16, 3, 4, 1)
        ids = torch.randint(0, 17, (2, 16))
        with torch.no_grad():
            target, _ = road67.target_states(model, ids, False)
        logits = system.router(target[:-1])
        mask, _ = road67.topk_mask(logits, 4, .5)
        actual = system.forward_states(ids, mask)
        loss, values = road67.normalized_state_loss(actual, target, 1, 2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(values.shape, (1,))
        self.assertTrue(any(parameter.grad is not None for parameter in
                            system.parameters() if parameter.requires_grad))

    def test_global_ownership_is_exactly_balanced(self):
        cfg = {"vocab_size": 17, "n_layer": 4, "n_head": 2, "n_embd": 8,
               "n_intermediate": 16, "rotary_base": 10000.0,
               "rms_norm_eps": 1e-6, "n_ctx": 16}
        model = legacy.S67.Model67(cfg, "plain").eval()
        basis = stream.exact_svd_basis(model, "cpu")
        system = road67.RoadSystem(model, basis, 16, 3, 4, 1)
        owners = torch.cat(list(system.hard_owners().values()))
        counts = torch.bincount(owners, minlength=16)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)


if __name__ == "__main__":
    unittest.main()
