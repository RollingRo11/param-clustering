import pytest
import torch
import parauto_core as core
import parauto_structured as structured
from profile_precision_sweep import projected_fp32, half_intermediate_sensitivity


def test_fp32_projection_and_backward_match_fp64_reference():
    if not torch.cuda.is_available(): pytest.skip('CUDA allocation required')
    torch.manual_seed(214)
    w = torch.randn(5, 11, device='cuda')
    p = core.initialize_pieces(w, 23)
    p.read.requires_grad_(); p.write.requires_grad_()
    fp32 = projected_fp32(w, p)
    reference = core.projected_pieces(w, p)
    assert core.reconstruction_error(w, fp32) < 1e-5
    grad32 = torch.autograd.grad(fp32.read.square().sum() + fp32.write.square().sum(), (p.read, p.write))
    grad64 = torch.autograd.grad(reference.read.square().sum() + reference.write.square().sum(), (p.read, p.write))
    for a, b in zip(grad32, grad64): torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_half_score_intermediates_stay_half_and_have_finite_gradients():
    if not torch.cuda.is_available(): pytest.skip('CUDA allocation required')
    torch.manual_seed(421)
    factor = torch.randn(8, 5, 5, device='cuda') * .1
    write = torch.randn(5, 23, device='cuda', requires_grad=True)
    entry = {'kind': 'input_qr', 'shared': {}, 'events': {'factor': factor.half()}}
    saved = []
    def pack(tensor):
        if tensor.shape == (8, 5, 23): saved.append(tensor.dtype)
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        with torch.autocast('cuda', dtype=torch.float16):
            actual = half_intermediate_sensitivity(entry, write)
    reference = structured.squared_sensitivity({**entry, 'events': {'factor': factor}}, write)
    torch.testing.assert_close(actual, reference, atol=1e-4, rtol=.006)
    actual.sum().backward()
    assert torch.isfinite(write.grad).all()
    assert saved and all(dtype == torch.float16 for dtype in saved)
