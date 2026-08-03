"""Compiler-safe entry point for :mod:`collect_fast_impl`.

The legacy capture object exports intermediates through a Python dictionary.
That works eagerly, but a compiled graph cannot retain autograd connections to
side-effect-only tensors.  This wrapper makes every captured pre/post tensor an
explicit graph output and reconstructs the same cache dictionary for the
unchanged collection code.
"""

from __future__ import annotations

import torch
from torch import nn

import collect_fast_impl as _impl
import geo67
from collection_runtime import compile_model


class _ExplicitCaptureModule(nn.Module):
    def __init__(self, target: nn.Module, capture: geo67.Capture):
        super().__init__()
        self.target = target
        self.capture = capture

    def forward(self, idx):
        self.capture.cache = {}
        logits = self.target(idx)
        pres = tuple(self.capture.cache[p]["pre"] for p in geo67.MODULES)
        posts = tuple(self.capture.cache[p]["post"] for p in geo67.MODULES)
        return (logits, *pres, *posts)


class _ExplicitCapture:
    def __init__(self, target, enabled, mode):
        self.capture = geo67.Capture(target)
        wrapper = _ExplicitCaptureModule(target, self.capture)
        self.target = compile_model(wrapper, enabled, mode)

    @property
    def wscale(self):
        return self.capture.wscale

    @wscale.setter
    def wscale(self, value):
        self.capture.wscale = value

    def run(self, idx):
        values = self.target(idx)
        count = len(geo67.MODULES)
        cache = {
            path: {"pre": values[1 + i], "post": values[1 + count + i]}
            for i, path in enumerate(geo67.MODULES)
        }
        return values[0], cache


def _setup_model(args, device):
    target = geo67.load_target(device)
    floating = {p.dtype for p in target.parameters() if p.is_floating_point()}
    if floating != {torch.float32}:
        raise RuntimeError(f"expected fp32 master parameters, found {floating}")
    if args.sensor == "gim":
        geo67.apply_gim(target, args.gim_tau)
    return _ExplicitCapture(target, args.compile, args.compile_mode)


_impl.setup_model = _setup_model

if __name__ == "__main__":
    _impl.main()
