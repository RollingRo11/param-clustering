"""Compiler-safe entry point for the optimized collector.

Compilation is deliberately placed at each captured linear boundary.  AOT
Autograd does not preserve differentiation between sibling outputs of a whole-
model compiled graph; linear-level compilation keeps every cached output on the
ordinary eager autograd graph while compiling the dominant matrix kernels.
"""

from __future__ import annotations

import torch

import collect_fast_impl as _impl
import geo67


def _setup_model(args, device):
    target = geo67.load_target(device)
    floating = {p.dtype for p in target.parameters() if p.is_floating_point()}
    if floating != {torch.float32}:
        raise RuntimeError(f"expected fp32 master parameters, found {floating}")
    if args.sensor == "gim":
        geo67.apply_gim(target, args.gim_tau)
    capture = geo67.Capture(target)
    if args.compile:
        torch.set_float32_matmul_precision("high")
        for path in geo67.MODULES:
            linear = target.get_submodule(path)
            linear.forward = torch.compile(
                linear.forward, mode=args.compile_mode,
                fullgraph=False, dynamic=False)
    return capture


_impl.setup_model = _setup_model

if __name__ == "__main__":
    _impl.main()
