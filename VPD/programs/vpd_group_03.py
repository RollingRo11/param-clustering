"""h.0.mlp.c_fc: on at the end-of-text token (inside a context), nowhere else."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if s == EOT else 0.0
