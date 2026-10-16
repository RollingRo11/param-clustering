"""h.2.attn.k_proj: on at the first position of the context, whatever the token; off everywhere else."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if p == 0 else 0.0
