"""h.1.attn.k_proj: on at the first position, and at positions 1-3 while the context opened with a content word
that is still being continued by word pieces."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if p == 0 or opening_run(prefix, 3) else 0.0
