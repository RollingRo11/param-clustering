"""h.0.mlp.c_fc: on at the word "of" in any case or spacing."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if word(s) == 'of' else 0.0
