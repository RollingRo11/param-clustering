"""h.0.mlp.down_proj: at the first position, on when the first token is a word or number but not one of the most
common function words (the, to, of, a, and, with, ...)."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if p == 0 and not is_punct(s) and word(s) not in COMMON else 0.0
