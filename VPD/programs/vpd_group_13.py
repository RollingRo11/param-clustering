"""h.0.mlp.down_proj: at the first position, on unless the first token is punctuation / whitespace."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if p == 0 and not is_punct(s) else 0.0
