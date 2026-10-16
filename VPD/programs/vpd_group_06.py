"""h.0.mlp.c_fc: at the first position, on when the first token is a content word (not punctuation, digits or a
function word)."""
from token_kinds67 import *

READS = ()


def step(carry, prefix, upstream):
    p = position(carry); s = prefix.token_strings[-1]
    return p, 1.0 if p == 0 and is_content(s) else 0.0
